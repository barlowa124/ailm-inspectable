"""OpenAI-compatible chat endpoint backed by a LOCAL model we own.

Runs on the instrumented host (rented GPU box today, the desktop when
its PSU is back). Every completion returns the reply text plus an
internals receipt: per-token top-k candidates, per-layer attention
entropy at the final decode step, logit-lens argmax trajectory through
depth, and residual norms. The point is not the chat - it is seeing
what the model did to produce it.

This is the AILM answer to "the API is a black box": HF eager
execution, forward instrumentation, no streaming provider between you
and the weights. Same eager path the deconstruction evidence pins.

Usage:
    python3 ailm_inspectable.py --model Qwen/Qwen2.5-1.5B-Instruct \
        [--port 8000] [--device cuda]
    curl -X POST http://127.0.0.1:8000/v1/chat/completions \
        -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'

Endpoints:
    GET  /health                 -> model, device, dtype, caps
    GET  /v1/models              -> OpenAI model list (one entry)
    POST /v1/chat/completions    -> OpenAI response + "ailm" receipt

Env caps:
    AILM_MAX_NEW_TOKENS   default 256 - generation bound per call
    AILM_TOPK             default 5   - per-step candidate count
    AILM_REQ_TIMEOUT_S    n/a - GPU is serialized by a lock instead

Request flag (optional, in the JSON body):
    "ailm_attention": false      -> skip attention tensors entirely
                                  (faster; receipt still has logits
                                  and logit lens)

AILM_API_KEY (env): when set, every /v1/* request must carry
    Authorization: Bearer <key>. Set it when the service is exposed
    through a public tunnel (Colab + cloudflared); leave unset for
    loopback/SSH-tunnel deployments.
"""
import argparse
import hashlib
import json
import math
import os
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = int(os.environ.get("AILM_MAX_NEW_TOKENS", "256"))
TOPK = int(os.environ.get("AILM_TOPK", "5"))
MAX_CONTENT_CHARS = 120000
MAX_MESSAGES = 100

_MODEL = None
_TOK = None
_DEVICE = "cpu"
_MODEL_ID = ""
_gpu_lock = threading.Lock()


def _load(model_id, device, dtype):
    global _MODEL, _TOK, _DEVICE, _MODEL_ID
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[dtype]
    _TOK = AutoTokenizer.from_pretrained(model_id)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dt,
        attn_implementation="eager")   # deconstruction baseline: eager
    _MODEL.to(device).eval()
    _DEVICE, _MODEL_ID = device, model_id


def _topk_rows(scores, generated_ids):
    """Per generated step: the k highest-probability candidates with
    the chosen token's rank. scores: tuple(len T) of (batch, vocab)."""
    rows = []
    for step, (sc, tid) in enumerate(zip(scores, generated_ids)):
        logp = torch.log_softmax(sc[0].float(), dim=-1)
        topv, topi = logp.topk(TOPK)
        chosen_rank = int((logp > logp[tid]).sum().item())
        rows.append({
            "step": step,
            "token": _TOK.decode([tid]),
            "token_id": int(tid),
            "rank": chosen_rank,
            "top": [{"token": _TOK.decode([i]), "logp": round(float(v), 4)}
                    for v, i in zip(topv.tolist(), topi.tolist())],
        })
    return rows


def _logit_lens(hidden_last_step, model):
    """Final generated position's hidden state at every layer pushed
    through the head: when did the answer lock in, and what held the
    top slot before it. hidden_last_step: tuple(layers) of (1,seq,hid).
    """
    rows = []
    norm = model.model.norm          # final RMSNorm before lm_head
    head = model.lm_head
    n = len(hidden_last_step)
    for li, h in enumerate(hidden_last_step):
        with torch.no_grad():
            # stay in model dtype through norm+head (the real forward
            # path); float() only the resulting logits.
            # hidden_states[-1] is already post-final-norm; earlier
            # elements are pre-norm decoder outputs
            x = h[:, -1:, :] if li == n - 1 else norm(h[:, -1:, :])
            logits = head(x)
        logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        topv, topi = logp.topk(3)
        rows.append({"layer": li,
                     "top": [{"token": _TOK.decode([i]),
                              "logp": round(float(v), 3)}
                             for v, i in zip(topv.tolist(),
                                             topi.tolist())]})
    return rows


def _attention_stats(attns_last_step):
    """Per layer at the final step: mean head entropy and top-attended
    source positions - where the model was 'looking'. attns_last_step:
    tuple(layers) of (1, heads, 1, src_len)."""
    rows = []
    for li, a in enumerate(attns_last_step):
        # a: (1, heads, 1, src) -> per-head distribution over source
        p = a[0, :, -1, :].float()
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)  # per head
        flat = p.mean(dim=0)
        topv, topi = flat.topk(min(3, flat.numel()))
        rows.append({
            "layer": li,
            "heads": p.shape[0],
            "entropy_mean": round(float(ent.mean()), 4),
            "entropy_max": round(float(ent.max()), 4),
            "entropy_min": round(float(ent.min()), 4),
            "top_attended": [{"pos": int(i), "mass": round(float(v), 4)}
                             for v, i in zip(topv.tolist(),
                                             topi.tolist())]})
    return rows


def _lens_per_step(hidden_states, gen_ids, model):
    """Logit-lens argmax at EVERY decode step: for each generated token,
    the argmax of head(norm(hidden)) at that step's last position, per
    layer. Bounded to argmax only (no distributions) - this is the data
    convergence analysis needs."""
    norm = model.model.norm
    head = model.lm_head
    rows = []
    for step, (hs, tid) in enumerate(zip(hidden_states, gen_ids)):
        layer_tops = []
        for li, h in enumerate(hs):
            with torch.no_grad():
                # hidden_states[-1] is already post-final-norm in
                # recent transformers (norm applied before append);
                # earlier elements are pre-norm decoder outputs
                x = h[:, -1:, :] if li == len(hs) - 1 else norm(h[:, -1:, :])
                logits = head(x)
            layer_tops.append(
                _TOK.decode([int(logits[0, -1].argmax())]))
        # rank of the emitted token under the raw final-layer lens -
        # distinguishes "selected by a processor" (low rank) from
        # "identity never committed to residual" (high rank)
        with torch.no_grad():
            last_logits = head(hs[-1][:, -1:, :])[0, -1].float()
        order = last_logits.argsort(descending=True)
        emitted_rank = int((order == tid).nonzero()[0, 0])
        rows.append({"step": step, "token": _TOK.decode([tid]),
                     "emitted_rank_last": emitted_rank,
                     "argmax_by_layer": layer_tops})
    return rows


def _attn_per_step(attns, gen_ids):
    """Per generated step, per layer: top-3 attended source positions at
    the last query row. This is where positional binding shows up - the
    step that emits the answer token, not just the final step."""
    rows = []
    for step, (layers, tid) in enumerate(zip(attns, gen_ids)):
        layer_rows = []
        for li, a in enumerate(layers):
            p = a[0, :, -1, :].float()
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            mass, idx = p.max(dim=0)
            topv, topi = mass.topk(min(3, mass.numel()))
            hpos = int(idx[topi[0]].item())
            layer_rows.append({
                "layer": li,
                "top": [{"pos": int(topi[j]), "mass": round(float(topv[j]), 3),
                         "head": int(idx[topi[j]].item())}
                        for j in range(topv.numel())]})
        rows.append({"step": step, "token": _TOK.decode([tid]),
                     "layers": layer_rows})
    return rows


def _hidden_norms(hidden_last_step):
    """Residual stream L2 norm per layer at the final position - the
    cheap 'how active is this layer' summary."""
    return [{"layer": li, "norm": round(float(h[0, -1].float().norm()), 3)}
            for li, h in enumerate(hidden_last_step)]


def _jsonable(o):
    """Replace non-finite floats with None - bf16 logits and early-layer
    lens projections legitimately produce -inf/NaN, which strict JSON
    (allow_nan=False, Starlette's encoder) refuses to serialize."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _ablate_forward(targets, target_step, step_ctr, layer_of):
    """Replacement eager attention: identical math to the stock
    implementation but zeroes post-softmax weights for
    (layer, head, pos) targets at target_step's last query row, without
    renormalization - attenuates that head's output. Same intervention
    semantics as the pinned CPU ablation harness.

    Step semantics: multi-position forwards are prefill (target_step
    "prefill" hits the last prompt row); single-position forwards are
    decode steps indexed by generated token (step 0 = first emitted
    token)."""
    import torch.nn.functional as F

    def fwd(module, query, key, value, attention_mask=None,
            scaling=None, dropout=0.0, **kwargs):
        is_decode = query.shape[-2] == 1
        if is_decode and layer_of.get(id(module)) == 0:
            step_ctr["n"] += 1
        sc = scaling if scaling is not None \
            else getattr(module, "scaling", query.shape[-1] ** -0.5)
        g = getattr(module, "num_key_value_groups", 1)
        key_states = key.repeat_interleave(g, dim=1)
        value_states = value.repeat_interleave(g, dim=1)
        attn_weights = torch.matmul(
            query, key_states.transpose(2, 3)) * sc
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        spec = targets.get(layer_of.get(id(module)))
        cur_step = step_ctr["n"] if is_decode else "prefill"
        if spec and cur_step == target_step:
            for head, poss in spec:
                for pp in poss:
                    attn_weights[:, head, -1, pp] = 0
        attn_weights = F.dropout(
            attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights
    return fwd


class AblateSpecError(ValueError):
    """Malformed or out-of-range ailm_ablate spec."""


CAPTURE_TAGS = int(os.environ.get("AILM_CAPTURE_TAGS", "4"))
_CAPTURED = {}   # tag -> {"layers": {li: {"prefill": t, "decode": [t..]}}}


def _install_capture(tag, layers_sel):
    """Hook decoder layers, storing each forward's hidden output so a
    later request can patch it back in (causal tracing). Keeps the
    last CAPTURE_TAGS runs."""
    store = {"layers": {li: {"prefill": None, "decode": []}
                        for li in layers_sel}}
    handles = []

    def hook_for(li):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            rec = store["layers"][li]
            if hs.shape[1] == 1:
                rec["decode"].append(hs.detach().clone())
            else:
                rec["prefill"] = hs.detach().clone()
        return hook

    for li in layers_sel:
        handles.append(
            _MODEL.model.layers[li].register_forward_hook(hook_for(li)))
    _CAPTURED[tag] = store
    while len(_CAPTURED) > CAPTURE_TAGS:
        _CAPTURED.pop(next(iter(_CAPTURED)))

    def restore():
        for h in handles:
            h.remove()
    return restore


def _validate_capture(cap, n_layers):
    if not isinstance(cap, dict):
        raise AblateSpecError("ailm_capture must be an object")
    tag = cap.get("tag")
    if not isinstance(tag, str) or not tag:
        raise AblateSpecError("ailm_capture needs a nonempty 'tag'")
    layers = cap.get("layers", "all")
    if layers != "all" and (
            not isinstance(layers, list) or not layers or any(
                not isinstance(li, int) or isinstance(li, bool)
                or not 0 <= li < n_layers for li in layers)):
        raise AblateSpecError(
            f"capture layers must be 'all' or ints 0..{n_layers - 1}")
    return tag, list(range(n_layers)) if layers == "all" else layers


def _validate_patch(patch, n_layers, step_bound):
    """Per-target: {layer, pos, step}. pos rules match resid targets -
    prompt rows at prefill, 'current' at a decode step."""
    if not isinstance(patch, dict):
        raise AblateSpecError("ailm_patch must be an object")
    tag = patch.get("tag")
    if not isinstance(tag, str) or not tag:
        raise AblateSpecError("ailm_patch needs a nonempty 'tag'")
    targets = patch.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AblateSpecError("patch targets must be a nonempty array")
    for t in targets:
        if not isinstance(t, dict):
            raise AblateSpecError("patch target must be an object")
        layer, pos, step = t.get("layer"), t.get("pos"), t.get("step")
        if not isinstance(layer, int) or not 0 <= layer < n_layers:
            raise AblateSpecError(f"layer must be 0..{n_layers - 1}")
        if step == "prefill":
            poss = pos if isinstance(pos, list) else [pos]
            if pos not in ("all", "current"):
                if not poss or any(not isinstance(p, int)
                                   for p in poss):
                    raise AblateSpecError(
                        "prefill pos must be int, 'all', or 'current'")
                if any(p < -1 for p in poss):
                    raise AblateSpecError(
                        "prefill pos may only use -1 as the last-row "
                        "sentinel")
        elif not isinstance(step, int) or not 0 <= step < step_bound:
            raise AblateSpecError(
                f"step must be 'prefill' or 0..{step_bound - 1}")
        elif pos not in ("current", "all") and pos not in (0, -1):
            raise AblateSpecError(
                "patch pos at a decode step must be 'current'")
    return tag, targets


def _patch_from_run(tag, targets, cur_prompt_len):
    """Overwrite hidden rows with a captured run's values. Checks
    against the stored shapes so a mismatched prompt or missing step
    fails at install, not mid-forward."""
    run = _CAPTURED.get(tag)
    if run is None:
        raise AblateSpecError(f"no captured run '{tag}'")
    by_layer = {}
    for t in targets:
        li = int(t["layer"])
        rec = run["layers"].get(li)
        if rec is None:
            raise AblateSpecError(
                f"layer {li} was not captured in run '{tag}'")
        step = t["step"]
        if step == "prefill":
            width = rec["prefill"].shape[1] if rec["prefill"] is not None \
                else 0
            pos = t["pos"]
            if pos == "all" and width != cur_prompt_len:
                raise AblateSpecError(
                    f"pos 'all' needs matching prompt length "
                    f"(captured {width}, current {cur_prompt_len})")
            poss = pos if isinstance(pos, list) else [pos]
            if pos not in ("all", "current") and any(
                    p >= width or p < -width for p in poss):
                raise AblateSpecError(
                    f"pos outside captured prompt width {width}")
        elif int(step) >= len(rec["decode"]):
            raise AblateSpecError(
                f"captured run '{tag}' has no decode step {step}")
        by_layer.setdefault(li, []).append((step, t["pos"]))

    handles = []

    def hook_for(li, entries):
        ctr = {"n": -1}

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            is_decode = hs.shape[1] == 1
            if is_decode:
                ctr["n"] += 1
            cur = ctr["n"] if is_decode else "prefill"
            hits = [p for s, p in entries if s == cur]
            rec = run["layers"][li]
            src = rec["prefill"] if cur == "prefill" else (
                rec["decode"][cur] if cur < len(rec["decode"])
                else None)
            if not hits or src is None:
                return None
            hs2 = hs.clone()
            for p in hits:
                poss = p if isinstance(p, list) else [p]
                for pp in poss:
                    if pp == "all":
                        hs2[:, :, :] = src[:, :, :]
                    elif pp == "current" or is_decode:
                        hs2[:, -1, :] = src[:, -1, :]
                    else:
                        hs2[:, int(pp), :] = src[:, int(pp), :]
            if isinstance(output, tuple):
                return (hs2,) + output[1:]
            return hs2
        return hook

    for li, entries in by_layer.items():
        handles.append(
            _MODEL.model.layers[li].register_forward_hook(
                hook_for(li, entries)))

    def restore():
        for h in handles:
            h.remove()
    return restore


def _validate_ablate(ablate, n_layers, n_heads, prompt_len, step_bound):
    """Reject specs that would silently no-op. An ablation that never
    fires returns a normal receipt - indistinguishable from 'attention
    was decorative' - so every index must be in range before generate.
    Position bounds are step-relative: the attention row holds
    prompt_len positions at prefill and prompt_len + step at decode
    step `step` - anything beyond crashes inside the forward."""
    if not isinstance(ablate, dict):
        raise AblateSpecError("ailm_ablate must be an object")
    step = ablate.get("step", 0)
    if step != "prefill":
        if not isinstance(step, int) or isinstance(step, bool) \
                or not 0 <= step < step_bound:
            raise AblateSpecError(
                f"step must be 'prefill' or 0..{step_bound - 1}")
    seq_bound = prompt_len if step == "prefill" else prompt_len + step
    targets = ablate.get("targets")
    if not isinstance(targets, list) or not targets:
        raise AblateSpecError("targets must be a nonempty array")
    for t in targets:
        if not isinstance(t, dict):
            raise AblateSpecError("target must be an object")
        kind = t.get("kind", "attn")
        if kind not in ("attn", "resid"):
            raise AblateSpecError("kind must be 'attn' or 'resid'")
        layer, head, pos = t.get("layer"), t.get("head"), t.get("pos")
        if not isinstance(layer, int) or not 0 <= layer < n_layers:
            raise AblateSpecError(f"layer must be 0..{n_layers - 1}")
        if kind == "resid":
            pass  # resid targets take layer+pos; head is ignored
        elif head == "all":
            pass
        elif isinstance(head, list) and head:
            if any(not isinstance(h, int) or not 0 <= h < n_heads
                   for h in head):
                raise AblateSpecError(f"head must be 0..{n_heads - 1}")
        elif not isinstance(head, int) or not 0 <= head < n_heads:
            raise AblateSpecError(f"head must be 0..{n_heads - 1}")
        poss = pos if isinstance(pos, list) else [pos]
        special = ("all", "current") if kind == "resid" else ("all",)
        if kind == "resid" and step != "prefill" \
                and pos not in special and pos not in (0, -1):
            raise AblateSpecError(
                "resid pos at a decode step must be 'current' "
                "(kv-cache carries only the last position)")
        if pos not in special and (not poss or any(
                not isinstance(p, int) or not 0 <= p < seq_bound
                for p in poss)):
            raise AblateSpecError(f"pos must be 0..{seq_bound - 1} "
                                  "or 'all'")


def _patch_resid(resid, target_step):
    """Zero residual-stream rows at chosen layers for one step.
    resid: {layer: [pos|...]}, pos = int prompt row (prefill), 'all',
    or 'current'. Decode forwards carry only the last position (kv
    cache) so decode targets always hit the current token's row.
    Positional causal test attention ablation cannot give: does the
    information at pos P's residual matter to the answer?"""
    ctr = {"n": -1}
    handles = []

    def hook_for(li):
        poss = resid.get(li)

        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            is_decode = hs.shape[1] == 1
            if li == 0 and is_decode:
                ctr["n"] += 1
            if poss is None:
                return None
            cur = ctr["n"] if is_decode else "prefill"
            if cur != target_step:
                return None
            hs2 = hs.clone()
            for p in poss:
                if p == "all":
                    hs2[:, :, :] = 0
                elif p == "current" or is_decode:
                    hs2[:, -1, :] = 0
                else:
                    hs2[:, int(p), :] = 0
            if isinstance(output, tuple):
                return (hs2,) + output[1:]
            return hs2
        return hook

    for li in sorted(set(resid) | {0}):   # layer 0 counts steps
        if li >= len(_MODEL.model.layers):
            continue
        handles.append(
            _MODEL.model.layers[li].register_forward_hook(hook_for(li)))

    def restore():
        for h in handles:
            h.remove()
    return restore


def _patch_attention(targets, target_step):
    """Install the ablated eager forward everywhere it may be resolved:
    the shared registry plus any modeling module that bound the name at
    import time. Returns a restore() callable."""
    import sys as _sys
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    layer_of = {}
    for li, layer in enumerate(_MODEL.model.layers):
        layer_of[id(layer.self_attn)] = li
    ctr = {"n": -1}
    new_fwd = _ablate_forward(targets, target_step, ctr, layer_of)
    orig = {}
    if "eager" in ALL_ATTENTION_FUNCTIONS:
        orig["__dict__"] = ALL_ATTENTION_FUNCTIONS["eager"]
        ALL_ATTENTION_FUNCTIONS["eager"] = new_fwd
    for name, m in list(_sys.modules.items()):
        f = getattr(m, "eager_attention_forward", None)
        if f is not None and name.startswith("transformers.models"):
            orig[name] = f
            m.eager_attention_forward = new_fwd

    def restore():
        for name, f in orig.items():
            if name == "__dict__":
                ALL_ATTENTION_FUNCTIONS["eager"] = f
            else:
                _sys.modules[name].eager_attention_forward = f
    return restore


def chat_complete(messages, max_new_tokens, temperature, want_attn,
                  ablate=None, capture=None, patch=None):
    prompt = _TOK.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOK(prompt, return_tensors="pt").to(_DEVICE)
    gen_kwargs = dict(
        max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS),
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=None if temperature <= 0 else 0.95,
        # neutralize generation_config defaults (Qwen ships rep. penalty
        # 1.1) - the receipt must reflect raw network computation, not
        # the model card's deployment-time sampling preferences
        repetition_penalty=1.0,
        top_k=None,
        output_scores=True,
        output_hidden_states=True,
        output_attentions=want_attn,
        return_dict_in_generate=True,
        pad_token_id=_TOK.eos_token_id,
    )
    restores = []
    if ablate:
        # spec: {"step": int|"prefill",
        #        "targets": [{layer, head, pos:int|"all"|[..],
        #                    kind:"attn"|"resid"}]}
        # step = generated-token index (0 = first emitted token);
        # "prefill" = last query row of the prompt forward.
        cfg = _MODEL.config
        prompt_len = inputs.input_ids.shape[1]
        _validate_ablate(
            ablate, cfg.num_hidden_layers, cfg.num_attention_heads,
            prompt_len, min(max_new_tokens, MAX_NEW_TOKENS))
        step = ablate.get("step", 0)
        row_len = prompt_len if step == "prefill" \
            else prompt_len + int(step)
        targets = {}
        resid = {}
        n_heads = _MODEL.config.num_attention_heads
        for t in ablate.get("targets", []):
            pos = t["pos"]
            if t.get("kind", "attn") == "resid":
                resid.setdefault(int(t["layer"]), []).extend(
                    pos if isinstance(pos, list) else [pos])
                continue
            poss = list(range(row_len)) \
                if pos == "all" else (pos if isinstance(pos, list) else [pos])
            heads = range(n_heads) if t["head"] == "all" \
                else (t["head"] if isinstance(t["head"], list)
                      else [t["head"]])
            for h in heads:
                targets.setdefault(int(t["layer"]), []).append(
                    (int(h), poss))
        step = step if step == "prefill" else int(step)
        if targets:
            restores.append(_patch_attention(targets, step))
        if resid:
            restores.append(_patch_resid(resid, step))
    if capture:
        cap_tag, cap_layers = _validate_capture(
            capture, _MODEL.config.num_hidden_layers)
        restores.append(_install_capture(cap_tag, cap_layers))
    if patch:
        p_tag, p_targets = _validate_patch(
            patch, _MODEL.config.num_hidden_layers,
            min(max_new_tokens, MAX_NEW_TOKENS))
        restores.append(_patch_from_run(
            p_tag, p_targets, inputs.input_ids.shape[1]))
    restore = (lambda: [r() for r in restores]) if restores else None
    try:
        with _gpu_lock, torch.no_grad():
            t0 = time.time()
            out = _MODEL.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0
    finally:
        if restore:
            restore()
    seq = out.sequences[0]
    gen_ids = seq[inputs.input_ids.shape[1]:]
    text = _TOK.decode(gen_ids, skip_special_tokens=True)

    # internals ------------------------------------------------------------
    scores = out.scores                       # per-step final logits
    hidden = out.hidden_states                # per-step tuple(layers)
    last_hidden = hidden[-1]                  # final position, all layers
    internals = {
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "prompt_decoded": [_TOK.decode([t])
                           for t in inputs.input_ids[0].tolist()],
        "generated_tokens": int(gen_ids.numel()),
        "attn_steps": _attn_per_step(out.attentions, gen_ids.tolist())
        if want_attn else None,
        "elapsed_s": round(elapsed, 2),
        "per_token": _topk_rows(scores, gen_ids.tolist()),
        "logit_lens": _logit_lens(last_hidden, _MODEL),
        "lens_steps": _lens_per_step(hidden, gen_ids.tolist(), _MODEL),
        "residual_norms": _hidden_norms(last_hidden),
        "ablate": ablate,
        "capture": capture,
        "patch": patch,
    }
    if want_attn and out.attentions:
        internals["attention"] = _attention_stats(out.attentions[-1])
    internals = _jsonable(internals)
    internals["internals_sha256"] = hashlib.sha256(
        json.dumps(internals, sort_keys=True).encode()).hexdigest()
    return text, internals


app = FastAPI(title="ailm-inspectable")

API_KEY = os.environ.get("AILM_API_KEY")


@app.middleware("http")
async def _auth(req: Request, call_next):
    if API_KEY and req.url.path.startswith("/v1/"):
        if req.headers.get("authorization") != f"Bearer {API_KEY}":
            return JSONResponse({"error": {"message":
                                         "missing or bad bearer"}},
                                status_code=401)
    return await call_next(req)


@app.get("/health")
def health():
    return {"ok": True, "model": _MODEL_ID, "device": _DEVICE,
            "attn": "eager",
            "caps": {"max_new_tokens": MAX_NEW_TOKENS, "topk": TOPK}}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{
        "id": _MODEL_ID, "object": "model", "created": 0,
        "owned_by": "ailm-inspectable"}]}


@app.post("/v1/chat/completions")
async def completions(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": {"message": "malformed JSON"}},
                            status_code=400)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": {
            "message": "messages must be a nonempty array"}},
            status_code=400)
    if len(messages) > MAX_MESSAGES:
        return JSONResponse({"error": {
            "message": "too many messages"}}, status_code=400)
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
           for m in messages):
        return JSONResponse({"error": {
            "message": "each message needs role + text content"}},
            status_code=400)
    if sum(len(m["content"]) for m in messages) > MAX_CONTENT_CHARS:
        return JSONResponse({"error": {
            "message": "messages exceed char cap"}}, status_code=400)

    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens")
                  or MAX_NEW_TOKENS)
    temperature = float(body.get("temperature", 0.7))
    want_attn = body.get("ailm_attention", True) is not False
    ablate = body.get("ailm_ablate")
    if ablate is not None and not (
            isinstance(ablate, dict)
            and isinstance(ablate.get("targets"), list)):
        return JSONResponse({"error": {
            "message": "ailm_ablate needs {step, targets:[{layer,head,"
                       "pos}]}"}}, status_code=400)
    try:
        text, internals = chat_complete(
            messages, max_new, temperature, want_attn, ablate,
            capture=body.get("ailm_capture"),
            patch=body.get("ailm_patch"))
    except AblateSpecError as exc:
        return JSONResponse({"error": {"message":
                                     f"spec rejected: {exc}"}},
                            status_code=400)
    except Exception as exc:  # GPU OOM, tokenizer, template
        return JSONResponse({"error": {
            "message": f"inspectable backend failed: {exc}"}},
            status_code=502)

    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": text}}],
            "usage": {"prompt_tokens": internals["prompt_tokens"],
                      "completion_tokens": internals["generated_tokens"],
                      "total_tokens": internals["prompt_tokens"]
                      + internals["generated_tokens"]},
            "ailm": {"provider": "inspectable",
                     "model": _MODEL_ID,
                     "device": _DEVICE,
                     "attn": "eager",
                     "internals": internals}}


def main():
    ap = argparse.ArgumentParser(description="instrumented local-model "
                                           "chat service")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    _load(args.model, args.device, args.dtype)
    print(f"inspectable on 127.0.0.1:{args.port} "
          f"model={args.model} dev={args.device}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
