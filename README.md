# ailm_inspectable

An OpenAI-compatible chat endpoint backed by a local model, instrumented
so every completion returns an internals receipt alongside the reply
text. The receipt shows per-token top-k candidates plus a logit-lens
argmax trajectory through depth. Per-layer attention entropy and
residual norms round it out.

The chat is the carrier. The goal is seeing what the model did to
produce each token, on the same eager execution path used for the
activation-patching and ablation work this service supports.

## What the receipt carries

- `per_token`: each generated step's top-k candidates and the chosen
  token's rank.
- `logit_lens`: the argmax of `head(norm(hidden))` at every layer and
  every decode step, so convergence is visible layer by layer.
- `attention`: per-layer head entropy plus the most-attended source
  positions, per step.
- `hidden_norms`: residual L2 norm per layer at the final position.
- Optional interventions: `ailm_attention:false` skips attention
  tensors. Ablation and residual-patching specs install eager-forward
  or hidden-state hooks for causal probing, and runs can be captured
  for later patching.

## Run it

Local:

```bash
python3 ailm_inspectable.py --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}]}'
```

Colab GPU (a free T4 runs the 1.5B default): open
`ailm_inspectable_colab.ipynb` in Colab and run the cells top to bottom.
The notebook picks the model by GPU memory and fetches this file. It
then starts the service and exposes it through a cloudflared quick
tunnel.
Set `AILM_API_KEY` before exposing it so `/v1/*` requires a bearer
token.

## Endpoints

- `GET /health`: model, device, dtype, caps.
- `GET /v1/models`: the loaded model.
- `POST /v1/chat/completions`: OpenAI response plus an `ailm` receipt.

## Env caps

- `AILM_MAX_NEW_TOKENS`: generation bound per call, default 256.
- `AILM_TOPK`: per-step candidate count, default 5.
- `AILM_CAPTURE_TAGS`: cap on captured runs held by the patch store.
