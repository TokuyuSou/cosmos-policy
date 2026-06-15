# Research: Lightweight Action Predictor for Cosmos Policy

Goal: train a small (tens-of-M) **action predictor** that, given
*(a)* the last 16 executed actions, *(b)* the VLA-predicted self-state
(Cosmos future-proprio), and *(c)* the Cosmos future-image latent,
predicts the **next 16 actions** — so that a VLA (Cosmos Policy) call can be
skipped. We benchmark it against simply executing Cosmos Policy's own
**remaining 16 actions** of the predicted 32-chunk (RMSE vs the ground-truth
successful episode).

Task: `PnPCounterToStove` with `PandaOmron` (= `PandaMobile`).

## Pipeline

```
collect_data.py        run Cosmos Policy, log per-VLA-call signals (stride=16)  -> data/<task>/ep_*.npz
collect_data_dense.py   shadow dense queries (stride=K)                           -> data/<task>_dense/ep_*.npz
launch_collect.py       run collection in parallel across GPUs (sets MUJOCO_EGL_DEVICE_ID)
dataset.py              build (input, target) pairs (auto-detects dense schema)
model.py                ActionPredictor (token transformer, future-image pooling modes)
train.py                train one variant -> <results>/<run>/{ckpt.pt,config.json,metrics.json}
evaluate.py             RMSE: predictor vs Cosmos-remaining-16 vs ground truth -> <results>/<run>/eval.json
run_experiments.py      train+eval all variants in parallel across GPUs, write RESULTS.md
```

Results live under `results/`:
- `results/nondense/` — trained on stride-16 data (`data/pnp_counter_to_stove`)
- `results/dense/`    — trained on stride-4 dense data (`data/pnp_counter_to_stove_dense`)

Each has `RESULTS.md` (per-variant RMSE table); `results/dense/OVERFITTING_vs_nondense.md`
compares overfitting between the two. See `action_predictor/` for code.

## Data schema (per successful VLA call i, every 16 env steps)

| key | shape | meaning |
|---|---|---|
| `chunk` | (32, 7) | full Cosmos action chunk (physical / unnormalized) |
| `cur_proprio` | (9,) | observed proprio at the call (physical) |
| `future_proprio_norm` | (9,) | VLA-predicted future proprio (extracted from latent frame 6, normalized space) |
| `future_img_latent` | (3, 16, 28, 28) | Cosmos future-image latents [wrist, primary, secondary] (fp16) |
| `call_t` | () | env timestep of the call |

Training sample from consecutive calls (i, i+1):
`input = {chunk[i][:16], future_proprio_norm[i], future_img_latent[i]}`,
`target = chunk[i+1][:16]` (the actually-executed next 16).
Baseline = `chunk[i][16:32]` (Cosmos's own remaining 16).
