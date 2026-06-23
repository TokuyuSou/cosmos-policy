# Fused-Encoder Retrieval VLA-Skip Pipeline — environment & run guide

A closed-loop **skip** pipeline for RoboCasa: a fused **action-metric encoder** (R3M image +
proprio + previous actions) is used as a nearest-neighbour **retrieval key** that replaces the heavy
VLA (Cosmos Policy) at open-loop skip decision points. Per task we: collect demos → train the fused
encoder → measure the full-VLA ceiling → run the fused-encoder retrieval closed-loop over a sweep of
skip rates → plot success vs *effective* skip rate against that ceiling.

## Environment

Base environment (Docker, CUDA, RoboCasa assets): follow the repo's top-level
[SETUP.md](../SETUP.md) and [ROBOCASA.md](../ROBOCASA.md). Everything below runs **inside the
project's `uv` environment**, selected per command:

```bash
# simulator-using steps (collection, closed-loop eval):
uv run --extra cu128 --group robocasa --python 3.10  python <script> ...
# encoder training (no simulator needed):
uv run --extra cu128                  --python 3.10  python <script> ...
```

- `--extra cu128`   → CUDA 12.8 / torch 2.7 wheels (`pyproject.toml` → `[project.optional-dependencies].cu128`)
- `--group robocasa`→ RoboCasa + robosuite + mujoco (`pyproject.toml` → `[dependency-groups].robocasa`)
- exact versions are pinned in **`uv.lock`**.
- extra research deps: **`scikit-learn`** (in `pyproject.toml`), **matplotlib** (plots), and the
  **R3M ResNet-18** weights, fetched automatically on the first encoder run.
- export `HF_HOME=<your hf cache>` and `HF_HUB_OFFLINE=1` for the Cosmos VLA weights.

## Run a task end-to-end

Substitute `TASK` (env name, e.g. `PnPCounterToStove`) and `DBASE` (snake_case, e.g.
`pnp_counter_to_stove`). Run from the repo root. A ready-made driver for all of the below is
`research/results/closed_loop/_batch/run_task_pipeline.sh <TASK> <DBASE>` (not under version control,
since it lives in the git-ignored `results/` tree — the canonical commands are reproduced here).

**1. Collect 150 demos** (dense, with VLA shadow queries at stride 4):
```bash
uv run --extra cu128 --group robocasa --python 3.10 \
  python research/action_predictor/launch_collect.py \
  --collector collect_data_dense.py --task TASK --total-episodes 150 \
  --gpus 0,1,2 --procs-per-gpu 4 --query-stride 4 \
  --out-dir research/data/DBASE_dense_img --seed 195 --denoising-steps 5
```

**2. Gripper auto-decision** — include the gripper action dim in the metric iff it actually varies:
`std(executed gripper) > 0.1` → add `--include-grip` (D=7), else leave it out (D=6). (Grasping tasks
→ D=7; push/turn tasks where the gripper is constant → D=6.)

**3. Train the fused encoder:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --python 3.10 \
  python research/r3m_action_encoder/train.py \
  --arch fused --state-enc mlp --fusion residual --mod-dropout 0.0 --freeze layer4 \
  --out-dim 256 --img-dim 512 --state-dim 512 --epochs 120 --seed 0 \
  --loss corr+supcon --supcon-k 8 --supcon-temp 0.1 [--include-grip] \
  --data research/data/DBASE_dense_img --tag fused_corrsupcon_k8_<grip|nogrip>
# -> research/r3m_action_encoder/results/DBASE/encoder_<tag>.pt
```

**4. Full-VLA ceiling** (skip rate 0 = never skip; same held-out protocol as the fused eval):
```bash
uv run --extra cu128 --group robocasa --python 3.10 \
  python research/action_predictor/run_closed_loop_parallel.py \
  --policy retrieval --data-dir research/data/DBASE_dense_img \
  --key prev_state --knn 1 --state-source actual_next_proprio \
  --skip-policy random --skip-rates 0 --task TASK --total-episodes 50 --episode-start 5000 \
  --gpus 1,2 --procs-per-gpu 4 --seed 195 \
  --out research/results/closed_loop/TASK/full_vla_DBASE
```

**5. Fused-encoder retrieval closed-loop** (even skip, rate sweep):
```bash
uv run --extra cu128 --group robocasa --python 3.10 \
  python research/action_predictor/run_closed_loop_parallel.py \
  --policy retrieval --data-dir research/data/DBASE_dense_img \
  --key prev_state --knn 1 --state-source actual_next_proprio \
  --fused-encoder research/r3m_action_encoder/results/DBASE/encoder_<tag>.pt \
  --skip-policy even --skip-rates 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --task TASK --total-episodes 50 --episode-start 5000 \
  --gpus 0,1,2 --procs-per-gpu 4 --seed 195 \
  --out research/results/closed_loop/TASK/retrieval_fused_DBASE
```

**6. Plot** fused vs the full-VLA ceiling (ceiling line + 95/90% bands):
```bash
cd research && ../.venv/bin/python action_predictor/compare_skip_curves.py \
  --runs "results/closed_loop/TASK/retrieval_fused_DBASE:fused retrieval" \
  --baseline "results/closed_loop/TASK/full_vla_DBASE:full VLA" --no-error-bars \
  --out results/closed_loop/TASK/fused_DBASE.png
```
(If there is no full-VLA run, pass a known ceiling value instead with `--baseline-value 0.NN`.)

### Optional: which cache frame did each skip hit?
Re-run a small eval with `DUMP_HIT_IMAGES=1` (records the live decision frame), then:
```bash
cd research && ../.venv/bin/python action_predictor/cache_hit_viz.py \
  --run results/closed_loop/TASK/retrieval_fused_DBASE --data data/DBASE_dense_img
```
→ a live-vs-matched-cache filmstrip, the retrieval-source timeline, and hit-frame progress. The skip
trace already records each hit's cache provenance (`hit_src_ep`/`hit_src_imgidx`/`hit_dist`) for every run.

## Validated settings
- **Encoder**: residual fusion, modality-dropout 0, R3M `layer4` fine-tune, 256-d output, branches 512,
  `corr+supcon` (k=8, temp 0.1), 120 epochs, checkpoint selected by **val RMSE@1**.
- **Eval**: 16-step open-loop blocks, **even** (Bresenham) skip schedule, 50 episodes from index 5000
  (held out from the cache/training, which use 0–149), seed 195, OSC_POSE, chunk 32, 5 denoising steps.
- **Data**: dense `query-stride 4`; the retrieval cache = the ~85% train split of the success episodes.

## Code map
| Area | Files |
|---|---|
| action-distance loss + retrieval metrics | `action_encoder/{losses,eval_retrieval}.py` |
| fused encoder training | `r3m_action_encoder/{train,model,data,eval,metric_losses}.py` |
| retrieval policy + closed-loop skip eval | `action_predictor/{retrieval_policy,predictor_policy,skip_policy,closed_loop,run_closed_loop_eval,run_closed_loop_parallel,obs_embed,dataset,model}.py` |
| dense data collection | `action_predictor/{collect_data_dense,launch_collect}.py` |
| plotting + cache-hit visualization | `action_predictor/{compare_skip_curves,cache_hit_viz}.py` |
