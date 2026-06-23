#!/bin/bash
# ============================================================================================
# Generic full FUSED pipeline for a FRESH RoboCasa task (no data yet):
#   STEP 1  collect 150 ep (VLA shadow, stride 4)                                  GPUs 0,1,2  [skip if data exists]
#   STEP A  gripper auto-decision: INCLUDE gripper (D=7) iff std>0.1, else D=6
#   STEP 2  fused encoder (corr+supcon, k=8)                                        GPU 0   ┐ parallel
#   STEP 3  full-VLA ceiling (skip 0), SAME 5000/seed195 protocol                   GPUs 1,2 ┘ (fair)
#   STEP 4  fused-encoder retrieval closed-loop, EVEN skip, rates 0.1..1.0          GPUs 0,1,2  [needs 2]
#   STEP 5  plot fused vs full-VLA ceiling (real --baseline -> ceiling + 95/90% bands)
# If the VLA can't do the task (too few successful collections), abort after STEP 1 (logged).
# Usage: run_task_pipeline.sh <TaskName> <data_basename>
# ============================================================================================
set -u
TASK="${1:?task}"; DBASE="${2:?dbase}"
ROOT=/workspace/cosmos-policy; cd "$ROOT"
export HF_HOME=${HF_HOME:-/home/cao/.cache/huggingface}; export HF_HUB_OFFLINE=1; export PYTHONUNBUFFERED=1
PDIR=research/results/closed_loop/$TASK/_pipeline; mkdir -p "$PDIR"
MLOG=$PDIR/fused_pipeline.log
log(){ echo "[$(date -u +%FT%TZ)] [$TASK] $*" | tee -a "$MLOG"; }
DATADIR=research/data/${DBASE}_dense_img
ENC_DIR=research/r3m_action_encoder/results/$DBASE
FUSED_OUT=research/results/closed_loop/$TASK/retrieval_fused_$DBASE
VLA_DIR=research/results/closed_loop/$TASK/full_vla_$DBASE
mkdir -p "$DATADIR" "$ENC_DIR" "$FUSED_OUT" "$VLA_DIR"
nsucc(){ ls "$DATADIR"/ep*_success=1.npz 2>/dev/null | wc -l; }

# ---- STEP 1: data collection (skip if already enough) ----
if [ "$(nsucc)" -lt 40 ]; then
  log "STEP 1: collecting 150 episodes (VLA shadow, query-stride 4, seed 195) ..."
  uv run --extra cu128 --group robocasa --python 3.10 \
    python research/action_predictor/launch_collect.py \
    --collector collect_data_dense.py --task "$TASK" --total-episodes 150 \
    --gpus 0,1,2 --procs-per-gpu 4 --query-stride 4 \
    --out-dir "$DATADIR" --seed 195 --denoising-steps 5 > "$DATADIR/collect.log" 2>&1
  log "STEP 1: collect rc=$? -> $(nsucc) successful episodes"
fi
NS=$(nsucc)
if [ "$NS" -lt 25 ]; then
  log "ABORT: only $NS successful episodes (VLA can't do $TASK reliably) -> skip the rest, continue batch."
  exit 0
fi

# ---- STEP A: gripper auto-decision ----
read SUF GSTD < <(.venv/bin/python - "$DATADIR" <<'PY'
import sys, glob, numpy as np
files = sorted(glob.glob(sys.argv[1] + "/ep*_success=1.npz"))
g = np.concatenate([np.load(f, allow_pickle=True)["realized_actions"][:, 6].astype("f4") for f in files])
print(("grip" if g.std() > 0.1 else "nogrip"), round(float(g.std()), 4))
PY
)
GRIP=""; [ "$SUF" = "grip" ] && GRIP="--include-grip"
log "STEP A: gripper std=$GSTD -> $([ -n "$GRIP" ] && echo 'INCLUDE (D=7)' || echo 'exclude (D=6)')"
TAG=fused_corrsupcon_k8_$SUF; ENC_CKPT=$ENC_DIR/encoder_$TAG.pt

# ---- STEP 2 encoder (GPU0)  ||  STEP 3 full-VLA ceiling (GPUs 1,2) ----
log "STEP 2: encoder tag=$TAG (GPU0) [parallel]"
( CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --python 3.10 \
    python research/r3m_action_encoder/train.py \
    --arch fused --state-enc mlp --fusion residual --mod-dropout 0.0 --freeze layer4 \
    --out-dim 256 --img-dim 512 --state-dim 512 --epochs 120 --seed 0 \
    --loss corr+supcon --supcon-k 8 --supcon-temp 0.1 $GRIP \
    --data "$DATADIR" --tag "$TAG" > "$ENC_DIR/train_$TAG.log" 2>&1; echo $? > "$PDIR/.enc_rc" ) &
ENC_PID=$!
log "STEP 3: full-VLA ceiling, skip 0 (GPUs 1,2) [parallel]"
( uv run --extra cu128 --group robocasa --python 3.10 \
    python research/action_predictor/run_closed_loop_parallel.py \
    --policy retrieval --data-dir "$DATADIR" --key prev_state --knn 1 --state-source actual_next_proprio \
    --skip-policy random --skip-rates 0 --skip-seed 0 \
    --task "$TASK" --total-episodes 50 --episode-start 5000 \
    --gpus 1,2 --procs-per-gpu 4 --seed 195 \
    --out "$VLA_DIR" > "$VLA_DIR/run.log" 2>&1; echo $? > "$PDIR/.vla_rc" ) &
VLA_PID=$!
wait $ENC_PID; log "STEP 2: encoder rc=$(cat "$PDIR/.enc_rc" 2>/dev/null)"
wait $VLA_PID; log "STEP 3: full-VLA rc=$(cat "$PDIR/.vla_rc" 2>/dev/null)"

# ---- STEP 4: fused closed-loop ----
if [ -f "$ENC_CKPT" ]; then
  log "STEP 4: fused closed-loop (GPUs 0,1,2)"
  uv run --extra cu128 --group robocasa --python 3.10 \
    python research/action_predictor/run_closed_loop_parallel.py \
    --policy retrieval --data-dir "$DATADIR" --key prev_state --knn 1 --state-source actual_next_proprio \
    --fused-encoder "$ENC_CKPT" \
    --skip-policy even --skip-rates 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --task "$TASK" --total-episodes 50 --episode-start 5000 \
    --gpus 0,1,2 --procs-per-gpu 4 --seed 195 \
    --out "$FUSED_OUT" > "$FUSED_OUT/run.log" 2>&1
  log "STEP 4: fused rc=$?"
else
  log "STEP 4 SKIPPED: encoder checkpoint missing ($ENC_CKPT)"
fi

# ---- STEP 5: plot vs full-VLA ceiling ----
if [ -f "$FUSED_OUT/closed_loop_eval.json" ]; then
  D=$([ -n "$GRIP" ] && echo 7 || echo 6)
  BL=(); [ -f "$VLA_DIR/closed_loop_eval.json" ] && BL=( --baseline "${VLA_DIR#research/}:full VLA" )
  (cd research && ../.venv/bin/python action_predictor/compare_skip_curves.py \
     --runs "${FUSED_OUT#research/}:fused+$SUF (D=$D) retrieval" "${BL[@]}" --no-error-bars \
     --title "$TASK: fused+$SUF (D=$D) retrieval vs full-VLA ceiling (EVEN skip, 50 ep)" \
     --out "results/closed_loop/$TASK/fused_${SUF}_${DBASE}.png") > "$PDIR/plot.log" 2>&1
  log "STEP 5: plot rc=$?"
else
  log "STEP 5 SKIPPED: no fused closed_loop_eval.json"
fi
log "TASK DONE. grip=$SUF(std=$GSTD) tag=$TAG successes=$NS"
