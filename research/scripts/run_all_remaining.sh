#!/bin/bash
# ============================================================================================
# Run the full fused pipeline for EVERY remaining RoboCasa atomic task, SEQUENTIALLY, each starting
# from data collection. One task fully finishes (collect -> encoder||full-VLA -> fused -> plot) before
# the next begins (they all use the 3 GPUs). A task that fails/aborts never stops the batch.
# Launch with setsid+nohup so it survives SSH disconnect.
# ============================================================================================
set -u
ROOT=/workspace/cosmos-policy; cd "$ROOT"
GEN=research/scripts/run_task_pipeline.sh            # the generic per-task driver (committed, this dir)
MLOG=research/results/closed_loop/_batch/batch.log   # batch log (under the git-ignored results/ tree)
mkdir -p "$(dirname "$MLOG")"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$MLOG"; }

# (TaskName  data_basename) -- quick single-action tasks first, then PnP, then coffee.
TASKS=(
  "OpenDrawer open_drawer"
  "CloseSingleDoor close_single_door"
  "OpenDoubleDoor open_double_door"
  "CloseDoubleDoor close_double_door"
  "TurnOffSinkFaucet turn_off_sink_faucet"
  "TurnOnStove turn_on_stove"
  "TurnOffStove turn_off_stove"
  "TurnOnMicrowave turn_on_microwave"
  "TurnOffMicrowave turn_off_microwave"
  "PnPCabToCounter pnp_cab_to_counter"
  "PnPCounterToCab pnp_counter_to_cab"
  "PnPCounterToMicrowave pnp_counter_to_microwave"
  "CoffeePressButton coffee_press_button"
  "CoffeeSetupMug coffee_setup_mug"
  "CoffeeServeMug coffee_serve_mug"
)
log "BATCH START: ${#TASKS[@]} remaining tasks"
i=0
for entry in "${TASKS[@]}"; do
  i=$((i+1)); set -- $entry; TASK=$1; DBASE=$2
  log ">>> [$i/${#TASKS[@]}] $TASK ($DBASE) START"
  bash "$GEN" "$TASK" "$DBASE"; rc=$?
  log "<<< [$i/${#TASKS[@]}] $TASK END (pipeline rc=$rc)"
done
log "BATCH COMPLETE ($i tasks processed)"
