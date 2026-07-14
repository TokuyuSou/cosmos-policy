# Reference-Servo TMT Ridge4

This directory tests one minimal extension of TMT without modifying any
existing policy or checkpoint:

1. TMT retrieves a successful action chunk every 16 environment steps.
2. At each intermediate four-step query, the reference advances by four steps
   within the same demonstration instead of running another TMT retrieval.
3. A ridge head predicts one shared 6-D arm correction for the next four
   actions. Gripper commands remain those of the reference.

The design asks a narrow question: can a persistent retrieved trajectory plus
a low-dimensional local correction improve the short-replan regime? It does
not introduce another action tokenizer or a full generative policy.

## Training and offline evaluation

`experiment.py` uses episode-disjoint train/validation/test splits. Ridge
regularization, the optional anchor-distance feature, and correction clipping
are selected on validation episodes. The frozen choices are then refit on
train+validation episodes and evaluated once on test episodes under the exact
TMT@16/Ridge@4 persistence contract used in closed loop. Re-retrieval is
allowed only at a scheduled anchor or if the reference trajectory ends.

## Run

From the repository root:

```bash
.venv/bin/python research/reference_servo_tmt/test_reference_servo.py
.venv/bin/python research/reference_servo_tmt/experiment.py
```

Results are written to `results/cab_seed0/metrics.json`. The frozen TMT
embeddings and the required Theia grid tokens are cached locally after the
first run; they are regenerated from the raw episode RGB when absent. The adapter and local
replanning are opt-in; existing TMT and VQ policy behavior is unchanged.

The JSON reports both cache contracts: `published_protocol_train_only_baseline`
matches the original offline TMT table, while `baseline` and the learned ridge
use the train+validation success cache available to the closed-loop policy.

## Closed-loop adapter

The opt-in adapter is `action_predictor/reference_servo_tmt_policy.py`. Existing
`--policy tmt` behavior is unchanged. A single-process smoke/evaluation command
is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/action_predictor/run_closed_loop_eval.py \
  --policy reference_servo_tmt \
  --data-dir research/data/pnp_cab_to_counter_dense_img_orig150 \
  --tmt-encoder research/r3m_action_encoder/results/pnp_cab_to_counter_orig150/tmt_d192L3_failq.pt \
  --servo-model research/reference_servo_tmt/results/cab_seed0/persistent_twist_ridge.npz \
  --local-replan-steps 4 --task PnPCabToCounter \
  --skip-policy even --skip-rates 0.4 --episode-start 5000 --num-episodes 1 \
  --out research/results/closed_loop/PnPCabToCounter/reference_servo_tmt_smoke
```

The runtime refuses a custom cache size, TMT-weight override, or non-4-step
local period because the frozen ridge was trained under the default cache,
learned TMT weight, TMT@16/Ridge@4 contract. Run the simulator-free real-cache
parity check with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  research/action_predictor/verify_reference_servo_tmt_policy.py
```
