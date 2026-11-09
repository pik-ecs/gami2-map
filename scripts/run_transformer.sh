#!/usr/bin/env bash
# Run train-transformer for every model below, one after another. The settings below are the
# experiment: edit them between rounds. Model defaults, search ranges, early stopping and
# pruning are in src/gami2_map/models/config.py; the data and the tuning metric in
# tasks_settings.py. From the project root:
#
#     bash scripts/run_transformer.sh
#     bash scripts/run_transformer.sh --trials 1     (extra options go to every run, e.g. a quick test)
#
# Each run's output goes to logs/transformer/<timestamp>/<model>.log; results go to
# results/<task>/transformer-<model>/<timestamp>/ as usual. A failed run is reported and skipped.

set -uo pipefail
source "$(dirname "$0")/common.sh"

# ---- Settings -----------------------------------------------------------------------------

TASK="inout"

# Any of config.py's TRANSFORMER checkpoints: bert
MODELS=(bert)

# true: tune on validation only (no final model, test unused). false: also train the final
# model on train + validation (for the best trial's best epoch) and score it once on test
VALIDATION_ONLY=true

# 12: enough to cover the one searched setting (learning_rate); each trial takes ~1-1.5 h on
# this Mac (MPS), a few minutes per epoch on a CUDA GPU
TRIALS=12

# Epochs without improvement on validation before a fit stops; 0 turns early stopping off,
# empty keeps config.py's
EARLY_STOPPING_PATIENCE="2"

# true / false: stop trials falling behind earlier ones after an epoch (holdout only); empty
# keeps config.py's
PRUNE="true"

# Any other train-transformer options for every run, e.g. (--set learning_rate=3e-5) to fix
# the learning rate, (--set learning_rate=1e-5,5e-5) to search it over a range, or
# (--set epochs=5) for the most epochs a fit trains for.

EXTRA_ARGS=(
    --set learning_rate=5e-6,3e-5
    --set epochs=8
)

# -------------------------------------------------------------------------------------------

model_args() {
    echo --trials "$TRIALS"
    [[ -n $EARLY_STOPPING_PATIENCE ]] && echo --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    [[ $PRUNE == true ]] && echo --prune
    [[ $PRUNE == false ]] && echo --no-prune
    return 0
}

EXTRA_ARGS+=("$@")
run_models train-transformer transformer "${MODELS[@]}"
