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

# Any of config.py's TRANSFORMER checkpoints: bert, scibert, bge-base, climatebert
MODELS=(scibert)

VALIDATION_ONLY=true

TRIALS=15

EARLY_STOPPING_PATIENCE="2"

PRUNE_FROM_EPOCH=""

EXTRA_ARGS=(--set learning_rate=1e-5,5e-5 --set epochs=12)

# -------------------------------------------------------------------------------------------

model_args() {
    echo --trials "$TRIALS"
    [[ -n $EARLY_STOPPING_PATIENCE ]] && echo --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    [[ -n $PRUNE_FROM_EPOCH ]] && echo --prune-from-epoch "$PRUNE_FROM_EPOCH"
    return 0
}

EXTRA_ARGS+=("$@")
run_models train-transformer transformer "${MODELS[@]}"
