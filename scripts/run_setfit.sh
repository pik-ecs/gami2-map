#!/usr/bin/env bash
# Run train-setfit for every model below, one after another. The settings below are the
# experiment: edit them between rounds. Model defaults and search ranges are in
# src/gami2_map/models/config.py; the data and the tuning metric in tasks_settings.py.
# From the project root:
#
#     bash scripts/run_setfit.sh
#     bash scripts/run_setfit.sh --trials 1     (extra options go to every run, e.g. a quick test)
#
# Each run's output goes to logs/setfit/<timestamp>/<model>.log; results go to
# results/<task>/setfit-<model>/<timestamp>/ as usual. A failed run is reported and skipped.

set -uo pipefail
source "$(dirname "$0")/common.sh"

# ---- Settings -----------------------------------------------------------------------------

TASK="inout"

# Any of config.py's SETFIT checkpoints: mpnet, allenai
MODELS=(mpnet)

# true: tune on validation only (no final model, test unused). false: train the final model
# on train + validation and score it once on test
VALIDATION_ONLY=true

TRIALS=0

# Any other train-setfit options for every run, e.g. (--set body_learning_rate) to search only
# that setting, or (--set epochs=2) to fix one
EXTRA_ARGS=()

# -------------------------------------------------------------------------------------------

model_args() {
    echo --trials "$TRIALS"
}

EXTRA_ARGS+=("$@")
run_models train-setfit setfit "${MODELS[@]}"
