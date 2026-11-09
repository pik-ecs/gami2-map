#!/usr/bin/env bash
# Run train-classical for every classifier × features combination below, one after another.
# The settings below are the experiment: edit them between rounds. Model defaults and search
# ranges are in src/gami2_map/models/config.py; the data and the tuning metric in tasks_settings.py.
# From the project root:
#
#     bash scripts/run_classical.sh
#     bash scripts/run_classical.sh --trials 3     (extra options go to every run, e.g. a quick test)
#
# Each run's output goes to logs/classical/<timestamp>/<model>.log; results go to
# results/<task>/classical-<model>/<timestamp>/ as usual. A failed run is reported and skipped.

set -uo pipefail
source "$(dirname "$0")/common.sh"

# ---- Settings -----------------------------------------------------------------------------

TASK="inout"

CLASSIFIERS=(logreg svm)
FEATURES=(tfidf count glove mpnet)

# true: tune with 5-fold CV and stop there, to compare models (no final model, test unused).
# false: also train the final model on all folds and score it once on test
VALIDATION_ONLY=true

# Optuna trials per feature type: tfidf / count tune C, binary, ngram_max and min_df; the
# embedders only C
trials_for() {
    case "$1" in
        tfidf | count) echo 80 ;;
        *) echo 30 ;;
    esac
}

# Any other train-classical options for every run, e.g. (--set C=0.01) to fix C, (--set C=1e-3,10)
# to search it over a range, or (--split holdout)
EXTRA_ARGS=()

# -------------------------------------------------------------------------------------------

model_args() {
    echo --trials "$(trials_for "${1#*-}")"
}

models=()
for classifier in "${CLASSIFIERS[@]}"; do
    for features in "${FEATURES[@]}"; do
        models+=("$classifier-$features")
    done
done

EXTRA_ARGS+=("$@")
run_models train-classical classical "${models[@]}"
