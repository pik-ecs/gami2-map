# Shared by run_classical.sh, run_transformer.sh and run_setfit.sh (sourced, not run).
#
#     run_models <command> <log name> <model>...
#
# runs `uv run <command> $TASK --model <model>` for each model, one after another, with the
# options printed by the calling script's model_args <model>, then --validation-only or
# --no-validation-only (from VALIDATION_ONLY), then EXTRA_ARGS. Later options win, so
# EXTRA_ARGS (and the script's own arguments, appended to it) override the settings.
# Each run's output goes to logs/<log name>/<timestamp>/<model>.log; a failed run is
# reported and skipped.

run_models() {
    local command=$1 name=$2
    shift 2
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    local log_dir
    log_dir="logs/$name/$(date +%Y-%m-%dT%H%M%S)"
    mkdir -p "$log_dir"

    local validation=--no-validation-only
    [[ $VALIDATION_ONLY == true ]] && validation=--validation-only

    local failed=() model
    for model in "$@"; do
        echo "=== $model ($(date +%H:%M:%S))"
        # model_args is split into words on purpose: it prints options, none with spaces
        # shellcheck disable=SC2046
        if uv run "$command" "$TASK" --model "$model" $(model_args "$model") "$validation" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} >"$log_dir/$model.log" 2>&1; then
            echo "    done"
        else
            echo "    FAILED, see $log_dir/$model.log"
            failed+=("$model")
        fi
    done

    echo
    echo "Logs: $log_dir"
    if ((${#failed[@]})); then
        echo "Failed: ${failed[*]}"
        return 1
    fi
    echo "All runs finished."
}
