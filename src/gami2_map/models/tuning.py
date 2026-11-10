"""Hyperparameter search with Optuna, for pipeline.py (transformer, SetFit and classical models).

The pipeline passes tune() a function that trains a model with given settings and returns
all validation metrics; Optuna only sees the tuning metric, and uses it to pick what to try
next.
"""

import csv
import logging
import time
from collections.abc import Callable
from pathlib import Path

import optuna

from gami2_map.models.metrics import TUNING_METRICS

logger = logging.getLogger(__name__)


def check_keys(given: dict, allowed: dict, what: str) -> None:
    unknown = set(given) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown {what} keys: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def search_space_for(defaults: dict, space: dict, params: dict, search: dict) -> dict:
    """The ranges Optuna searches: those in `search` (None: the model's own), or, with no
    `search`, every setting in `space` that `params` doesn't fix."""
    check_keys(params, defaults, "params")
    check_keys(search, space, "search")
    if not search:
        search = {key: None for key in space if key not in params}
    both = set(search) & set(params)
    if both:
        raise ValueError(
            f"{sorted(both)} both fixed and searched; give each setting once"
        )
    return {key: space[key] if values is None else values for key, values in search.items()}


def suggest_params(
    trial: optuna.Trial, space: dict, log_scale: frozenset[str] = frozenset()
) -> dict:
    """One Optuna sample from a search space; keys in `log_scale` are sampled on a log scale."""
    params = {}
    for key, values in space.items():
        if isinstance(values, list):
            params[key] = trial.suggest_categorical(key, values)
        elif all(isinstance(value, int) for value in values):
            params[key] = trial.suggest_int(key, *values, log=key in log_scale)
        else:
            params[key] = trial.suggest_float(key, *values, log=key in log_scale)
    return params


# Pruning (see tune): no trial is pruned until PRUNE_STARTUP_TRIALS have finished, giving the
# median something to go on
PRUNE_STARTUP_TRIALS = 5


def tune(
    score: Callable[[dict, Callable | None], dict[str, float]],
    space: dict,
    trials: int,
    seed: int,
    metric: str,
    log_scale: frozenset[str] = frozenset(),
    params: dict | None = None,
    trials_path: Path | None = None,
    prune_from_epoch: int | None = None,
) -> tuple[dict, dict[str, float]]:
    """Search `space` for the settings whose `score(settings, report)[metric]` is best;
    `params` are fixed in every trial. Returns the best settings (with `params`) and their
    metrics.

    With `prune_from_epoch` (pruning), `report` is a function `score` calls after each epoch
    with report(epoch, metrics); from that epoch on, it raises optuna.TrialPruned when the
    trial falls behind the median of earlier ones (so slow starters get a chance). None: no
    pruning, and `report` is None."""
    params = params or {}
    prune = prune_from_epoch is not None

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        sampled = suggest_params(trial, space, log_scale)
        last_epoch: dict = {}

        def report(epoch: int, metrics: dict[str, float]) -> None:
            last_epoch.update(epoch=epoch, metrics=metrics)
            trial.report(metrics[metric], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        def log(metrics: dict, status: str) -> None:
            # One row per trial as it ends, so the trials survive a run that is stopped
            if trials_path is not None:
                duration = round(time.perf_counter() - started, 1)
                row = {
                    "trial": trial.number,
                    **sampled,
                    "status": status,
                    **metrics,
                    "duration_s": duration,
                }
                _append_row(trials_path, row)

        try:
            metrics = score({**params, **sampled}, report if prune else None)
        except optuna.TrialPruned:
            logger.info(f"Trial {trial.number} pruned after epoch {last_epoch['epoch']}")
            log(last_epoch["metrics"], "pruned")
            raise
        trial.set_user_attr("metrics", metrics)
        log(metrics, "complete")
        return metrics[metric]

    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=PRUNE_STARTUP_TRIALS, n_warmup_steps=prune_from_epoch
        )
        if prune
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction=TUNING_METRICS[metric],
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=trials)
    if prune:
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        logger.info(f"{pruned} of {len(study.trials)} trials pruned")
    logger.info(f"Best params: {study.best_params}")
    return {**params, **study.best_params}, study.best_trial.user_attrs["metrics"]


def _append_row(path: Path, row: dict) -> None:
    """Add `row` to the CSV at `path`, writing the header first if the file is new."""
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
