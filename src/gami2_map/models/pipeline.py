"""The training pipeline for transformer_model.py, setfit_model.py and classical_model.py:
tune, train the final model, test.
Main functions:
    run_experiment       one run: settings → data → tune_and_train → test → results.json
    make_cli             the command line (train-transformer / train-setfit / train-classical)
                         → run_experiment

A run's settings come from config.py and the command line (in practice scripts/run_*.sh);
the data, text columns and tuning metric from tasks_settings.py.
"""

import ast
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from datasets import Dataset, DatasetDict, concatenate_datasets

from gami2_map.data.inputs import concat_input_text
from gami2_map.data.load import load_huggingface_splits, load_settings
from gami2_map.models.metrics import TUNING_METRICS, scores
from gami2_map.models.tuning import check_keys, search_space_for, tune
from gami2_map.paths import OUTPUT_DIR, available_tasks
from gami2_map.tasks import data_config, task_config

logger = logging.getLogger(__name__)

TRAIN_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    """Everything the pipeline needs to know about one kind of classifier."""

    method: str
    # (data, model key, max_length) -> data with the model's input columns, once per run
    prepare: Callable[[DatasetDict, str, int | None], DatasetDict]
    # (model key, train data, settings, *, seed, max_length, output_dir) -> trained model
    fit: Callable[..., Any]
    # (model, data) -> (probability of include per paper, true labels)
    predict_include: Callable[[Any, Dataset], tuple[np.ndarray, np.ndarray]]
    # (model, folder) -> None
    save: Callable[[Any, Path], None]
    log_scale: frozenset[str] = frozenset()
    default_trials: int = 10
    # "holdout" or "cv<n>" (see data/splits.py); --split overrides it
    default_split: str = "holdout"
    # Per model (keyed by its --model name, e.g. logreg-tfidf): its full default settings
    # (what --trials 0 trains with), and the ranges Optuna searches (same keys)
    defaults: dict[str, dict] = field(default_factory=dict)
    search_spaces: dict[str, dict] = field(default_factory=dict)
    # Tokens per paper, per model, from config.py; passed to prepare and fit (None if unset)
    max_lengths: dict[str, int | None] = field(default_factory=dict)
    # The pretrained model each one uses, from config.py, recorded in results.json (none
    # for tfidf / count)
    checkpoints: dict[str, str] = field(default_factory=dict)
    # The classifier's class weighting from config.py (e.g. "balanced"), recorded in
    # results.json; None for unweighted
    class_weight: str | None = None
    # Models trained epoch by epoch (transformers). Their fit also takes eval_data, patience,
    # metric, train_sample and on_epoch (see transformer_model.fit), and the model it returns
    # has best_epoch: the epoch whose weights it kept
    by_epoch: bool = False
    # by_epoch only. Early stopping: each tuning fit stops once the tuning metric on
    # validation hasn't improved for this many epochs and keeps its best epoch; the final
    # model trains for the best trial's best epoch (defaults["epochs"] is then the most a fit
    # trains for). None: no early stopping
    early_stopping_patience: int | None = None
    # by_epoch only. Pruning: from this epoch on, tuning stops trials falling behind earlier
    # ones (see tuning.tune); holdout only. None: no pruning
    prune_from_epoch: int | None = None

    def run_name(self, model_key: str) -> str:
        """e.g. classical-logreg-tfidf: the run's folder under results/<task>/."""
        return f"{self.method}-{model_key}"


@dataclass
class Fold:
    """One (train, validation) pair a trial is scored on, plus a sample of its train set as
    large as its validation set: scoring the model on it too shows overfitting."""

    train: Dataset
    validation: Dataset
    train_sample: Dataset


@dataclass
class Tuned:
    """What tune_and_train returns."""

    # The final model; None with train_final_model=False
    model: Any
    # The best trial's settings, over the model's defaults: what the final model trains with
    params: dict
    # What Optuna searched ({} with --trials 0)
    search_space: dict
    # The best trial's validation scores ({} with --trials 0 and a final model)
    best_validation: dict


class BadArguments(ValueError):
    """A run's options don't fit the task or model; the command line reports these as
    usage errors rather than crashes."""


def run_experiment(
    spec: ModelSpec,
    task: str,
    model_key: str,
    trials: int | None = None,
    params: dict | None = None,
    search: dict | None = None,
    validation_only: bool = False,
    split: str | None = None,
    early_stopping_patience: int | None = None,
    prune_from_epoch: int | None = None,
) -> dict:
    """Tune (on holdout validation or with CV, see `split`), train the final model, test it
    once and write results/<task>/<run>/<timestamp>/. `early_stopping_patience` (0 turns it
    off) and `prune_from_epoch` (turns pruning on) override the spec's."""
    # 1. Settings and checks
    trials = spec.default_trials if trials is None else trials
    params, search = params or {}, search or {}
    split = split or spec.default_split
    metric = task_config(task).get("metric", "f1")
    if early_stopping_patience is not None or prune_from_epoch is not None:
        if not spec.by_epoch:
            raise BadArguments(
                f"{spec.method} models don't train by epoch: no early stopping or pruning"
            )
        if early_stopping_patience is not None:
            spec = replace(spec, early_stopping_patience=early_stopping_patience or None)
        if prune_from_epoch is not None:
            spec = replace(spec, prune_from_epoch=prune_from_epoch)
    # Pruning compares one learning curve per trial; with CV there is one per fold
    if spec.prune_from_epoch is not None and split != "holdout":
        logger.info(f"No pruning with --split {split}: it needs holdout")
        spec = replace(spec, prune_from_epoch=None)
    space = check_args(spec, task, model_key, params, search, metric, split)
    max_length = spec.max_lengths.get(model_key)

    started = datetime.now()
    run_dir = (
        OUTPUT_DIR
        / task
        / spec.run_name(model_key)
        / started.strftime("%Y-%m-%dT%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load the split and prepare the model's inputs (e.g. embeddings). Without a final
    # model the test set is dropped here, so it can't be used by accident
    data = load_huggingface_splits(task, split=split)
    split_settings = load_settings(task)
    if validation_only:
        del data["test"]
    text_columns = data_config(task)["text_columns"]
    data = spec.prepare(concat_input_text(data, text_columns), model_key, max_length)
    logger.info({name: group.num_rows for name, group in data.items()})

    # 3. Tune, then train the final model on everything but test
    tuned = tune_and_train(
        spec,
        model_key,
        data,
        space,
        trials=trials,
        params=params,
        metric=metric,
        trials_path=run_dir / "trials.csv",
        train_final_model=not validation_only,
    )

    # 4. Score the final model once on test, and save it
    if not validation_only:
        test_metrics = scores(*spec.predict_include(tuned.model, data["test"]))
        logger.info(f"Test metrics: {test_metrics}")
        spec.save(tuned.model, run_dir / "model")

    # 5. Write results.json: everything needed to tell what this run did
    results = {
        "task": task,
        "model": model_key,
        "checkpoint": spec.checkpoints.get(model_key),
        "method": spec.method,
        "trained_at": started.isoformat(timespec="seconds"),
        "validation_only": validation_only,
        "split": {
            "name": split,
            "test_size": split_settings["test_size"],
            "seed": split_settings["seed"],
            "counts": split_settings["splits"][split]["counts"],
            "data_fingerprint": split_settings["splits"][split]["fingerprint"],
        },
        "seed": TRAIN_SEED,
        "trials": trials,
        "tuning_metric": metric,
        "tuned": sorted(tuned.search_space),
        "search_space": tuned.search_space,
        "best_validation": tuned.best_validation,
        "overrides": params,
        "params": tuned.params,
        "max_length": max_length,
        "class_weight": spec.class_weight,
    }
    if spec.by_epoch:
        results["early_stopping_patience"] = spec.early_stopping_patience
        # None: no pruning
        results["prune_from_epoch"] = spec.prune_from_epoch if trials > 0 else None
    if not validation_only:
        results["test_metrics"] = test_metrics
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {run_dir}")
    return results


def check_args(
    spec: ModelSpec,
    task: str,
    model_key: str,
    params: dict,
    search: dict,
    metric: str,
    split: str,
) -> dict:
    """The space Optuna searches (see search_space_for). Raises BadArguments for an unknown
    task, model, metric or split, or settings the model doesn't have."""
    try:
        if task not in available_tasks():
            raise ValueError(f"Unknown task {task!r}; available: {available_tasks()}")
        if model_key not in spec.defaults:
            raise ValueError(
                f"Unknown model {model_key!r}; available: {list(spec.defaults)}"
            )
        if metric not in TUNING_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; tuning can use: {list(TUNING_METRICS)}"
            )
        if split != "holdout" and not (split.startswith("cv") and split[2:].isdigit()):
            raise ValueError(
                f"Unknown split {split!r}; use 'holdout' or 'cv<n>', e.g. 'cv5'"
            )
        space = search_space_for(
            spec.defaults[model_key],
            spec.search_spaces[model_key],
            params,
            search,
        )
        if spec.early_stopping_patience is not None and "epochs" in space:
            raise ValueError(
                "epochs can't be searched with early stopping, which picks it; fix the "
                "most a fit trains for with --set epochs=<n>, or turn early stopping off"
            )
        return space
    except ValueError as e:
        raise BadArguments(str(e)) from e


def tune_and_train(
    spec: ModelSpec,
    model_key: str,
    data: DatasetDict,
    space: dict,
    trials: int,
    params: dict,
    metric: str,
    trials_path: Path | None = None,
    train_final_model: bool = True,
) -> Tuned:
    """Search `space` with Optuna (each trial scored on validation or the CV folds), then
    train the final model on all non-test data. trials=0: no search, train with `params`.
    With early stopping, the final model trains for the best trial's best epoch (for CV,
    the mean over the folds)."""
    folds = validation_folds(data)
    early_stopping = spec.early_stopping_patience is not None

    def objective(trial_params: dict, report: Callable | None = None) -> dict[str, float]:
        return score_trial(spec, model_key, folds, trial_params, metric, report)

    if trials == 0:
        logger.info(
            f"Skipping tuning; training {spec.run_name(model_key)} with "
            f"{ {**spec.defaults[model_key], **params} }"
        )
        best_params, space, best_validation = params, {}, {}
        # Without a final model, still score these settings so the run reports something;
        # with early stopping, also to find how many epochs the final model trains for
        if not train_final_model or early_stopping:
            best_validation = objective(params)
            logger.info(f"Validation metrics: {best_validation}")
    else:
        best_params, best_validation = tune(
            objective,
            space,
            trials=trials,
            seed=TRAIN_SEED,
            metric=metric,
            log_scale=spec.log_scale,
            params=params,
            trials_path=trials_path,
            prune_from_epoch=spec.prune_from_epoch,
        )

    if early_stopping:
        best_params = {**best_params, "epochs": max(1, round(best_validation["epoch"]))}

    model = None
    if train_final_model:
        everything_but_test = concatenate_datasets(
            [data[name] for name in data if name != "test"]
        )
        model = train(spec, model_key, everything_but_test, best_params)
    return Tuned(
        model=model,
        params={**spec.defaults[model_key], **best_params},
        search_space=space,
        best_validation=best_validation,
    )


def validation_folds(data: DatasetDict) -> list[Fold]:
    """The (train, validation) pairs a trial is scored on: one for holdout; for CV, each
    fold in turn as validation, the other folds combined as train."""
    # Holdout data has "train" and "validation" groups
    if "train" in data:
        pairs = [(data["train"], data["validation"])]
    else:
        # CV data has "fold_0", "fold_1", ... groups
        fold_names = [str(name) for name in data if str(name).startswith("fold_")]
        fold_names.sort(key=lambda name: int(name.removeprefix("fold_")))
        pairs = []
        for validation_name in fold_names:
            train_names = [name for name in fold_names if name != validation_name]
            train_data = concatenate_datasets([data[name] for name in train_names])
            pairs.append((train_data, data[validation_name]))

    folds = []
    for train_data, validation in pairs:
        sample_size = min(len(validation), len(train_data))
        sample = train_data.shuffle(seed=TRAIN_SEED).select(range(sample_size))
        folds.append(Fold(train_data, validation, sample))
    return folds


def score_trial(
    spec: ModelSpec,
    model_key: str,
    folds: list[Fold],
    trial_params: dict,
    metric: str,
    report: Callable | None = None,
) -> dict[str, float]:
    """Train with `trial_params` on each fold; score on its validation set and on its train
    sample (as train_*, to spot overfitting). Several folds are averaged. by_epoch models
    also report the epoch they kept, and, given `report` (pruning, holdout only), call
    report(epoch, scores) after each epoch."""
    fold_scores = []
    for fold in folds:
        fit_options = {}
        if spec.early_stopping_patience is not None:
            fit_options |= {
                "eval_data": fold.validation,
                "patience": spec.early_stopping_patience,
                "metric": metric,
            }
        if report is not None:
            fit_options |= {
                "eval_data": fold.validation,
                "train_sample": fold.train_sample,
                "on_epoch": lambda epoch, predictions: report(
                    epoch,
                    {
                        **validation_and_train(
                            predictions["validation"], predictions["train"]
                        ),
                        "epoch": epoch,
                    },
                ),
            }
        model = train(
            spec,
            model_key,
            fold.train,
            trial_params,
            output_dir=str(OUTPUT_DIR / "tuning" / spec.run_name(model_key)),
            **fit_options,
        )
        fold_score = validation_and_train(
            spec.predict_include(model, fold.validation),
            spec.predict_include(model, fold.train_sample),
        )
        if spec.by_epoch:
            fold_score["epoch"] = model.best_epoch
        fold_scores.append(fold_score)

    if len(fold_scores) == 1:
        return fold_scores[0]
    return average_folds(fold_scores, metric)


def validation_and_train(
    validation: tuple[np.ndarray, np.ndarray], train_sample: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    """Scores from (probabilities, labels) on validation, and on the train sample as train_*."""
    train_scores = scores(*train_sample)
    return {
        **scores(*validation),
        **{f"train_{name}": value for name, value in train_scores.items()},
    }


def average_folds(fold_scores: list[dict[str, float]], metric: str) -> dict[str, float]:
    """Average each metric over the folds; add the tuning metric's std and per-fold scores
    (e.g. average_precision_std, average_precision_fold_0, ...)."""
    averaged = {}
    for name in fold_scores[0]:
        averaged[name] = float(np.mean([fold[name] for fold in fold_scores]))

    tuning_scores = [fold[metric] for fold in fold_scores]
    averaged[f"{metric}_std"] = float(np.std(tuning_scores))
    for k, score in enumerate(tuning_scores):
        averaged[f"{metric}_fold_{k}"] = score
    return averaged


def train(
    spec: ModelSpec,
    model_key: str,
    train_data: Dataset,
    params: dict | None = None,
    output_dir: str | None = None,
    **fit_options: Any,
) -> Any:
    """Train one model with spec.defaults[model_key] overridden by `params`; `fit_options`
    go to spec.fit (by_epoch models only, see ModelSpec)."""
    params = params or {}
    config = spec.defaults[model_key]
    check_keys(params, config, "params")
    return spec.fit(
        model_key,
        train_data,
        {**config, **params},
        seed=TRAIN_SEED,
        max_length=spec.max_lengths.get(model_key),
        output_dir=output_dir or str(OUTPUT_DIR / "models" / spec.run_name(model_key)),
        **fit_options,
    )


def parse_settings(items: list[str]) -> tuple[dict, dict]:
    """--set values, read as Python, by type: one value (0.5, True, balanced) fixes the setting;
    low,high or [a,b] searches a range or choices; a name alone searches config.py's range."""
    params, search = {}, {}
    for item in items:
        key, sep, text = item.partition("=")
        if not key:
            raise ValueError(f"--set {item!r} has no setting name")
        if not sep:
            search[key] = None
            continue
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            params[key] = text  # not a Python value: kept as text, e.g. balanced
            continue
        if isinstance(value, tuple):
            numbers = all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in value
            )
            if len(value) != 2 or not numbers:
                raise ValueError(f"--set {item!r}: a range is two numbers, low,high")
            search[key] = value
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"--set {item!r}: choices need at least one value")
            search[key] = value
        else:
            params[key] = value
    return params, search


def make_cli(spec: ModelSpec) -> Callable[[], None]:
    """The command-line entry point (for [project.scripts]) that runs run_experiment for `spec`."""

    def run(
        task: Annotated[str, typer.Argument(help="Classification task, e.g. inout")],
        model: Annotated[str, typer.Option(help=f"One of {list(spec.defaults)}")],
        trials: Annotated[
            int | None,
            typer.Option(
                min=0,
                help=f"Optuna trials for the search; 0 skips tuning and trains with the model's defaults [default: {spec.default_trials}]",
            ),
        ] = None,
        set_: Annotated[
            list[str] | None,
            typer.Option(
                "--set",
                help="A setting, in Python syntax, by type: one value fixes it (C=0.5, binary=True); low,high searches a range (C=1e-3,10), [a,b,...] choices (min_df=[2,5]); a name alone searches config.py's range (C). Repeat for several. Once any setting is searched, only those are [default: search everything in config.py, fix nothing]",
            ),
        ] = None,
        validation_only: Annotated[
            bool,
            typer.Option(
                help="Tune and score on validation only: no final model, the test set is never used"
            ),
        ] = False,
        split: Annotated[
            str | None,
            typer.Option(
                help="How tuning scores each trial: holdout (train / validation) or cv<n> (mean over n folds, e.g. cv5); the final model trains on all of it either way [default: "
                + spec.default_split
                + "]"
            ),
        ] = None,
        early_stopping_patience: Annotated[
            int | None,
            typer.Option(
                min=0,
                help="Transformers only: stop each tuning fit once the tuning metric on validation hasn't improved for this many epochs, and train the final model for the best trial's best epoch; 0 turns it off [default: config.py's]",
            ),
        ] = None,
        prune_from_epoch: Annotated[
            int | None,
            typer.Option(
                min=1,
                help="Transformers only: turns pruning on; from this epoch on, tuning stops trials that fall behind earlier ones (holdout only). Leave out for no pruning [default: config.py's]",
                show_default=False,
            ),
        ] = None,
    ) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        try:
            params, search = parse_settings(set_ or [])
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        try:
            run_experiment(
                spec,
                task,
                model,
                trials,
                params=params,
                search=search,
                validation_only=validation_only,
                split=split,
                early_stopping_patience=early_stopping_patience,
                prune_from_epoch=prune_from_epoch,
            )
        except BadArguments as e:
            raise typer.BadParameter(str(e)) from e

    run.__doc__ = (
        f"Tune on the {spec.default_split} split by default (see --split), train the final "
        "model on everything but test, and evaluate it once on test. Writes the model and "
        f"results.json to a new folder, results/<task>/{spec.method}-<model>/<timestamp>/."
    )

    def main() -> None:
        typer.run(run)

    return main
