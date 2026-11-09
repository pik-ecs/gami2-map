"""Train a SetFit classifier (contrastive fine-tune of a sentence embedder + logistic regression
head) on one task's labels.

Reads data/processed/<task>/labels.parquet (written by `build-labels`) and its holdout split,
then (see pipeline.run_experiment):

    1. joins the text columns (default title + text) into one model input
    2. tunes hyperparameters with Optuna on train, scoring the task's metric (tasks_settings.py)
       on validation (skipped with --trials 0)
    3. trains the final model with the best settings on train + validation combined
    4. evaluates it once on test

and writes the model and its results to:

    results/<task>/setfit-<model>/<timestamp>/model/          (the trained model)
    results/<task>/setfit-<model>/<timestamp>/results.json    (settings used, split used, test metrics)

Run after `build-labels`:

    uv run train-setfit inout --model mpnet
    uv run train-setfit inout --model mpnet --trials 20
    uv run train-setfit inout --model mpnet --trials 0   (no tuning: the defaults in config.py)
    uv run train-setfit --help
"""

from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from setfit import SetFitModel, Trainer, TrainingArguments

from gami2_map.models.config import SETFIT, SETFIT_LOG_SCALE
from gami2_map.models.pipeline import ModelSpec, make_cli


def prepare(data: DatasetDict, model_key: str, max_length: int) -> DatasetDict:
    """Keep only the "input" and "label" columns, with "input" renamed to the "text" SetFit
    expects. SetFit tokenizes during training, so `model_key` and `max_length` go unused here."""
    return data.select_columns(["input", "label"]).rename_column("input", "text")


def include_probabilities(
    model: SetFitModel, dataset: Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """The model's probability of "include" for every paper in `dataset`, and the true labels."""
    probabilities = model.predict_proba(dataset["text"], as_numpy=True)
    return probabilities[:, 1], np.asarray(dataset["label"])


def fit(
    model_key: str,
    train_data: Dataset,
    params: dict,
    *,
    seed: int,
    max_length: int,
    output_dir: str,
) -> SetFitModel:
    """Train one model with the full set of `params` (a config.py entry's defaults) and return it.
    `train_data` needs "text" and "label" columns (see prepare). `model_key` is the --model
    name, e.g. mpnet; its checkpoint is in config.py's SETFIT."""
    model = SetFitModel.from_pretrained(SETFIT["checkpoints"][model_key]["checkpoint"])

    args = TrainingArguments(
        output_dir=output_dir,
        body_learning_rate=params["body_learning_rate"],
        num_iterations=params["num_iterations"],
        batch_size=params["batch_size"],
        num_epochs=params["epochs"],
        max_length=max_length,
        save_strategy="no",
        logging_strategy="no",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_data)
    trainer.train()
    return model


def save(model: SetFitModel, path: Path) -> None:
    model.save_pretrained(str(path))


SPEC = ModelSpec(
    method="setfit",
    checkpoints={key: entry["checkpoint"] for key, entry in SETFIT["checkpoints"].items()},
    max_lengths={key: entry["max_length"] for key, entry in SETFIT["checkpoints"].items()},
    # Every SetFit model is tuned the same way
    defaults={key: SETFIT["defaults"] for key in SETFIT["checkpoints"]},
    search_spaces={key: SETFIT["search_space"] for key in SETFIT["checkpoints"]},
    prepare=prepare,
    fit=fit,
    predict_include=include_probabilities,
    save=save,
    log_scale=SETFIT_LOG_SCALE,
    default_trials=20,
)

main = make_cli(SPEC)


if __name__ == "__main__":
    main()
