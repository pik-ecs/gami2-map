"""Fine-tune a transformer classifier (e.g. BERT) on one task's labels.

Reads data/processed/<task>/labels.parquet (written by `build-labels`) and its holdout split,
then (see pipeline.run_experiment):

    1. joins the text columns (default title + text) into one model input and tokenizes it
    2. tunes hyperparameters with Optuna on train, scoring the task's metric (tasks_settings.py)
       on validation (skipped with --trials 0); each fit stops early once validation stops
       improving, and with --prune-from-epoch, trials falling behind are stopped (see
       config.py's TRANSFORMER)
    3. trains the final model with the best settings, for the best trial's best epoch, on
       train + validation combined
    4. evaluates it once on test

and writes the model and its results to:

    results/<task>/transformer-<model>/<timestamp>/model/          (the trained model)
    results/<task>/transformer-<model>/<timestamp>/results.json    (settings used, split used, test metrics)

Run after `build-labels`:

    uv run train-transformer inout --model bert
    uv run train-transformer inout --model bert --trials 20
    uv run train-transformer inout --model bert --trials 0     (no tuning: the defaults in config.py)
    uv run train-transformer inout --model bert --early-stopping-patience 0
    uv run train-transformer inout --model bert --prune-from-epoch 3
    uv run train-transformer --help
"""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from gami2_map.models.config import TRANSFORMER, TRANSFORMER_LOG_SCALE
from gami2_map.models.metrics import TUNING_METRICS, scores
from gami2_map.models.pipeline import ModelSpec, make_cli

NUM_LABELS = 2  # exclude (0) / include (1)


def tokenize_huggingface_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: AutoTokenizer,
    text_column: str = "text",
    max_length: int = 512,
    truncation: bool = True,
) -> Dataset | DatasetDict:
    """Tokenize a Dataset or DatasetDict.

    Args:
        dataset: Dataset or DatasetDict to tokenize.
        tokenizer: Tokenizer to apply, e.g. AutoTokenizer.from_pretrained("bert-base-uncased").
        text_column: Column holding the text to tokenize.
        max_length: Maximum sequence length in tokens.
        truncation: Whether to cut sequences longer than max_length.

    Returns:
        The same Dataset or DatasetDict with the tokenizer's columns added.
    """

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch[text_column], max_length=max_length, truncation=truncation
        )

    return dataset.map(tokenize_batch, batched=True)


def checkpoint_for(model_key: str) -> str:
    """The pretrained model for a --model name, e.g. bert -> bert-base-uncased."""
    return TRANSFORMER["checkpoints"][model_key]["checkpoint"]


def prepare(data: DatasetDict, model_key: str, max_length: int) -> DatasetDict:
    """Tokenize the "input" column with the model's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_for(model_key))
    return tokenize_huggingface_dataset(
        data, tokenizer, text_column="input", max_length=max_length
    )


def _logits(predictions) -> np.ndarray:
    return (
        predictions[0] if isinstance(predictions, tuple) else predictions
    )  # some models also return hidden states


def _include_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax probability of the include class (1)."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=-1)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """scores() on the include probabilities.

    Trainer calls this on evaluate() and reports each key with an "eval_" (or "test_") prefix.
    """
    return scores(
        _include_probabilities(_logits(eval_pred.predictions)), eval_pred.label_ids
    )


def include_probabilities(
    trainer: Trainer, dataset: Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """The model's probability of "include" for every paper in `dataset`, and the true labels."""
    output = trainer.predict(dataset)
    return _include_probabilities(_logits(output.predictions)), output.label_ids


def class_weights(labels: list[int], class_weight: str | None) -> torch.Tensor | None:
    """Loss weight per class. "balanced" weights each class by n_papers / (2 * n_in_class), as
    sklearn does, so the rarer include class counts as much as exclude in total; None keeps
    the plain, unweighted loss."""
    if class_weight is None:
        return None
    if class_weight != "balanced":
        raise ValueError(
            f"class_weight must be 'balanced' or None, not {class_weight!r}"
        )
    counts = np.bincount(labels, minlength=NUM_LABELS)
    return torch.tensor(len(labels) / (NUM_LABELS * counts), dtype=torch.float)


class WeightedLossTrainer(Trainer):
    """A Trainer whose cross-entropy loss weights each class by `class_weights`."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.class_weights is None:
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


class EpochScores(TrainerCallback):
    """After each epoch, predicts on each of `datasets` ({"validation": ..., maybe "train":
    ...}) and calls on_epoch(epoch, {name: (include probabilities, true labels)}): how tuning
    trials report progress, so bad ones are pruned."""

    def __init__(self, datasets: dict[str, Dataset], on_epoch):
        self.datasets = datasets
        self.on_epoch = on_epoch
        self.trainer: Trainer | None = None  # set once the Trainer exists

    def on_epoch_end(self, args, state, control, **kwargs):
        predictions = {
            name: include_probabilities(self.trainer, dataset)
            for name, dataset in self.datasets.items()
        }
        self.on_epoch(round(state.epoch), predictions)


class KeepBestEpoch(TrainerCallback):
    """Early stopping: after each epoch's evaluation, keeps a copy of the weights if `metric`
    improved, stops training once it hasn't for `patience` epochs, and puts the best epoch's
    weights back at the end (its number in best_epoch).

    Weights are copied in memory rather than through Trainer's load_best_model_at_end, which
    saves BERT's LayerNorm under its old gamma / beta names and then fails to load them back."""

    def __init__(self, metric: str, patience: int):
        self.key = f"eval_{metric}"
        self.maximize = TUNING_METRICS[metric] == "maximize"
        self.patience = patience
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.best_weights: dict[str, torch.Tensor] | None = None
        self.epochs_without_improvement = 0

    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        score = metrics[self.key]
        improved = self.best_score is None or (
            score > self.best_score if self.maximize else score < self.best_score
        )
        if improved:
            self.best_score, self.best_epoch = score, round(state.epoch)
            self.best_weights = {
                name: tensor.detach().to("cpu", copy=True)
                for name, tensor in model.state_dict().items()
            }
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            control.should_training_stop = (
                self.epochs_without_improvement >= self.patience
            )

    def on_train_end(self, args, state, control, model, **kwargs):
        model.load_state_dict(self.best_weights)


def fit(
    model_key: str,
    train_data: Dataset,
    params: dict,
    *,
    seed: int,
    max_length: int,
    output_dir: str,
    eval_data: Dataset | None = None,
    on_epoch=None,
    train_sample: Dataset | None = None,
    patience: int | None = None,
    metric: str = "f1",
) -> Trainer:
    """Fine-tune one model with the full set of `params` (config.py's TRANSFORMER defaults)
    and return its Trainer, with the epoch whose weights it kept as trainer.best_epoch.
    `train_data` is already tokenized to `max_length` by prepare.

    With `eval_data` and `on_epoch` (tuning with pruning), the model is scored on `eval_data`
    (and `train_sample`, if given, to compare with: see EpochScores) after each epoch and
    on_epoch(epoch, {"validation": (probabilities, labels), "train": ...}) is called; it may
    raise optuna.TrialPruned to stop training.

    With `eval_data` and `patience` (early stopping), the model is also evaluated on
    `eval_data` after each epoch, training stops once `metric` (a metrics.TUNING_METRICS key)
    hasn't improved for that many epochs, and the best epoch's weights are kept.
    params["epochs"] is then the most it trains for; without early stopping, it trains for
    exactly that many and best_epoch is the last."""
    early_stopping = patience is not None
    if early_stopping and eval_data is None:
        raise ValueError("early stopping needs eval_data to score each epoch on")
    checkpoint = checkpoint_for(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=NUM_LABELS
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch" if early_stopping else "no",
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        # transformers 5 dropped warmup_ratio; a float below 1 here is read as that ratio
        warmup_steps=float(params["warmup_ratio"]),
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"],
        fp16=TRANSFORMER["use_amp"] and torch.cuda.is_available(),
        save_strategy="no",
        logging_strategy="no",
        seed=seed,
        report_to="none",
    )

    callbacks = []
    if eval_data is not None and on_epoch is not None:
        datasets = {"validation": eval_data}
        if train_sample is not None:
            datasets["train"] = train_sample
        callbacks.append(EpochScores(datasets, on_epoch))
    if early_stopping:
        keep_best = KeepBestEpoch(metric, patience)
        callbacks.append(keep_best)
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data if early_stopping else None,
        processing_class=AutoTokenizer.from_pretrained(checkpoint),
        compute_metrics=compute_metrics,
        class_weights=class_weights(train_data["label"], TRANSFORMER["class_weight"]),
        callbacks=callbacks,
    )
    for callback in callbacks:
        callback.trainer = trainer
    trainer.train()
    trainer.best_epoch = keep_best.best_epoch if early_stopping else params["epochs"]
    return trainer


def save(trainer: Trainer, path: Path) -> None:
    trainer.save_model(str(path))


SPEC = ModelSpec(
    method="transformer",
    checkpoints={key: entry["checkpoint"] for key, entry in TRANSFORMER["checkpoints"].items()},
    max_lengths={key: entry["max_length"] for key, entry in TRANSFORMER["checkpoints"].items()},
    # The shared defaults, overridden by each checkpoint's own
    defaults={
        key: {**TRANSFORMER["defaults"], **entry["defaults"]}
        for key, entry in TRANSFORMER["checkpoints"].items()
    },
    search_spaces={key: TRANSFORMER["search_space"] for key in TRANSFORMER["checkpoints"]},
    prepare=prepare,
    fit=fit,
    predict_include=include_probabilities,
    save=save,
    log_scale=TRANSFORMER_LOG_SCALE,
    class_weight=TRANSFORMER["class_weight"],
    by_epoch=True,
    early_stopping_patience=TRANSFORMER["early_stopping_patience"],
    prune_from_epoch=TRANSFORMER["prune_from_epoch"],
    default_trials=20,
)

main = make_cli(SPEC)


if __name__ == "__main__":
    main()
