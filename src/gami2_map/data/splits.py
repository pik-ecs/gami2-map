"""Assign every labelled paper to test / development data for one split configuration.

Reads data/processed/<task>/labels.parquet and writes one file per run, named after the
configuration:

    data/processed/<task>/splits/test_20/holdout_gami2-v1.json       (defaults)
    data/processed/<task>/splits/test_20/cv5_gami2-v1.json   (--strategy cv5)

Assignment is hash-ranked and stratified: within each class, papers are ordered by a hash of
seed + item_id and cut at the exact fractions, so every split has the same include rate as the
full data. Files with the same test size and seed share the same test set, and re-running after
new resolutions only moves papers that sit right at a cut-off. Run after `build-labels`:

    uv run make-splits inout
    uv run make-splits inout --strategy cv5
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from gami2_map.paths import available_tasks, task_labels_path, task_splits_dir

logger = logging.getLogger(__name__)

ID_COLUMN = "item_id"
LABEL_COLUMN = "label"  # 1 = include, 0 = exclude

VERSION = 1
TEST_SIZE = (
    0.20  # holdout also uses it as the validation fraction of what's left after test
)
STRATEGY = "holdout"  # "cv<n>" for n-fold stratified CV

SEED = "gami2-v1"  # Hash salt, not an RNG seed: a new string gives a new, equally balanced split

Assignment = dict[str, str | int | None]


def _hash_key(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def _ranked_by_class(labels: pd.DataFrame, seed: str) -> list[list[str]]:
    """Item ids per class, each list ordered by hash: the fixed order every cut is taken from."""
    return [
        sorted(
            group[ID_COLUMN].astype(str), key=lambda item_id: _hash_key(seed, item_id)
        )
        for _, group in labels.groupby(LABEL_COLUMN)
    ]


def _cut(n: int, fraction: float) -> int:
    """How many of n items make up `fraction`, rounding halves up."""
    return int(n * fraction + 0.5)


def holdout_assignments(
    labels: pd.DataFrame, test_size: float, validation_size: float, seed: str
) -> dict[str, Assignment]:
    """Per class: the first test_size of the ranking is test, the next validation_size of
    what's left is validation, the rest is train."""
    assignments: dict[str, Assignment] = {}
    for ids in _ranked_by_class(labels, seed):
        n_test = _cut(len(ids), test_size)
        development = ids[n_test:]
        n_validation = _cut(len(development), validation_size)
        for item_id in ids[:n_test]:
            assignments[item_id] = {"dataset": "test", "fold": None}
        for item_id in development[:n_validation]:
            assignments[item_id] = {"dataset": "validation", "fold": None}
        for item_id in development[n_validation:]:
            assignments[item_id] = {"dataset": "train", "fold": None}
    return assignments


def kfold_assignments(
    labels: pd.DataFrame, test_size: float, n_splits: int, seed: str
) -> dict[str, Assignment]:
    """Per class: the first test_size of the ranking is test; the rest is development, cut into
    n_splits consecutive chunks of the ranking (so a new paper only shifts the chunk edges)."""
    assignments: dict[str, Assignment] = {}
    for ids in _ranked_by_class(labels, seed):
        n_test = _cut(len(ids), test_size)
        development = ids[n_test:]
        for item_id in ids[:n_test]:
            assignments[item_id] = {"dataset": "test", "fold": None}
        for rank, item_id in enumerate(development):
            fold = rank * n_splits // len(development)
            assignments[item_id] = {"dataset": "development", "fold": fold}
    return assignments


def split_path(strategy: str, test_size: float, seed: str) -> str:
    """File path for one configuration, relative to the splits dir, e.g. "test_20/cv5_gami2-v1.json"."""
    return f"test_{round(test_size * 100)}/{strategy}_{seed}.json"


def split_settings(strategy: str, test_size: float, seed: str) -> dict:
    """Header written to a split file: "holdout", or "cv<n>" for stratified k-fold with n folds.

    Holdout's validation_size is always test_size (a fraction of the papers left after the test
    set), so the test_<pct> folder describes both.
    """
    settings = {"version": VERSION, "test_size": test_size}
    if strategy == "holdout":
        settings |= {
            "validation_strategy": "holdout",
            "validation_size": test_size,
        }
    elif strategy.startswith("cv") and strategy[2:].isdigit():
        settings |= {
            "validation_strategy": "stratified_kfold",
            "n_splits": int(strategy[2:]),
        }
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}; use 'holdout' or 'cv<n>', e.g. 'cv5'"
        )
    return settings | {"seed": seed}


def assign(labels: pd.DataFrame, settings: dict) -> dict[str, Assignment]:
    if settings["validation_strategy"] == "holdout":
        return holdout_assignments(
            labels, settings["test_size"], settings["validation_size"], settings["seed"]
        )
    return kfold_assignments(
        labels, settings["test_size"], settings["n_splits"], settings["seed"]
    )


def load_splits(
    task: str,
    test_size: float = TEST_SIZE,
    strategy: str = STRATEGY,
    seed: str = SEED,
) -> dict[str, Assignment]:
    """item_id → {"dataset", "fold"}, as written by `make-splits <task>`.

    `strategy` is "holdout" or "cv<n>", e.g. "cv5".
    """
    path = task_splits_dir(task) / split_path(strategy, test_size, seed)
    return json.loads(path.read_text())["assignments"]


def _log_changes(name: str, old: dict | None, new: dict[str, Assignment]) -> None:
    """Report how a re-run moved papers compared to the file it replaces."""
    if old is None:
        return
    old_assignments = old["assignments"]
    moved = sum(old_assignments[i] != new[i] for i in set(old_assignments) & set(new))
    added = len(set(new) - set(old_assignments))
    removed = len(set(old_assignments) - set(new))
    if moved or added or removed:
        logger.info(f"  {name}: {added} added, {removed} removed, {moved} moved")


def summarise(
    name: str, labels: pd.DataFrame, assignments: dict[str, Assignment]
) -> None:
    df = labels.assign(
        group=labels[ID_COLUMN]
        .astype(str)
        .map(
            lambda i: assignments[i]["dataset"]
            if assignments[i]["fold"] is None
            else f"fold {assignments[i]['fold']}"
        )
    )
    counts = df.groupby("group")[LABEL_COLUMN].agg(["size", "sum"])
    logger.info(
        f"{name}: "
        + ", ".join(
            f"{group} {row['size']} ({row['sum']} incl)"
            for group, row in counts.iterrows()
        )
    )


def make_splits(
    task: Annotated[
        str,
        typer.Argument(
            help="Classification task, e.g. inout (a folder under data/labeling)"
        ),
    ],
    labels_path: Annotated[
        Path | None,
        typer.Option(
            help="Labels written by build-labels [default: data/processed/<task>/labels.parquet]"
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the split files [default: data/processed/<task>/splits]"
        ),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(help='"holdout", or "cv<n>" for n-fold stratified CV, e.g. cv3'),
    ] = STRATEGY,
    test_size: Annotated[
        float, typer.Option(help="Fraction of papers held out as the test set")
    ] = TEST_SIZE,
    seed: Annotated[
        str, typer.Option(help="Hash salt; a new value gives a different split")
    ] = SEED,
) -> None:
    """Assign labelled papers to test / development data and write one split file."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        settings = split_settings(strategy, test_size, seed)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    if task not in available_tasks():
        raise typer.BadParameter(
            f"Unknown task {task!r}; available: {available_tasks()}"
        )
    labels_path = labels_path or task_labels_path(task)
    output_dir = output_dir or task_splits_dir(task)

    if not labels_path.exists():
        raise typer.BadParameter(
            f"{labels_path} not found; run `uv run build-labels {task}` first"
        )
    labels = pd.read_parquet(labels_path)
    if labels[ID_COLUMN].duplicated().any():
        raise ValueError(f"{labels_path} has duplicate {ID_COLUMN}s")
    logger.info(
        f"{len(labels):,} labelled papers, {labels[LABEL_COLUMN].sum():,} includes"
    )

    name = split_path(strategy, test_size, seed)
    assignments = assign(labels, settings)
    path = output_dir / name
    old = json.loads(path.read_text()) if path.exists() else None

    summarise(name, labels, assignments)
    _log_changes(name, old, assignments)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**settings, "assignments": dict(sorted(assignments.items()))}
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote {path}")


def main() -> None:
    typer.run(make_splits)


if __name__ == "__main__":
    main()
