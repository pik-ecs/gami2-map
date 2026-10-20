"""Read labels.parquet, written by `build-labels`, for the models and notebooks.

from gami2_map.data.load import load_huggingface_splits, load_labels

df = load_labels("inout")                                # DataFrame with split_<name> columns
ds = load_huggingface_splits("inout")                    # holdout: test, train, validation
ds = load_huggingface_splits("inout", split="cv5")       # test, fold_0 ... fold_4
"""

import json

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

from gami2_map.data.splits import SETTINGS_KEY, split_column
from gami2_map.paths import task_labels_path


def load_labels(task: str) -> pd.DataFrame:
    """labels.parquet for `task`: one row per paper, with a split_<name> column per setup."""
    return pd.read_parquet(task_labels_path(task))


def load_settings(task: str) -> dict:
    """The split settings stored in labels.parquet: seed, test_size, and per setup in
    "splits" its column, [papers, includes] counts per group, and fingerprint."""
    metadata = pq.read_schema(task_labels_path(task)).metadata
    return json.loads(metadata[SETTINGS_KEY.encode()])


def load_huggingface_splits(task: str, split: str = "holdout") -> DatasetDict:
    """One Dataset per group of a split setup: "test", plus "train" and "validation" for
    holdout, or "fold_0", "fold_1", ... for CV."""
    column = split_column(split)
    all_papers = Dataset.from_pandas(load_labels(task), preserve_index=False)
    if column not in all_papers.column_names:
        raise KeyError(
            f"labels.parquet for {task!r} has no {column} column; add {split!r} to the "
            f'task\'s "splits" in tasks.py and re-run `uv run build-labels {task}`'
        )
    groups = ["test"] + sorted(set(all_papers[column]) - {"test"})
    return DatasetDict(
        {
            group: all_papers.filter(
                lambda g, group=group: g == group, input_columns=column
            )
            for group in groups
        }
    )
