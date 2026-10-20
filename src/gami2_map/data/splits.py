"""Assign every labelled paper to train / validation / test, as columns of labels.parquet.

build-labels calls add_split_columns after filtering, adding one column per setup in the
task's "splits" list in tasks.py:

    split_holdout    "train" / "validation" / "test"
    split_cv5        "fold_0" ... "fold_4" / "test"

Assignment is hash-ranked and stratified: within each stratum (the label, plus any columns in
the task's "stratify" list, e.g. decision), papers are ordered by a hash of seed + item_id and
cut at the exact fractions, so every split has the same mix of strata as the full data. All setups share the same test set, and rebuilding after new resolutions only moves
papers that sit right at a cut-off.
"""

import hashlib
import logging

import pandas as pd

logger = logging.getLogger(__name__)

ID_COLUMN = "item_id"
LABEL_COLUMN = "label"  # 1 = include, 0 = exclude

SEED = "gami2-v1"  # Hash salt, not an RNG seed: a new string gives a new, equally balanced split

# Key in labels.parquet's metadata holding the settings add_split_columns returns
SETTINGS_KEY = "gami2_map"

# item_id → the group it belongs to: "test", "train", "validation", or "fold_<k>"
Groups = dict[str, str]


def split_column(name: str) -> str:
    """Column holding one setup's assignments, e.g. "holdout" → "split_holdout"."""
    return f"split_{name}"


def _hash_key(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def _ranked_by_stratum(
    labels: pd.DataFrame, seed: str, stratify: list[str]
) -> list[list[str]]:
    """Item ids per stratum (label plus the `stratify` columns; a missing value is its own
    stratum), each list ordered by hash: the fixed order every cut is taken from."""
    return [
        sorted(
            group[ID_COLUMN].astype(str), key=lambda item_id: _hash_key(seed, item_id)
        )
        for _, group in labels.groupby([LABEL_COLUMN, *stratify], dropna=False)
    ]


def _cut(n: int, fraction: float) -> int:
    """How many of n items make up `fraction`, rounding halves up."""
    return int(n * fraction + 0.5)


def holdout_assignments(
    labels: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: str,
    stratify: list[str],
) -> Groups:
    """Per stratum: the first test_size of the ranking is test, the next validation_size of
    what's left is validation, the rest is train."""
    groups: Groups = {}
    for ids in _ranked_by_stratum(labels, seed, stratify):
        n_test = _cut(len(ids), test_size)
        development = ids[n_test:]
        n_validation = _cut(len(development), validation_size)
        groups |= dict.fromkeys(ids[:n_test], "test")
        groups |= dict.fromkeys(development[:n_validation], "validation")
        groups |= dict.fromkeys(development[n_validation:], "train")
    return groups


def kfold_assignments(
    labels: pd.DataFrame,
    test_size: float,
    n_splits: int,
    seed: str,
    stratify: list[str],
) -> Groups:
    """Per stratum: the first test_size of the ranking is test; the rest is development, cut into
    n_splits consecutive chunks of the ranking (so a new paper only shifts the chunk edges)."""
    groups: Groups = {}
    for ids in _ranked_by_stratum(labels, seed, stratify):
        n_test = _cut(len(ids), test_size)
        development = ids[n_test:]
        groups |= dict.fromkeys(ids[:n_test], "test")
        for rank, item_id in enumerate(development):
            groups[item_id] = f"fold_{rank * n_splits // len(development)}"
    return groups


def assign(
    labels: pd.DataFrame, name: str, test_size: float, seed: str, stratify: list[str]
) -> Groups:
    """Assignments for one setup: "holdout" (validation is test_size of what's left after
    test), or "cv<n>" for stratified n-fold CV."""
    if name == "holdout":
        return holdout_assignments(labels, test_size, test_size, seed, stratify)
    if name.startswith("cv") and name[2:].isdigit():
        return kfold_assignments(labels, test_size, int(name[2:]), seed, stratify)
    raise ValueError(f"Unknown split {name!r}; use 'holdout' or 'cv<n>', e.g. 'cv5'")


def counts(labels: pd.DataFrame, column: str) -> dict[str, list[int]]:
    """[papers, includes] per group in `column`."""
    by_group = labels.groupby(column)[LABEL_COLUMN].agg(["size", "sum"])
    return {str(group): [int(n), int(k)] for group, (n, k) in by_group.iterrows()}


def fingerprint(labels: pd.DataFrame, column: str) -> str:
    """Short hash of every paper's (item_id, label, group): equal fingerprints mean two runs
    used exactly the same papers, labels and splits."""
    rows = labels[[ID_COLUMN, LABEL_COLUMN, column]].astype(str).sort_values(ID_COLUMN)
    text = "\n".join(",".join(row) for row in rows.itertuples(index=False))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _log_changes(column: str, old: pd.DataFrame | None, new: pd.Series) -> None:
    """Report how a rebuild moved papers compared to the labels.parquet it replaces."""
    if old is None or column not in old:
        return
    old_groups = old.set_index(ID_COLUMN)[column]
    shared = old_groups.index.intersection(new.index)
    moved = int((old_groups[shared] != new[shared]).sum())
    added = len(new.index.difference(old_groups.index))
    removed = len(old_groups.index.difference(new.index))
    if moved or added or removed:
        logger.info(f"  {column}: {added} added, {removed} removed, {moved} moved")


def add_split_columns(
    labels: pd.DataFrame,
    names: list[str],
    test_size: float,
    seed: str = SEED,
    stratify: list[str] | None = None,
    previous: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Add a split_<name> column per setup, stratified on the label plus the `stratify`
    columns. Returns the labels and the settings to store in the parquet metadata: seed,
    test size, stratify columns, and each setup's counts and fingerprint.
    `previous` is the labels.parquet being replaced, used only to log what moved."""
    stratify = list(stratify or [])
    missing = set(stratify) - set(labels.columns)
    if missing:
        raise KeyError(
            f"Stratify columns {sorted(missing)} not found; available: {list(labels.columns)}"
        )
    sizes = labels.groupby([LABEL_COLUMN, *stratify], dropna=False).size()
    logger.info(
        f"Strata ({', '.join([LABEL_COLUMN, *stratify])}):\n{sizes.to_string()}"
    )
    if (sizes < 10).any():
        logger.warning(
            "Some strata have fewer than 10 papers, so their splits will be uneven: "
            + ", ".join(str(key) for key in sizes[sizes < 10].index)
        )

    labels = labels.copy()
    settings = {
        "seed": seed,
        "test_size": test_size,
        "stratify": stratify,
        "splits": {},
    }
    for name in names:
        column = split_column(name)
        groups = assign(labels, name, test_size, seed, stratify)
        labels[column] = labels[ID_COLUMN].map(groups)
        settings["splits"][name] = {
            "column": column,
            "counts": counts(labels, column),
            "fingerprint": fingerprint(labels, column),
        }
        logger.info(
            f"{column}: "
            + ", ".join(
                f"{group} {n} ({k} incl)"
                for group, (n, k) in settings["splits"][name]["counts"].items()
            )
        )
        _log_changes(column, previous, labels.set_index(ID_COLUMN)[column])
    return labels, settings
