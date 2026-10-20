"""Per-task settings: the raw resolved files under data/labeling/<task>/, and how build-labels
filters and splits them.

Add an entry when a new task's files arrive, and edit it when the folder layout, the
export's column names, the filter or the splits change; then re-run `build-labels <task>`.
Command-line options override these for a single run.
"""

TASKS = {
    "inout": {
        # Resolved files, relative to data/labeling/inout
        "resolved_glob": "**/04_resolved/*_resolved.csv",
        # Unique per paper across files
        "id_column": "item_id",
        # Title and abstract; joined into the model input
        "text_columns": ["title", "text"],
        # The final adjudicated label
        "label_column": "resolved",
        # Papers to keep (a pandas query on the merged table); leave out to keep all.
        # Drops papers with no abstract unless a coder marked them ineligible.
        "filter": "not abstract_missing or any_marked_ineligible",
        # Split setups, each a split_<name> column: "holdout" (train / validation / test)
        # or "cv<n>" (fold_0 ... / test). All share the same test set.
        "splits": ["holdout", "cv3", "cv5"],
        # Fraction held out as test; holdout also takes this fraction of the rest as validation
        "test_size": 0.20,
        # Columns to stratify the splits on besides the label; leave out for label only.
        # decision: unanimous vs adjudicated, so every split has the same share of hard cases
        "stratify": ["decision"],
        # What tuning optimises on validation (see models/metrics.py); test reports all.
        # f2 weights recall twice as much as precision: missing an include costs more
        "metric": "f2",
    },
}


def task_config(task: str) -> dict:
    """The TASKS entry for `task`."""
    if task not in TASKS:
        raise KeyError(f"No entry for task {task!r} in tasks.py; known: {list(TASKS)}")
    return TASKS[task]
