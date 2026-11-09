"""Per-task settings:

    "data"      the raw resolved files under data/labeling/<task>/ and how build-labels
                relabels, filters and splits them
    "metric"    what tuning optimises, for every model

What each experiment runs is set in scripts/run_*.sh, not here.
Command-line options override these for a single run.
"""

TASKS = {
    "inout": {
        # Everything build-labels reads: where the labelled papers are, which columns hold
        # what, and how they are relabelled, filtered and split
        "data": {
            # Resolved files, relative to data/labeling/inout
            "resolved_glob": "**/04_resolved/*_resolved.csv",
            # Unique per paper across files
            "id_column": "item_id",
            # Title and abstract; joined into the model input
            "text_columns": ["title", "text"],
            # The final adjudicated label
            "label_column": "resolved",
            # Labels changed after resolution. Leave out for none.
            "relabels": "relables.csv",
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
        },
        # What tuning optimises on validation (see models/metrics.py); test reports all.
        "metric": "average_precision",
    },
}
