"""Project-wide paths, defined once so no module has to count folder levels itself."""

from pathlib import Path

# src/gami2_map/paths.py → parents: [0] gami2_map, [1] src, [2] project root
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
LABELING_DIR = DATA_DIR / "labeling"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"


def task_labeling_dir(task: str) -> Path:
    """Raw labelling data for one classification task, e.g. data/labeling/inout."""
    return LABELING_DIR / task


def task_labels_path(task: str) -> Path:
    """labels.parquet for one task, written by `build-labels <task>`."""
    return PROCESSED_DIR / task / "labels.parquet"


def task_splits_dir(task: str) -> Path:
    """Split files for one task (splits/test_<pct>/<strategy>_seed<n>.json), written by `make-splits <task>`."""
    return PROCESSED_DIR / task / "splits"


def available_tasks() -> list[str]:
    """Task names, i.e. the folders under data/labeling."""
    return sorted(p.name for p in LABELING_DIR.iterdir() if p.is_dir())
