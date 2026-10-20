"""Build data/processed/<task>/labels.parquet from adjudicated label files.

Raw files under data/labeling/<task>/ are the source of truth; this script only reads them.
It merges them, applies the task's filter, and adds one split_<name> column per setup in the
task's "splits" list (see splits.py). Settings are in tasks.py. Re-run whenever more
resolutions are completed, or after changing the task's settings:

    uv run build-labels inout
    uv run build-labels --help
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from gami2_map.data.splits import (
    ID_COLUMN,
    LABEL_COLUMN,
    SEED,
    SETTINGS_KEY,
    add_split_columns,
)
from gami2_map.paths import available_tasks, task_labeling_dir, task_labels_path
from gami2_map.tasks import task_config

logger = logging.getLogger(__name__)


def read_resolved(path: Path, labeling_dir: Path, id_column: str) -> pd.DataFrame:
    """Read one resolved CSV and tag each row with where it came from."""
    df = pd.read_csv(path, dtype={id_column: str})
    df["source_file"] = str(path.relative_to(labeling_dir))
    return df


def merge_resolved(
    files: list[Path],
    labeling_dir: Path,
    label_column: str,
    id_column: str,
    text_columns: str | list[str],
) -> pd.DataFrame:
    """Stack all resolved files into one table; each paper must appear exactly once.

    Only the id, text and label columns are required. The id and label columns come out
    renamed to item_id/label; each text column keeps its name, with blanks filled with ""
    and whitespace stripped. Every other column is kept as-is. Rows whose label is still
    empty are reported and left out.
    """
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    df = pd.concat(
        [read_resolved(f, labeling_dir, id_column) for f in files], ignore_index=True
    )

    missing = {id_column, label_column, *text_columns} - set(df.columns)
    if missing:
        raise KeyError(
            f"Columns {sorted(missing)} not found in resolved files; available: {list(df.columns)}"
        )

    dupes = df[df.duplicated(id_column, keep=False)]
    if not dupes.empty:
        details = dupes.sort_values(id_column)[
            [id_column, label_column, "source_file"]
        ].to_string(index=False)
        raise ValueError(
            f"{dupes[id_column].nunique()} papers appear more than once in the resolved files. "
            f"Fix the raw data before building:\n{details}"
        )

    pending = df[df[label_column].isna()]
    if not pending.empty:
        logger.warning(
            f"{len(pending):,} papers not yet resolved (left out):\n"
            + pending.groupby("source_file").size().to_string()
        )
    df = df.dropna(subset=[label_column])

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.rename(columns={id_column: ID_COLUMN, label_column: LABEL_COLUMN})
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame, text_columns: list[str]) -> None:
    logger.info(f"{len(df):,} labelled papers, {df[LABEL_COLUMN].mean():.1%} include")
    logger.info(
        "Empty: "
        + "; ".join(f"{col} {(df[col] == '').sum():,}" for col in text_columns)
    )
    logger.info(
        "Per file:\n"
        + df.groupby("source_file")[LABEL_COLUMN]
        .agg(["size", "sum", "mean"])
        .to_string()
    )


def write_labels(df: pd.DataFrame, path: Path, settings: dict) -> None:
    """Write labels.parquet with `settings` (split seed, test size, counts, fingerprints)
    stored in its metadata; load.load_settings reads them back."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {**(table.schema.metadata or {}), SETTINGS_KEY: json.dumps(settings)}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def build(
    task: Annotated[
        str,
        typer.Argument(
            help="Classification task, e.g. inout (a folder under data/labeling)"
        ),
    ],
    labeling_dir: Annotated[
        Path | None,
        typer.Option(
            help="Root folder of the raw labelling data [default: data/labeling/<task>]"
        ),
    ] = None,
    resolved_glob: Annotated[
        str | None,
        typer.Option(
            help="Glob for resolved files, relative to labeling-dir [default: from tasks.py]"
        ),
    ] = None,
    id_column: Annotated[
        str | None,
        typer.Option(
            help="Column identifying each paper; must be unique across files [default: from tasks.py]"
        ),
    ] = None,
    text_column: Annotated[
        list[str] | None,
        typer.Option(
            help="Column holding paper text; repeat for several, e.g. title and text [default: from tasks.py]"
        ),
    ] = None,
    label_column: Annotated[
        str | None,
        typer.Option(
            help="Column holding the final adjudicated label [default: from tasks.py]"
        ),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            help='Papers to keep, as a pandas query, e.g. "not abstract_missing" [default: from tasks.py]'
        ),
    ] = None,
    test_size: Annotated[
        float | None,
        typer.Option(
            help="Fraction of papers held out as test; holdout also uses it for validation [default: from tasks.py]"
        ),
    ] = None,
    stratify: Annotated[
        list[str] | None,
        typer.Option(
            help="Column to stratify the splits on besides the label; repeat for several [default: from tasks.py]"
        ),
    ] = None,
    seed: Annotated[
        str,
        typer.Option(
            help="Hash salt for the splits; a new value gives different splits"
        ),
    ] = SEED,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the parquet [default: data/processed/<task>/labels.parquet]"
        ),
    ] = None,
) -> None:
    """Build labels.parquet, with split columns, from one task's adjudicated label files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if labeling_dir is None and task not in available_tasks():
        raise typer.BadParameter(
            f"Unknown task {task!r}; available: {available_tasks()}"
        )
    config = task_config(task)
    resolved_glob = resolved_glob or config["resolved_glob"]
    id_column = id_column or config["id_column"]
    text_column = text_column or config["text_columns"]
    label_column = label_column or config["label_column"]
    filter = filter or config.get("filter")
    test_size = test_size or config["test_size"]
    stratify = stratify or config.get("stratify", [])
    labeling_dir = labeling_dir or task_labeling_dir(task)
    output = output or task_labels_path(task)

    files = sorted(labeling_dir.glob(resolved_glob))
    if not files:
        raise typer.BadParameter(f"No files match {resolved_glob} under {labeling_dir}")
    logger.info(f"Reading {len(files)} resolved files")

    df = merge_resolved(
        files,
        labeling_dir,
        label_column=label_column,
        id_column=id_column,
        text_columns=text_column,
    )
    if filter:
        kept = df.query(filter).reset_index(drop=True)
        logger.info(f"Filter {filter!r} dropped {len(df) - len(kept):,} papers")
        df = kept
    summarise(df, text_column)

    previous = pd.read_parquet(output) if output.exists() else None
    df, settings = add_split_columns(
        df, config["splits"], test_size, seed, stratify=stratify, previous=previous
    )

    write_labels(df, output, settings)
    logger.info(f"Wrote {output}")


def main() -> None:
    typer.run(build)


if __name__ == "__main__":
    main()
