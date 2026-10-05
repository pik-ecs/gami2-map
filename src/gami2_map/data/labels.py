"""Build data/processed/<task>/labels.parquet from adjudicated label files.

Raw files under data/labeling/<task>/ are the source of truth; this script only reads them.
Re-run whenever more resolutions are completed:

    uv run build-labels inout
    uv run build-labels --help
"""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from gami2_map.paths import available_tasks, task_labeling_dir, task_labels_path

logger = logging.getLogger(__name__)


def read_resolved(path: Path, labeling_dir: Path, id_column: str) -> pd.DataFrame:
    """Read one resolved CSV and tag each row with where it came from."""
    df = pd.read_csv(path, dtype={id_column: str})
    df["source_file"] = str(path.relative_to(labeling_dir))
    return df


def merge_resolved(
    files: list[Path],
    labeling_dir: Path,
    id_column: str = "item_id",
    text_columns: str | list[str] = ("title", "text"),
    label_column: str = "resolved",
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
    df = df.rename(columns={id_column: "item_id", label_column: "label"})
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


def summarise(df: pd.DataFrame, text_columns: list[str]) -> None:
    logger.info(f"{len(df):,} labelled papers, {df['label'].mean():.1%} include")
    logger.info(
        "Empty: "
        + "; ".join(f"{col} {(df[col] == '').sum():,}" for col in text_columns)
    )
    logger.info(
        "Per file:\n"
        + df.groupby("source_file")["label"].agg(["size", "sum", "mean"]).to_string()
    )


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
        str,
        typer.Option(help="Glob for resolved files, relative to labeling-dir"),
    ] = "**/04_resolved/*_resolved.csv",
    id_column: Annotated[
        str,
        typer.Option(help="Column identifying each paper; must be unique across files"),
    ] = "item_id",
    text_column: Annotated[
        list[str],
        typer.Option(
            help="Column holding paper text; repeat for several, e.g. title and text"
        ),
    ] = ["title", "text"],
    label_column: Annotated[
        str, typer.Option(help="Column holding the final adjudicated label")
    ] = "resolved",
    output: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the parquet [default: data/processed/<task>/labels.parquet]"
        ),
    ] = None,
) -> None:
    """Build labels.parquet from one task's adjudicated label files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if labeling_dir is None and task not in available_tasks():
        raise typer.BadParameter(
            f"Unknown task {task!r}; available: {available_tasks()}"
        )
    labeling_dir = labeling_dir or task_labeling_dir(task)
    output = output or task_labels_path(task)

    files = sorted(labeling_dir.glob(resolved_glob))
    if not files:
        raise typer.BadParameter(f"No files match {resolved_glob} under {labeling_dir}")
    logger.info(f"Reading {len(files)} resolved files")

    df = merge_resolved(files, labeling_dir, id_column, text_column, label_column)
    summarise(df, text_column)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info(f"Wrote {output}")


def main() -> None:
    typer.run(build)


if __name__ == "__main__":
    main()
