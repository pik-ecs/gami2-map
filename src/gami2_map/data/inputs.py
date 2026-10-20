"""Turning the splits from load.py into model input: one text column, optionally sampled.

from gami2_map.data.inputs import concat_input_text, sample_per_class

ds = concat_input_text(load_huggingface_splits("inout"), ["title", "text"])   # adds "input"
few_shot = sample_per_class(ds["train"], 16, ["decision"], seed=42)
"""

from datasets import Dataset, DatasetDict

from gami2_map.data.splits import LABEL_COLUMN


def concat_input_text(dataset: DatasetDict, text_columns: list[str]) -> DatasetDict:
    """Join the text columns (e.g. title and abstract) into one "input" column, skipping blanks."""

    def join(row: dict) -> dict:
        parts = [str(row[col]).strip() for col in text_columns if row[col]]
        return {"input": ". ".join(part for part in parts if part)}

    return dataset.map(join)


def sample_per_class(
    data: Dataset, n: int | None, stratify: list[str], seed: int
) -> Dataset:
    """At most n papers per class (a class with fewer is kept whole, never duplicated), drawn
    in proportion to the `stratify` columns (e.g. decision) within each class, so the sample
    keeps their mix; per-stratum rounding can make a class n ± 1. n=None returns `data`."""
    if n is None:
        return data
    df = data.select_columns([LABEL_COLUMN, *stratify]).to_pandas()
    keep: list[int] = []
    for _, papers in df.groupby(LABEL_COLUMN):
        if len(papers) <= n:
            keep += papers.index.tolist()
            continue
        strata = (
            papers.groupby(stratify, dropna=False) if stratify else [(None, papers)]
        )
        for _, stratum in strata:
            k = round(n * len(stratum) / len(papers))
            keep += stratum.sample(k, random_state=seed).index.tolist()
    return data.select(sorted(keep))
