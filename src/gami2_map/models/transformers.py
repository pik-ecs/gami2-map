from datasets import DatasetDict, load_dataset

from gami2_map.data.splits import load_splits
from gami2_map.paths import task_labels_path

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "modernbert": "answerdotai/ModernBERT-base",
}


def make_DatasetDict(task: str, strategy: str, test_size: float = 0.20) -> DatasetDict:
    labels = load_dataset(
        "parquet", data_files=str(task_labels_path(task)), split="train"
    )
    splits = load_splits(task, strategy=strategy, test_size=test_size)

    def keep(ids):
        ids = set(ids)
        return labels.filter(lambda i: i in ids, input_columns="item_id")

    split_data_dictionary = {"test": keep(splits["test"])}
    for split, ids in splits["development"].items():
        split_data_dictionary[split] = keep(ids)

    return DatasetDict(split_data_dictionary)
