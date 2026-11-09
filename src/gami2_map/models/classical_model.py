"""Train a classical classifier (logistic regression or linear SVM) on one task's labels.
--model picks the classifier and the features, as <classifier>-<features>:

    classifier    logreg, svm
    features      tfidf, count: bag-of-words (TF-IDF weights or raw counts), fit on the
                      training papers
                  mpnet, allenai-specter, bge-base, glove: frozen embeddings

    e.g. logreg-tfidf, svm-count, logreg-mpnet.
    - C is tuned for every model
    - For tfidf / count the following are also tuned
        - binary (word presence instead of counts),
        - ngram_max
        -  min_df;
    - the defaults settings are set under CLASSICAL in config.py.

Reads data/processed/<task>/labels.parquet (written by `build-labels`) and its cv5 split
(fold_0 ... fold_4 / test; --split holdout for train / validation / test), then:

    1. joins the text columns (default title + text) into one model input (and embeds it once,
       for the embedders)
    2. tunes C (and binary, ngram_max, min_df) with Optuna, scoring each trial by 5-fold CV:
       trained on 4 folds and scored on the 5th, for each fold, then averaged (skipped with
       --trials 0)
    3. trains the final model with the best settings on all 5 folds combined
    4. evaluates it once on test

and writes the model and its results to:

    results/<task>/classical-<model>/<timestamp>/model/classifier.joblib    (the trained model)
    results/<task>/classical-<model>/<timestamp>/results.json               (settings used, split used, test metrics)

Run after `build-labels`:

    uv run train-classical inout --model logreg-tfidf
    uv run train-classical inout --model logreg-mpnet --trials 30
    uv run train-classical inout --model svm-tfidf --trials 0     (no tuning: C from config.py)
    uv run train-classical inout --model svm-tfidf --split holdout  (tune on one validation set)
    uv run train-classical --help
"""

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from gami2_map.models.config import CLASSICAL, CLASSICAL_LOG_SCALE
from gami2_map.models.pipeline import ModelSpec, make_cli
from gami2_map.paths import EMBEDDINGS_DIR

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32


CLASSIFIER_AND_FEATURES: dict[str, tuple[str, str]] = {}
for classifier in CLASSICAL["classifiers"]:
    for features in [*CLASSICAL["vectorizers"], *CLASSICAL["embedders"]]:
        CLASSIFIER_AND_FEATURES[f"{classifier}-{features}"] = (classifier, features)


def make_vectorizer(name: str, params: dict) -> TfidfVectorizer | CountVectorizer:
    """A tfidf / count vectorizer with params["binary"], params["ngram_max"] and
    params["min_df"]."""
    settings = {
        "binary": params["binary"],
        "ngram_range": (1, params["ngram_max"]),
        "min_df": params["min_df"],
    }
    if name == "tfidf":
        # sublinear_tf: 1 + log(count) instead of count (no effect once binary is on)
        return TfidfVectorizer(**settings, sublinear_tf=True)
    return CountVectorizer(**settings)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embeddings_cache_path(checkpoint: str, max_length: int | None) -> Path:
    """One cache file per embedder and max_length, e.g.
    data/processed/embeddings/sentence-transformers__all-mpnet-base-v2_len384.parquet
    (no _len for an embedder without one, e.g. glove)."""
    length = f"_len{max_length}" if max_length else ""
    return EMBEDDINGS_DIR / f"{checkpoint.replace('/', '__')}{length}.parquet"


# 1.1 PREPARE: called by make_features, for embedders only
def make_embeddings(
    data: DatasetDict, checkpoint: str, max_length: int | None
) -> DatasetDict:
    """Embed the "input" column with a frozen sentence model into an "embedding" column.

    Embeddings are cached on disk by a hash of each input text (see embeddings_cache_path),
    so only texts not embedded before are run through the model"""
    path = embeddings_cache_path(checkpoint, max_length)
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        cached = pd.read_parquet(path)
        cache = dict(zip(cached["text_key"], cached["embedding"]))

    texts = {
        _text_key(text): text for split in data.values() for text in split["input"]
    }
    missing = {key: text for key, text in texts.items() if key not in cache}
    logger.info(
        f"Embeddings for {checkpoint}: {len(texts) - len(missing)} cached, "
        f"{len(missing)} to compute"
    )
    if missing:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(checkpoint)
        if max_length:
            embedder.max_seq_length = max_length
        vectors = embedder.encode(
            list(missing.values()),
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        cache |= dict(zip(missing, vectors))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"text_key": list(cache), "embedding": [list(v) for v in cache.values()]}
        ).to_parquet(path, index=False)

    return data.map(
        lambda batch: {
            "embedding": [cache[_text_key(text)] for text in batch["input"]]
        },
        batched=True,
    )


def make_classifier(name: str, C: float, seed: int) -> LogisticRegression | SVC:
    class_weight = CLASSICAL["class_weight"]
    if name == "logreg":
        return LogisticRegression(
            C=C, class_weight=class_weight, max_iter=1000, random_state=seed
        )
    if name == "svm":
        # Linear kernel: the usual choice for text, and it keeps C the only setting. probability=True fits Platt scaling (internal 5-fold CV)
        return SVC(
            kernel="linear",
            C=C,
            class_weight=class_weight,
            probability=True,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier {name!r}; use 'logreg' or 'svm'")


def make_features(
    data: DatasetDict, model_key: str, max_length: int | None
) -> DatasetDict:
    """Each model's input:
    1. for tfidf / count, the "input" text as is (the vectorizer is fit
    inside fit
    2. for an embedder, the embeddings from make_embeddings.
    """
    _, features = CLASSIFIER_AND_FEATURES[model_key]
    if features in CLASSICAL["vectorizers"]:
        return data.select_columns(["input", "label"])
    checkpoint = CLASSICAL["embedders"][features]["checkpoint"]
    return make_embeddings(data, checkpoint, max_length).select_columns(
        ["embedding", "label"]
    )


def _features(dataset: Dataset) -> list[str] | np.ndarray:
    """The model input from make_features: texts for tfidf / count, an embedding matrix otherwise."""
    if "embedding" in dataset.column_names:
        return dataset.with_format("numpy")[:]["embedding"]
    return list(dataset["input"])


def fit(
    model_key: str,
    train_data: Dataset,
    params: dict,
    *,
    seed: int,
    max_length: int | None,
    output_dir: str,
) -> ClassifierMixin:
    """Nothing
    is written to `output_dir`, and `max_length` was used in make_features."""
    classifier, features = CLASSIFIER_AND_FEATURES[model_key]
    model = make_classifier(classifier, params["C"], seed)
    if features in CLASSICAL["vectorizers"]:
        model = make_pipeline(make_vectorizer(features, params), model)
    return model.fit(_features(train_data), np.asarray(train_data["label"]))


def include_probabilities(
    model: ClassifierMixin, dataset: Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """The model's probability of "include" for every paper in `dataset`, and the true labels."""
    probabilities = model.predict_proba(_features(dataset))
    return probabilities[:, 1], np.asarray(dataset["label"])


def save(model: ClassifierMixin, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "classifier.joblib")


# tfidf / count
vectorizer = CLASSICAL["vectorizer"]
defaults, search_spaces = {}, {}
for key, (classifier, features) in CLASSIFIER_AND_FEATURES.items():
    settings = CLASSICAL["classifiers"][classifier]
    defaults[key] = settings["defaults"]
    search_spaces[key] = settings["search_space"]
    if features in CLASSICAL["vectorizers"]:
        defaults[key] = {**settings["defaults"], **vectorizer["defaults"]}
        search_spaces[key] = {
            **settings["search_space"],
            **vectorizer["search_space"],
        }

SPEC = ModelSpec(
    method="classical",
    defaults=defaults,
    search_spaces=search_spaces,
    # The embedders' max_length from config.py (none for tfidf / count, or glove)
    max_lengths={
        key: CLASSICAL["embedders"][features].get("max_length")
        for key, (_, features) in CLASSIFIER_AND_FEATURES.items()
        if features in CLASSICAL["embedders"]
    },
    checkpoints={
        key: CLASSICAL["embedders"][features]["checkpoint"]
        for key, (_, features) in CLASSIFIER_AND_FEATURES.items()
        if features in CLASSICAL["embedders"]
    },
    prepare=make_features,
    fit=fit,
    predict_include=include_probabilities,
    save=save,
    log_scale=CLASSICAL_LOG_SCALE,
    class_weight=CLASSICAL["class_weight"],
    default_trials=30,
    # Each trial is scored on 5 folds; classical models train in seconds, so it's cheap
    default_split="cv5",
)

main = make_cli(SPEC)


if __name__ == "__main__":
    main()
