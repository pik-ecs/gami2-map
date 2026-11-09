"""Scoring include probabilities, the same way in the pipeline and in notebooks.

from gami2_map.models.metrics import scores

scores(include_probabilities, true_labels)   # every metric below, as a dict
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
)

# Metrics tuning can optimise on validation, and in which direction. They only choose between
# settings; the weights are trained by each model's loss (see tuning.py). Recall, precision and
# accuracy are reported but left out: each can be maxed by flagging everything or nothing.
TUNING_METRICS = {
    "f1": "maximize",  # precision and recall weighted equally
    "f2": "maximize",  # recall weighted twice as much as precision
    # mean of recall on includes and on excludes; flagging everything or nothing scores 0.5
    "balanced_accuracy": "maximize",
    "average_precision": "maximize",  # ranking quality over every threshold
}


def predict_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Include (1) where the include probability is at least `threshold`, else exclude (0)."""
    return (probabilities >= threshold).astype(int)


def scores(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """At the `threshold` cut-off: accuracy, balanced accuracy, precision / recall / F1 / F2
    for the include class (label 1), the share of papers predicted include ("flagged", i.e.
    how many a screener would still have to read), and the confusion-matrix counts. Over
    every cut-off: average precision."""
    predicted = predict_labels(probabilities, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "flagged": float((predicted == 1).mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
        # at the `threshold` cut-off, like precision / recall above
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
