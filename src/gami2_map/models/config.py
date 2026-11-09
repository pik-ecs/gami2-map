"""Per-model settings for setfit_model.py and classical_model.py."""

# Search-space keys sampled on a log scale
SETFIT_LOG_SCALE = frozenset({"body_learning_rate"})
CLASSICAL_LOG_SCALE = frozenset({"C"})


# SetFit fine-tunes every embedder the same way, so the settings are shared and only the embedding model changes with --model
SETFIT = {
    "checkpoints": {
        # 384: the length mpnet was trained on (its own max_seq_length)
        "mpnet": {
            "checkpoint": "sentence-transformers/all-mpnet-base-v2",
            "max_length": 384,
        },
        "allenai": {"checkpoint": "sentence-transformers/allenai-specter"},
    },
    # For the logistic regression head: weights includes up so the rare class isn't ignored
    # at the 0.5 threshold (as for CLASSICAL)
    "class_weight": "balanced",
    "defaults": {
        "body_learning_rate": 2e-5,
        "num_iterations": 20,  # contrastive pairs generated per training example
        "batch_size": 16,
        "epochs": 1,
    },
    "search_space": {
        "body_learning_rate": (1e-6, 1e-4),
        "num_iterations": [10, 20, 40],
        "batch_size": [8, 16, 32],
        "epochs": (1, 3),
    },
}


# A logistic regression or linear SVM on bag-of-words (tfidf, count) or frozen sentence
# embeddings. --model picks both as <classifier>-<features>, e.g. logreg-tfidf, svm-mpnet.
# C is tuned for every model; binary, ngram_max and min_df too for tfidf / count
CLASSICAL = {
    "vectorizers": ["tfidf", "count"],
    "embedders": {
        "mpnet": {
            "checkpoint": "sentence-transformers/all-mpnet-base-v2",
            "max_length": 384,
        },
        # Averages word vectors over the whole text: no max_length
        "glove": {
            "checkpoint": "sentence-transformers/average_word_embeddings_glove.6B.300d"
        },
    },
    "classifiers": {
        "logreg": {"defaults": {"C": 1.0}, "search_space": {"C": (1e-3, 1e2)}},
        "svm": {"defaults": {"C": 1.0}, "search_space": {"C": (1e-3, 1e1)}},
    },
    "vectorizer": {
        "defaults": {
            "binary": False,
            "ngram_max": 2,
            "min_df": 5,
        },
        "search_space": {
            "binary": [True, False],
            "ngram_max": (1, 3),
            "min_df": (1, 10),
        },
    },
    "class_weight": "balanced",
}
