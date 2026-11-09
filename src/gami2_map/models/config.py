"""Per-model settings for transformer_model.py, setfit_model.py and classical_model.py."""

# Search-space keys sampled on a log scale
TRANSFORMER_LOG_SCALE = frozenset({"learning_rate"})
SETFIT_LOG_SCALE = frozenset({"body_learning_rate"})
CLASSICAL_LOG_SCALE = frozenset({"C"})


# A pretrained encoder fine-tuned end to end with a classification head. Every checkpoint is
# tuned the same way, so the settings are shared and only the model changes with --model
TRANSFORMER = {
    "checkpoints": {
        # 512: BERT's own maximum
        "bert": {
            "checkpoint": "bert-base-uncased",
            "max_length": 512,
        },
    },
    # Weights the cross-entropy loss so the rare include class counts as much as exclude
    # (as for SETFIT and CLASSICAL)
    "class_weight": "balanced",
    # Mixed precision (fp16): ~2-3x faster on a CUDA GPU (e.g. Colab's T4); ignored without one
    "use_amp": True,
    # Early stopping: each fit scores validation after every epoch, stops once the tuning
    # metric hasn't improved for this many epochs and keeps the best epoch's weights; the final
    # model then trains for the best trial's best epoch. None: always train for "epochs"
    "early_stopping_patience": 2,
    # Pruning: tuning stops a trial after an epoch if its validation score is below the median
    # of earlier trials' at that epoch (holdout only; see tuning.py)
    "prune": True,
    "defaults": {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "batch_size": 16,
        # With early stopping, the most a fit trains for
        "epochs": 10,
    },
    # Epochs are left to early stopping rather than searched
    "search_space": {
        "learning_rate": (1e-5, 1e-4),
    },
}


# SetFit fine-tunes every embedder the same way, so the settings are shared and only the embedding model changes with --model
SETFIT = {
    "checkpoints": {
        # 384: the length mpnet was trained on (its own max_seq_length)
        "mpnet": {
            "checkpoint": "sentence-transformers/all-mpnet-base-v2",
            "max_length": 384,
        },
        # SPECTER: SciBERT trained on scientific papers (title + abstract); 512 is its own
        # max_seq_length
        "allenai-specter": {
            "checkpoint": "sentence-transformers/allenai-specter",
            "max_length": 512,
        },
        # BGE: a strong general-purpose embedder, same size as mpnet; 512 is its own
        # max_seq_length. Its query instruction is for retrieval only, so none is added
        "bge-base": {
            "checkpoint": "BAAI/bge-base-en-v1.5",
            "max_length": 512,
        },
    },
    # For the logistic regression head: weights includes up so the rare class isn't ignored
    # at the 0.5 threshold (as for CLASSICAL)
    "class_weight": "balanced",
    # Mixed precision (fp16) for the embedder's fine-tuning: ~2-3x faster on a CUDA GPU
    # (e.g. Colab's T4); ignored without one
    "use_amp": True,
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
        # SPECTER: SciBERT trained on scientific papers (title + abstract)
        "allenai-specter": {
            "checkpoint": "sentence-transformers/allenai-specter",
            "max_length": 512,
        },
        # BGE: a strong general-purpose embedder (no query instruction: that is for retrieval)
        "bge-base": {
            "checkpoint": "BAAI/bge-base-en-v1.5",
            "max_length": 512,
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
