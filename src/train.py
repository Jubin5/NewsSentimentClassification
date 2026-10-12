"""Train and evaluate a news sentiment classification model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "news_sentiment.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load, validate, and clean the sentiment dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = data[["text", "sentiment"]].copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()

    label_mapping = {
        "0": "negative",
        "0.0": "negative",
        "1": "positive",
        "1.0": "positive",
    }

    raw_labels = data["sentiment"].astype(str).str.strip()
    data["sentiment"] = raw_labels.map(label_mapping)

    data = data.dropna(subset=["sentiment"])
    data = data[data["text"] != ""]

    # Exclude identical text assigned conflicting sentiment labels.
    conflicting_texts = (
        data.groupby("text")["sentiment"].nunique()
    )
    conflicting_texts = conflicting_texts[
        conflicting_texts > 1
    ].index

    data = data[~data["text"].isin(conflicting_texts)]
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if data["sentiment"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both negative and positive labels."
        )

    LOGGER.info("Usable records: %d", len(data))
    LOGGER.info(
        "Class distribution:\n%s",
        data["sentiment"].value_counts().to_string(),
    )

    return data


def build_pipeline(
    class_weight: str | dict[str, int] = "balanced",
) -> Pipeline:
    """Build a TF-IDF and logistic regression classifier."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=20_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    """Train, evaluate, and save the sentiment classifier."""
    data = load_dataset(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["sentiment"],
    )

    candidates = [
        "balanced",
        {"negative": 3, "positive": 1},
        {"negative": 6, "positive": 1},
        {"negative": 10, "positive": 1},
    ]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = []

    for weights in candidates:
        candidate = build_pipeline(weights)

        scores = cross_val_score(
            candidate,
            x_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        mean_score = float(scores.mean())

        cv_results.append(
            {
                "class_weight": str(weights),
                "mean_macro_f1": mean_score,
                "fold_scores": scores.tolist(),
            }
        )

        LOGGER.info(
            "Class weight %s | Mean CV macro F1: %.4f",
            weights,
            mean_score,
        )

    best_index = max(
        range(len(cv_results)),
        key=lambda index: cv_results[index]["mean_macro_f1"],
    )

    best_weights = candidates[best_index]
    model = build_pipeline(best_weights)

    LOGGER.info("Selected class weights: %s", best_weights)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    labels = ["negative", "positive"]

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "label_mapping": {
            "0": "negative",
            "1": "positive",
        },
        "random_state": RANDOM_STATE,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "selected_class_weight": str(best_weights),
        "cross_validation": cv_results,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test, predictions, average="weighted", zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    (REPORTS_DIR / "classification_report.txt").write_text(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("News Sentiment Confusion Matrix")
    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    LOGGER.info("Final test results")
    LOGGER.info("Accuracy: %.2f%%", metrics["accuracy"] * 100)
    LOGGER.info("Macro F1: %.4f", metrics["f1_macro"])
    LOGGER.info("Weighted F1: %.4f", metrics["f1_weighted"])
    LOGGER.info(
        "Classification report:\n%s",
        classification_report(
            y_test, predictions, labels=labels, zero_division=0
        ),
    )
    LOGGER.info("Model saved to %s", MODEL_PATH)
    LOGGER.info("Evaluation reports saved to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()