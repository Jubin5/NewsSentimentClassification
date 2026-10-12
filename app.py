"""Streamlit application for news sentiment classification."""

import json
import logging
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="News Sentiment Classification",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained sentiment classification pipeline."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load saved model evaluation metrics."""
    if not METRICS_PATH.is_file():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    """Render the application interface."""
    st.title("📰 News Sentiment Classification")
    st.write(
        "Enter a news headline or article to classify its sentiment "
        "as negative or positive."
    )

    headline = st.text_area(
        "News headline or article",
        placeholder="Example: The company reported record profits this quarter.",
        height=150,
    )

    if st.button("Analyze sentiment", type="primary"):
        if not headline.strip():
            st.warning("Please enter some news text.")
        else:
            try:
                model = load_model()
                prediction = str(
                    model.predict([headline.strip()])[0]
                ).lower()

                if prediction == "positive":
                    st.success("Predicted sentiment: POSITIVE")
                elif prediction == "negative":
                    st.error("Predicted sentiment: NEGATIVE")
                else:
                    st.warning(f"Predicted sentiment: {prediction}")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(
                        [headline.strip()]
                    )[0]
                    class_names = model.classes_

                    st.subheader("Prediction probabilities")
                    st.bar_chart(
                        {
                            str(label).title(): float(probability)
                            for label, probability in zip(
                                class_names, probabilities
                            )
                        }
                    )

                    st.caption(
                        "Probabilities are model estimates, not guarantees "
                        "that a prediction is correct."
                    )

            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.exception("Prediction failed.")
                st.error(f"Could not generate prediction: {exc}")

    st.divider()
    st.header("Model evaluation")

    try:
        metrics = load_metrics()
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Could not load metrics.")
        st.error(f"Could not load evaluation metrics: {exc}")
        metrics = None

    if metrics is None:
        st.info("Train the model to generate evaluation metrics.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Macro F1-score", f"{metrics['f1_macro']:.2%}")
    col3.metric("Weighted F1-score", f"{metrics['f1_weighted']:.2%}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Macro precision", f"{metrics['precision_macro']:.2%}")
    col5.metric("Macro recall", f"{metrics['recall_macro']:.2%}")
    col6.metric("Test samples", str(metrics["test_samples"]))

    st.subheader("Classification report")

    report = metrics["classification_report"]
    rows = []

    for label in ("negative", "positive"):
        result = report[label]
        rows.append(
            {
                "Sentiment": label.title(),
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-score": result["f1-score"],
                "Support": int(result["support"]),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix")
    matrix = metrics["confusion_matrix"]

    st.dataframe(
        {
            "Actual / Predicted": ["Negative", "Positive"],
            "Negative": matrix[0],
            "Positive": matrix[1],
        },
        use_container_width=True,
        hide_index=True,
    )

    image_path = PROJECT_ROOT / "reports" / "confusion_matrix.png"
    if image_path.is_file():
        st.image(
            str(image_path),
            caption="Confusion matrix heatmap",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()