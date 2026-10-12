# 📰 News Sentiment Classification

A machine learning application that classifies news text as **Positive** or **Negative** using Natural Language Processing (NLP), TF-IDF feature extraction, and Logistic Regression. The project includes a Streamlit web interface and Docker support for reproducible deployment.

<!-- Add your Docker Hub image badge below after publishing the image -->
<!-- Example: ![Docker Pulls](https://img.shields.io/docker/pulls/YOUR_DOCKERHUB_USERNAME/news-sentiment-classifier) -->

## 📌 Project Overview

News sentiment classification helps identify the overall sentiment expressed in a news headline or text. This project processes user-provided text and predicts whether its sentiment is positive or negative.

The application provides:
- Real-time sentiment predictions through a Streamlit interface.
- TF-IDF-based text feature extraction.
- Logistic Regression classification.
- Prediction confidence scores.
- Model evaluation metrics and classification reports.
- Docker-based packaging for consistent execution.

## ✨ Key Features

- **Text preprocessing and feature extraction:** Converts text into numerical features using TF-IDF.
- **Sentiment prediction:** Classifies input text as Positive or Negative.
- **Class imbalance handling:** Evaluates class-weight configurations using stratified cross-validation.
- **Model evaluation:** Reports accuracy, precision, recall, macro F1-score, and a confusion matrix.
- **Interactive web application:** Built with Streamlit.
- **Containerization:** Packages the application and trained model into a Docker image.

## 🧠 Model Architecture

The application follows this pipeline:

```text
User Input: News Text
        |
        v
Text Vectorization
(TF-IDF, Unigrams + Bigrams)
        |
        v
Logistic Regression
(Classification)
        |
        v
Predicted Sentiment
Positive / Negative
        |
        v
Prediction Confidence
```

### Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Text feature extraction | TF-IDF |
| Machine learning model | Logistic Regression |
| ML framework | Scikit-learn |
| Data processing | Pandas, NumPy |
| Web interface | Streamlit |
| Model serialization | Joblib |
| Containerization | Docker |
| Testing | Pytest |

## 📊 Dataset

The project uses a CSV dataset containing news text and sentiment labels.

| Property | Details |
|---|---|
| Dataset file | `data/news_sentiment.csv` |
| Original records | 848 |
| Duplicate text records removed | 1 |
| Usable records after cleaning | 847 |
| Input feature | `text` |
| Target column | `sentiment` |
| Positive samples | 748 |
| Negative samples | 100 |

### Label Mapping

| Original label | Sentiment |
|---|---|
| `1` | Positive |
| `0` | Negative |

The dataset is imbalanced, with substantially more positive than negative samples. To address this, different class-weight configurations were compared using stratified cross-validation and macro F1-score.

**Dataset note:** The raw dataset is not included in this repository by default.

## 📈 Model Performance

The following metrics were obtained from the held-out test set using the selected TF-IDF + Logistic Regression pipeline.

| Evaluation Metric | Score |
|---|---:|
| **Accuracy** | **91.76%** |
| Macro Precision | 82.51% |
| Macro Recall | 73.67% |
| **Macro F1-Score** | **77.12%** |
| Weighted F1-Score | 91.12% |

### Key Evaluation Details

- Train-test split: 80:20.
- Random state: `42`.
- Training samples: 677.
- Testing samples: 170.
- Cross-validation: 3-fold stratified cross-validation for class-weight selection.
- Selected class weights: Negative = 10, Positive = 1.

### Confusion Matrix

The held-out test set produced the following results:

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| **Negative** | 10 | 10 |
| **Positive** | 4 | 146 |

The model correctly classified 156 of 170 test samples. However, the negative class has a lower recall than the positive class, indicating that further improvement is needed for detecting negative news.

The reported results are from the current experiment; performance may vary with changes to the dataset, preprocessing, or model configuration.

## 🖥️ Application Preview

<!-- Add a screenshot of the running Streamlit application here -->
<!-- Example: ![Application Preview](docs/images/app-preview.png) -->

The Streamlit interface allows users to enter news text, view the predicted sentiment, and inspect the model's confidence score and evaluation results.

## 🐳 Docker

Docker packages the application, trained model, and evaluation reports so the application can run without manually configuring the project's Python environment.

### Docker Hub Image

**Docker image:** `YOUR_DOCKERHUB_USERNAME/news-sentiment-classifier`

**Docker Hub repository:** [Add your Docker Hub repository URL here](https://hub.docker.com/)

**Pull command:**

```bash
docker pull YOUR_DOCKERHUB_USERNAME/news-sentiment-classifier:1.0
```

**Run the container:**

```bash
docker run --rm -p 8501:8501 YOUR_DOCKERHUB_USERNAME/news-sentiment-classifier:1.0
```

Open the application in your browser:

[http://localhost:8501](http://localhost:8501)

> Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username after publishing the image. The image must be pushed to Docker Hub before other users can pull it.

### Build the Image Locally

From the project root, run:

```bash
docker build -t news-sentiment-classifier:1.0 .
```

### Run the Local Image

```bash
docker run --rm -p 8501:8501 news-sentiment-classifier:1.0
```

The application should then be available at `http://localhost:8501`.

## ⚙️ Run Locally Without Docker

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/NewsSentimentClassification.git
cd NewsSentimentClassification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the trained model

The application requires the trained model at:

```text
models/sentiment_pipeline.joblib
```

The evaluation reports are expected in the `reports/` directory.

If these files are not included in your checkout, obtain the model artifacts from your project's training workflow or the published release. The raw dataset is required to retrain the model.

### 5. Launch Streamlit

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🏋️ Train the Model

Place your dataset at `data/news_sentiment.csv`, activate your virtual environment, and run:

```bash
python -m src.train --data data/news_sentiment.csv
```

The training script validates the data, creates a stratified train-test split, selects class weights through cross-validation, trains the model, and saves the evaluation results.

Generated artifacts include:

```text
models/
└── sentiment_pipeline.joblib

reports/
├── metrics.json
├── classification_report.txt
└── confusion_matrix.png
```

## 📁 Project Structure

```text
NewsSentimentClassification/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── src/
│   └── train.py
│
├── tests/
│
├── data/
│   └── news_sentiment.csv
│
├── models/
│   └── sentiment_pipeline.joblib
│
└── reports/
    ├── metrics.json
    ├── classification_report.txt
    └── confusion_matrix.png
```

The dataset and other locally generated artifacts may be excluded from Git using `.gitignore`. Ensure the trained model and reports are available when building the Docker image.

## 🔬 Future Improvements

- Improve recall and F1-score for the negative sentiment class.
- Experiment with additional text preprocessing and feature extraction methods.
- Compare Logistic Regression with Linear SVM and other text classification algorithms.
- Evaluate transformer-based models such as DistilBERT.
- Expand the dataset to improve class balance and generalization.
- Add automated tests and continuous integration.
- Publish versioned Docker images.

## 👨‍💻 Author

**Jubin K. Babu**
