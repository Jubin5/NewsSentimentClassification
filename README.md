# News Sentiment Classification

A machine-learning application that predicts whether a piece of news text expresses **positive** or **negative** sentiment. It uses TF-IDF text features and Logistic Regression, with a Streamlit interface and a published Docker image.

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-news--sentiment--classifier-blue?logo=docker)](https://hub.docker.com/r/jubin5/news-sentiment-classifier)

**Docker Hub image:** [`jubin5/news-sentiment-classifier:1.0`](https://hub.docker.com/r/jubin5/news-sentiment-classifier/tags)

## Project Overview

News sentiment classification is an NLP task that assigns a sentiment label to a news headline or text. This project provides a web app where users can enter news text and receive a predicted sentiment and confidence score.

## Features

- Predicts **Positive** or **Negative** sentiment.
- Uses TF-IDF unigrams and bigrams for text features.
- Uses Logistic Regression for classification.
- Compares class-weight settings with stratified cross-validation using macro F1.
- Provides a Streamlit interface.
- Packages the app and trained model in Docker.
- Publishes a versioned image on Docker Hub.

## Model and Workflow

```text
News text entered by the user
             |
             v
       TF-IDF vectorizer
       (unigrams + bigrams)
             |
             v
      Logistic Regression
             |
             v
 Positive / Negative prediction
             |
             v
      Streamlit web interface
```

| Area | Technology |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Text representation | Scikit-learn TF-IDF |
| Classifier | Scikit-learn Logistic Regression |
| Web app | Streamlit |
| Model serialization | Joblib |
| Containerization | Docker |
| Tests | Pytest |

## Dataset

The development dataset is `data/news_sentiment.csv`, with `text` and `sentiment` columns.

| Dataset detail | Value |
|---|---:|
| Original records | 848 |
| Duplicate text records removed | 1 |
| Records after cleaning | 847 |
| Positive examples (label `1`) | 748 |
| Negative examples (label `0`) | 100 |

Label mapping: `1` = Positive; `0` = Negative.

The dataset is imbalanced, with positive examples much more common than negative examples. Class-weight configurations were compared using stratified cross-validation.

**Dataset availability:** The raw CSV is not included in this repository. Place it at `data/news_sentiment.csv` if you want to retrain the model.

## Evaluation Results

The following results are from the current held-out test set.

| Metric | Score |
|---|---:|
| **Accuracy** | **91.76%** |
| Macro precision | 82.51% |
| Macro recall | 73.67% |
| **Macro F1-score** | **77.12%** |
| Weighted F1-score | 91.12% |

### Evaluation setup

- Train/test split: 80/20, stratified
- Random state: `42`
- Training samples: `677`
- Test samples: `170`
- Class-weight selection: 3-fold stratified cross-validation on the training split
- Selected class weights: negative `10`, positive `1`

### Confusion matrix

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 10 | 10 |
| Positive | 4 | 146 |

The model correctly classified 156 of 170 test examples. Negative-class recall is lower than positive-class recall, so the model misses some negative examples. Accuracy alone does not fully describe performance on this imbalanced dataset; macro F1 gives equal importance to both classes.

These results are from this particular split and do not guarantee performance on new data.

## Run with Docker

**Docker Hub repository:** https://hub.docker.com/r/jubin5/news-sentiment-classifier  
**Image:** `jubin5/news-sentiment-classifier:1.0`

### 1. Pull the image

```bash
docker pull jubin5/news-sentiment-classifier:1.0
```

### 2. Run the container

```bash
docker run --rm -p 8501:8501 jubin5/news-sentiment-classifier:1.0
```

### 3. Open the app

Visit [http://localhost:8501](http://localhost:8501). Keep the terminal running while using the app; press `Ctrl+C` in the terminal to stop it.

## Run Locally

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username after creating the repository.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/NewsSentimentClassification.git
cd NewsSentimentClassification
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Make the trained model available

The app expects the trained pipeline at:

```text
models/sentiment_pipeline.joblib
```

Generated model and report files are excluded from Git in this project, so a fresh clone will not contain them automatically. To run the app from source, generate or copy the required artifacts from your training run. Alternatively, use the published Docker image above.

### 5. Start Streamlit

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Train the Model

Place the dataset at `data/news_sentiment.csv`, activate the virtual environment, and run:

```powershell
python src/train.py
```

The training script validates the dataset, creates a stratified train/test split, compares class weights using cross-validation, trains the pipeline, and saves model/evaluation artifacts under `models/` and `reports/`.

## Project Structure

```text
NewsSentimentClassification/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── src/
│   └── train.py
├── tests/
├── data/       # local dataset; excluded from Git
├── models/     # generated model; excluded from Git
└── reports/    # generated metrics/plots; excluded from Git
```

## Limitations and Future Improvements

- Improve negative-class recall and macro F1.
- Collect more negative examples and evaluate on additional data.
- Compare Logistic Regression with Linear SVM and transformer-based models.
- Add automated tests and a continuous integration workflow.
- Track model and dataset versions for reproducibility.

## Author

**Jubin K. Babu**  
MSc Data Analytics and Computational Science

---

*This project is intended for learning and experimentation. Predictions may be incorrect, especially for ambiguous text or news outside the training distribution.*
