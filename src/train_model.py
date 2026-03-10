"""
train_model.py - Train and evaluate the ticket classification model
"""

import os
import pickle
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths from environment or defaults
DATA_PATH = os.getenv("DATA_PATH", "data/tickets.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/classifier.pkl")


def load_data(path: str) -> pd.DataFrame:
    """Load and validate the training dataset."""
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    required_cols = {"ticket", "department"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    df = df.dropna(subset=["ticket", "department"])
    df["ticket"] = df["ticket"].astype(str).str.strip()
    df = df[df["ticket"].str.len() > 0]

    logger.info(f"Loaded {len(df)} records across {df['department'].nunique()} departments")
    logger.info(f"Department distribution:\n{df['department'].value_counts().to_string()}")
    return df


def build_pipeline() -> Pipeline:
    """Build the sklearn ML pipeline: TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams
            max_features=5000,
            min_df=1,
            sublinear_tf=True,        # log scaling for TF
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def evaluate_model(pipeline: Pipeline, X_test, y_test, departments):
    """Print full model evaluation metrics."""
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\n{'='*60}")
    logger.info(f"  MODEL EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"\n  Classification Report:\n")

    report = classification_report(y_test, y_pred, target_names=sorted(departments))
    for line in report.split("\n"):
        logger.info(f"  {line}")

    cm = confusion_matrix(y_test, y_pred, labels=sorted(departments))
    logger.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    logger.info(f"  Labels: {sorted(departments)}")
    for i, row in enumerate(cm):
        logger.info(f"  {sorted(departments)[i]:20s}: {row.tolist()}")

    logger.info(f"{'='*60}\n")
    return accuracy


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    """Full training pipeline."""
    # 1. Load data
    df = load_data(data_path)

    # 2. Preprocess tickets
    logger.info("Preprocessing ticket texts...")
    df["ticket_clean"] = preprocess_batch(df["ticket"].tolist())

    # Remove any empty preprocessed texts
    df = df[df["ticket_clean"].str.len() > 0]
    logger.info(f"Records after preprocessing: {len(df)}")

    X = df["ticket_clean"].tolist()
    y = df["department"].tolist()
    departments = df["department"].unique()

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 4. Build and train pipeline
    logger.info("Building and training pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 6. Evaluate
    evaluate_model(pipeline, X_test, y_test, departments)

    # 7. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train()
