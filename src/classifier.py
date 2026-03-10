"""
classifier.py - Load trained model and predict department for tickets
"""

import os
import pickle
import logging
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text, validate_ticket

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/classifier.pkl")

# Module-level model cache
_model_cache: Optional[object] = None


def load_model(model_path: str = MODEL_PATH):
    """Load the trained model pipeline from disk (with caching)."""
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Please run 'python src/train_model.py' first."
        )

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        _model_cache = pickle.load(f)

    logger.info("Model loaded successfully")
    return _model_cache


def predict_department(ticket_text: str, model_path: str = MODEL_PATH) -> dict:
    """
    Predict the department for a ticket.

    Args:
        ticket_text: Raw ticket text
        model_path: Path to saved model

    Returns:
        dict with 'department' and 'confidence'
    """
    # Validate input
    is_valid, error_msg = validate_ticket(ticket_text)
    if not is_valid:
        raise ValueError(error_msg)

    # Preprocess
    cleaned = clean_text(ticket_text)
    logger.debug(f"Cleaned ticket: '{cleaned}'")

    # Load model and predict
    model = load_model(model_path)
    department = model.predict([cleaned])[0]

    # Get confidence probabilities
    proba = model.predict_proba([cleaned])[0]
    confidence = float(max(proba))
    classes = model.classes_.tolist()

    # Build confidence map for all departments
    confidence_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    logger.info(f"Predicted department: '{department}' (confidence: {confidence:.2%})")

    return {
        "department": department,
        "confidence": round(confidence, 4),
        "all_scores": confidence_map,
    }


def get_supported_departments(model_path: str = MODEL_PATH) -> list:
    """Return the list of departments the model can predict."""
    model = load_model(model_path)
    return model.classes_.tolist()


def reload_model(model_path: str = MODEL_PATH):
    """Force reload model from disk (clears cache)."""
    global _model_cache
    _model_cache = None
    return load_model(model_path)
