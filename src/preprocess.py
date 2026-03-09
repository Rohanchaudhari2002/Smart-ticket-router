"""
preprocess.py - Text preprocessing for ticket classification
"""

import re
import string
import logging
from typing import List

logger = logging.getLogger(__name__)

# Common stopwords (avoiding NLTK dependency for portability)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Tokenize and rejoin
    """
    if not text or not isinstance(text, str):
        logger.warning("Received empty or non-string input for preprocessing")
        return ""

    # Step 1: Lowercase
    text = text.lower().strip()

    # Step 2: Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)  # remove standalone numbers

    # Step 3: Tokenize
    tokens = text.split()

    # Step 4: Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    cleaned = " ".join(tokens)
    logger.debug(f"Preprocessed: '{text}' -> '{cleaned}'")
    return cleaned


def preprocess_batch(texts: List[str]) -> List[str]:
    """Preprocess a list of ticket texts."""
    return [clean_text(t) for t in texts]


def validate_ticket(ticket: str) -> tuple[bool, str]:
    """
    Validate ticket input.
    Returns (is_valid, error_message)
    """
    if not ticket:
        return False, "Ticket text cannot be empty"
    if not isinstance(ticket, str):
        return False, "Ticket must be a string"
    if len(ticket.strip()) < 3:
        return False, "Ticket text is too short (minimum 3 characters)"
    if len(ticket) > 5000:
        return False, "Ticket text is too long (maximum 5000 characters)"
    cleaned = clean_text(ticket)
    if not cleaned:
        return False, "Ticket contains no meaningful content after preprocessing"
    return True, ""
