"""
utils.py - Utility functions and shared helpers
"""

import os
import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the entire application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} completed in {elapsed_ms:.2f}ms")
        return result
    return wrapper


def measure_latency(func: Callable) -> tuple:
    """Run a function and return (result, latency_ms)."""
    start = time.perf_counter()
    result = func()
    latency_ms = (time.perf_counter() - start) * 1000
    return result, round(latency_ms, 2)


def ensure_dirs(*paths: str) -> None:
    """Ensure all required directories exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Ensured directory: {path}")


def get_env(key: str, default: str = None, required: bool = False) -> str:
    """Get environment variable with optional validation."""
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(f"Required environment variable '{key}' is not set")
    return value


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text for safe logging."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, total {len(text)} chars]"


def sanitize_ticket(ticket: str) -> str:
    """Basic sanitization of ticket input."""
    if not ticket:
        return ""
    # Strip leading/trailing whitespace, limit length
    return ticket.strip()[:5000]


# Request counter for monitoring (in-memory, resets on restart)
class RequestCounter:
    """Simple in-memory request counter for monitoring."""

    def __init__(self):
        self._counts: dict = {}
        self._total = 0

    def increment(self, endpoint: str) -> None:
        self._counts[endpoint] = self._counts.get(endpoint, 0) + 1
        self._total += 1

    def get(self, endpoint: str = None) -> int:
        if endpoint:
            return self._counts.get(endpoint, 0)
        return self._total

    def summary(self) -> dict:
        return {"total": self._total, "by_endpoint": dict(self._counts)}


# Global request counter instance
request_counter = RequestCounter()
