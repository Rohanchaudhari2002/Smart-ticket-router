"""
priority.py - Rule-based priority detection for IT support tickets
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Priority keyword rules (ordered high -> medium -> low)
PRIORITY_RULES: Dict[str, List[str]] = {
    "High": [
        "error", "failure", "crash", "crashed", "crashing", "failed",
        "down", "outage", "critical", "breach", "corrupted", "corruption",
        "lost", "deleted", "missing", "broken", "dead", "emergency",
        "urgent", "production", "security", "attack", "virus", "malware",
        "ransomware", "hack", "unauthorized", "data loss", "not working",
        "blue screen", "bsod", "kernel panic", "system down", "server down",
    ],
    "Medium": [
        "unable", "cannot", "can't", "wont", "won't", "doesn't", "doesn",
        "not responding", "not loading", "not syncing", "not connecting",
        "access denied", "permission denied", "blocked", "locked",
        "timeout", "expired", "invalid", "mismatch", "conflict",
        "slow performance", "lagging", "freezing", "frozen", "stuck",
        "warning", "issue", "problem", "trouble", "difficulty",
    ],
    "Low": [
        "slow", "delay", "delayed", "laggy", "sluggish", "minor",
        "occasionally", "sometimes", "intermittent", "flicker", "flickering",
        "cosmetic", "question", "inquiry", "request", "help", "how to",
        "assistance", "information", "update", "upgrade", "install",
        "setup", "configure", "configuration",
    ],
}

# Escalation modifiers that increase priority
ESCALATION_WORDS = ["all users", "everyone", "entire", "whole team", "company-wide", "production"]


def detect_priority(ticket_text: str) -> str:
    """
    Detect priority level from ticket text using rule-based matching.

    Priority levels: High > Medium > Low
    Default: Medium

    Args:
        ticket_text: Raw ticket text

    Returns:
        Priority string: "High", "Medium", or "Low"
    """
    if not ticket_text:
        return "Medium"

    text_lower = ticket_text.lower()

    # Check for escalation modifiers — auto-escalate to High
    for escalation_word in ESCALATION_WORDS:
        if escalation_word in text_lower:
            logger.info(f"Escalated to High due to '{escalation_word}' in ticket")
            return "High"

    # Check priorities from highest to lowest
    for priority, keywords in PRIORITY_RULES.items():
        for keyword in keywords:
            if keyword in text_lower:
                logger.debug(f"Priority '{priority}' matched keyword '{keyword}'")
                return priority

    # Default to Medium if no keywords match
    logger.debug("No priority keywords matched, defaulting to Medium")
    return "Medium"


def get_priority_score(priority: str) -> int:
    """Convert priority string to numeric score for sorting."""
    scores = {"High": 3, "Medium": 2, "Low": 1}
    return scores.get(priority, 2)


def get_priority_summary() -> Dict[str, List[str]]:
    """Return the full priority rules for documentation/debugging."""
    return PRIORITY_RULES.copy()
