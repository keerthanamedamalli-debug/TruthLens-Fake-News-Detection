"""
Text Preprocessing and Stylistic Signal Extraction Module for TruthLens.
- Cleans and normalizes text for BERT (casing & punctuation preserved).
- Lowers and strips text for classical baselines.
- Computes stylistic indicators: ALL-CAPS ratio, exclamation marks, clickbait phrases,
  and sentiment/sensationalism extremity.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

# Extreme emotion / sensational words
EXTREME_WORDS = {
    "shocking", "horrifying", "terrifying", "unbelievable", "mind-blowing",
    "miracle", "catastrophic", "banned", "secret", "exposed", "conspiracy",
    "hoax", "apocalyptic", "monstrous", "nightmare", "devastating", "panicking",
    "traitor", "corrupt", "sinister", "furious", "unprecedented", "bombshell"
}


def clean_text_bert(text: str, max_chars: int = 3000) -> str:
    """
    Cleans raw text while preserving casing, punctuation, and sentence structure
    essential for transformer contextual embeddings.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Normalize weird unicode spaces / quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")

    # Normalize excessive spaces and linebreaks
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate character ceiling to avoid memory bloat
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."

    return text


def clean_text_baseline(text: str) -> str:
    """
    Normalizes text for TF-IDF / classical baselines: lowercased, minimal punctuation.
    """
    text = clean_text_bert(text)
    text = text.lower()
    # Strip characters that are not alphanumeric or basic whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_style_signals(text: str) -> Dict[str, Any]:
    """
    Extracts heuristic stylistic features associated with disinformation:
    - ALL-CAPS ratio in alphabetic tokens
    - Exclamation and question mark frequencies
    - Presence of known clickbait / sensational phrasing
    - Extremity of emotional language
    Returns a dictionary of raw counts and a normalized style_risk_score (0.0 to 1.0).
    """
    if not text or not text.strip():
        return {
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "clickbait_phrases_found": [],
            "clickbait_count": 0,
            "sentiment_extremity": 0.0,
            "style_risk_score": 0.0
        }

    # Extract uppercase ratio among alphabetical characters
    alpha_chars = [c for c in text if c.isalpha()]
    caps_chars = [c for c in alpha_chars if c.isupper()]
    caps_ratio = len(caps_chars) / max(len(alpha_chars), 1)

    # Punctuation counts
    exclamation_count = text.count("!")
    question_count = text.count("?")

    # Clickbait phrases detection (case-insensitive)
    lower_text = text.lower()
    clickbait_found = []
    for phrase in config.CLICKBAIT_PHRASES:
        if phrase in lower_text:
            clickbait_found.append(phrase)

    # Sentiment / sensationalism extremity
    words = re.findall(r"\b\w+\b", lower_text)
    extreme_matches = [w for w in words if w in EXTREME_WORDS]
    extreme_ratio = len(extreme_matches) / max(len(words), 1)

    # Heuristic scoring formula (0.0 to 1.0):
    # - Caps ratio > 0.15 is abnormal in standard journalism
    # - Multiple exclamations (>= 2) strongly signal tabloid/sensational writing
    # - Any clickbait phrases found adds direct penalty
    caps_penalty = min(max(caps_ratio - 0.10, 0.0) * 2.5, 0.35)
    excl_penalty = min(exclamation_count * 0.10, 0.25)
    clickbait_penalty = min(len(clickbait_found) * 0.15, 0.30)
    extreme_penalty = min(extreme_ratio * 4.0, 0.20)

    raw_score = caps_penalty + excl_penalty + clickbait_penalty + extreme_penalty
    style_risk_score = min(max(raw_score, 0.0), 1.0)

    return {
        "caps_ratio": round(caps_ratio, 3),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "clickbait_phrases_found": clickbait_found,
        "clickbait_count": len(clickbait_found),
        "sentiment_extremity": round(extreme_ratio, 3),
        "style_risk_score": round(style_risk_score, 3)
    }


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validates user input text for the Streamlit dashboard or API.
    Returns (is_valid, message).
    """
    if not text or not text.strip():
        return False, "Input text is empty. Please provide a headline or article excerpt."

    cleaned = clean_text_bert(text)
    word_count = len(cleaned.split())

    if word_count < 5:
        return False, "Input is too brief (less than 5 words). Please provide a fuller headline or snippet."

    if word_count < 20:
        return True, f"Note: Input is relatively short ({word_count} words). Prediction will rely heavily on lexical signals and credibility."

    return True, f"Input accepted ({word_count} words)."
