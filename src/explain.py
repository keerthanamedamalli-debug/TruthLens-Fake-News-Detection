"""
Explainable AI (XAI) Module for TruthLens.
- Utilizes LIME (Local Interpretable Model-agnostic Explanations) to extract word-level contributions.
- Renders the original article with styled HTML highlights:
    * Red highlights: Words pushing the prediction towards FAKE (Class 1)
    * Green/Teal highlights: Words pushing the prediction towards REAL (Class 0)
    * Color intensity proportional to feature contribution weight.
- Generates comprehensive plain-English reasoning combining linguistic cues, style anomalies,
  source credibility, and fact-checking evidence.
"""

import sys
import re
import html
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

logger = logging.getLogger("TruthLens-Explain")


class TextExplainer:
    """
    Manages LIME text explanations and visual HTML generation.
    """
    def __init__(self):
        self._explainer = None

    def _get_lime_explainer(self):
        if self._explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer
                self._explainer = LimeTextExplainer(
                    class_names=["REAL", "FAKE"],
                    random_state=config.RANDOM_SEED,
                    split_expression=r"\W+",
                    bow=True
                )
            except Exception as e:
                logger.error("Failed to initialize LimeTextExplainer: %s", e)
        return self._explainer

    def explain(
        self,
        text: str,
        predict_fn: Callable[[List[str]], Any],
        num_samples: int = config.LIME_NUM_SAMPLES,
        num_features: int = config.LIME_NUM_FEATURES
    ) -> List[Tuple[str, float]]:
        """
        Runs LIME on the input text with the given predict_proba function.
        Returns a list of (word, weight) tuples.
        Positive weight -> pushes to FAKE.
        Negative weight -> pushes to REAL.
        """
        explainer = self._get_lime_explainer()
        if explainer is None or not text.strip():
            return []

        try:
            # Generate explanation for class 1 (FAKE)
            exp = explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=(1,)
            )
            # List of (feature_string, weight)
            return exp.as_list(label=1)
        except Exception as e:
            logger.warning("LIME explanation error: %s. Using heuristic keyword weights.", e)
            return self._heuristic_fallback_weights(text)

    def _heuristic_fallback_weights(self, text: str) -> List[Tuple[str, float]]:
        """Fallback word weights if LIME experiences an issue."""
        words = re.findall(r"\b[A-Za-z]{3,}\b", text)
        weights = []
        for w in set(words):
            low = w.lower()
            if low in config.CLICKBAIT_PHRASES or low in {"shocking", "miracle", "banned", "secret", "hoax"}:
                weights.append((w, 0.45))
            elif low in {"official", "confirmed", "spokesperson", "statement", "parliament", "research"}:
                weights.append((w, -0.35))
        return weights[:config.LIME_NUM_FEATURES]


def render_highlighted_html(text: str, word_weights: List[Tuple[str, float]]) -> str:
    """
    Renders text with interactive HTML highlighting.
    Red for FAKE-pushing words, Green/Teal for REAL-pushing words.
    """
    if not text:
        return "<em>No text to display</em>"

    if not word_weights:
        return f"<div style='line-height: 1.8; color: #e0e6ed;'>{html.escape(text)}</div>"

    # Map lowercase words to their weight
    weight_map = {w.lower(): wt for w, wt in word_weights}
    max_weight = max(abs(wt) for _, wt in word_weights) if word_weights else 1.0
    if max_weight == 0:
        max_weight = 1.0

    # Tokenize preserving spaces and punctuation
    tokens = re.split(r"(\b[A-Za-z0-9_-]+\b)", text)
    highlighted_parts = []

    for token in tokens:
        token_clean = token.strip().lower()
        if token_clean in weight_map:
            weight = weight_map[token_clean]
            intensity = min(max(abs(weight) / max_weight, 0.25), 0.95)

            if weight > 0:
                # Pushes to FAKE -> Red / Coral
                bg_color = f"rgba(255, 75, 75, {intensity:.2f})"
                border_color = "#ff4b4b"
                label = f"Pushes towards FAKE (+{weight:.2f})"
            else:
                # Pushes to REAL -> Green / Teal
                bg_color = f"rgba(0, 204, 150, {intensity:.2f})"
                border_color = "#00cc96"
                label = f"Pushes towards REAL ({weight:.2f})"

            escaped = html.escape(token)
            styled = (
                f"<span title='{label}' style='"
                f"background-color: {bg_color}; "
                f"border-bottom: 2px solid {border_color}; "
                f"padding: 1px 4px; "
                f"border-radius: 4px; "
                f"font-weight: 600; "
                f"color: #ffffff; "
                f"cursor: help;'>"
                f"{escaped}"
                f"</span>"
            )
            highlighted_parts.append(styled)
        else:
            highlighted_parts.append(html.escape(token))

    return (
        "<div style='"
        "font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif; "
        "font-size: 1.05rem; "
        "line-height: 1.85; "
        "color: #e0e6ed; "
        "background: rgba(11, 19, 43, 0.6); "
        "padding: 16px; "
        "border-radius: 8px; "
        "border: 1px solid #1c2541;'>"
        + "".join(highlighted_parts)
        + "</div>"
    )


def generate_plain_english_reason(
    prediction: str,
    confidence_pct: float,
    word_weights: List[Tuple[str, float]],
    style_signals: Dict[str, Any],
    credibility_info: Dict[str, Any],
    factcheck_info: Dict[str, Any]
) -> str:
    """
    Synthesizes all multimodal evidence into a concise, professional explanation paragraph.
    Combines:
    1. Verdict & Model Confidence
    2. Influential Lexical Words
    3. Stylistic & Clickbait Indicators
    4. Source Credibility Assessment
    5. Fact-Check Corroboration / Contradiction
    """
    parts = []

    # 1. Prediction summary
    if prediction.upper() == "FAKE":
        parts.append(f"Flagged as **FAKE** with **{confidence_pct:.1f}% confidence** by the AI model.")
    else:
        parts.append(f"Classified as **REAL** with **{confidence_pct:.1f}% confidence** by the AI model.")

    # 2. Key influential words from LIME
    fake_words = [w for w, wt in word_weights if wt > 0]
    real_words = [w for w, wt in word_weights if wt < 0]

    if prediction.upper() == "FAKE" and fake_words:
        top_words_str = ", ".join(f"'{w}'" for w in fake_words[:4])
        parts.append(f"Linguistic analysis identified strong sensational and unverified phrasing ({top_words_str}).")
    elif prediction.upper() == "REAL" and real_words:
        top_words_str = ", ".join(f"'{w}'" for w in real_words[:4])
        parts.append(f"Linguistic analysis identified measured and objective journalistic wording ({top_words_str}).")

    # 3. Stylistic anomalies
    caps = style_signals.get("caps_ratio", 0.0)
    clickbaits = style_signals.get("clickbait_phrases_found", [])
    exclamations = style_signals.get("exclamation_count", 0)

    style_notes = []
    if caps > 0.18:
        style_notes.append(f"unusually high ALL-CAPS text ({int(caps * 100)}%)")
    if clickbaits:
        phrases_str = ", ".join(f"'{p}'" for p in clickbaits[:2])
        style_notes.append(f"clickbait headline markers ({phrases_str})")
    if exclamations >= 2:
        style_notes.append(f"frequent exclamation marks ({exclamations})")

    if style_notes:
        parts.append(f"Stylistic signals detected: {'; '.join(style_notes)}.")

    # 4. Source credibility
    cred_score = credibility_info.get("credibility_score", 50)
    cat = credibility_info.get("category", "Unknown")
    domain = credibility_info.get("domain", "")

    if credibility_info.get("is_known", False):
        if cat == "Trusted":
            parts.append(f"The source ('{domain}') is a verified reputable publication (Credibility: {cred_score}/100).")
        elif cat in {"Unreliable", "Satire"}:
            parts.append(f"The source ('{domain}') has a poor credibility rating ({cred_score}/100, category: {cat}).")
        else:
            parts.append(f"The source ('{domain}') has a mixed editorial track record ({cred_score}/100).")
    elif domain and domain != "None Specified":
        parts.append(f"The source domain ('{domain}') is unindexed; a neutral baseline (50/100) was applied.")

    # 5. Fact check match
    top_fc = factcheck_info.get("top_match")
    if top_fc and top_fc.get("similarity_pct", 0) >= config.FACTCHECK_SIMILARITY_THRESHOLD * 100:
        sim = top_fc["similarity_pct"]
        fc_verdict = top_fc["verdict"]
        fc_org = top_fc["fact_checker"]
        if fc_verdict.lower() == "false":
            parts.append(
                f"**Critical Fact-Check Match**: Matches a known debunked claim ('{top_fc['claim'][:80]}...') "
                f"verified as FALSE by {fc_org} ({sim}% semantic similarity)."
            )
        elif fc_verdict.lower() == "true":
            parts.append(
                f"**Corroborated Fact**: Aligns closely with a verified factual event confirmed by {fc_org} ({sim}% match)."
            )

    return " ".join(parts)


# Singleton explainer
_explainer_instance = None

def get_text_explainer() -> TextExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = TextExplainer()
    return _explainer_instance
