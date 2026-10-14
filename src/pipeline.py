"""
Unified End-to-End Pipeline for TruthLens.
Exposes a single core entrypoint:
    analyze(text: str, source: Optional[str] = None) -> Dict[str, Any]
Fuses:
1. Input preprocessing & Stylistic signal extraction
2. Transformer Inference (DistilBERT with seamless SVM baseline fallback)
3. Source Credibility Analysis & Network Propagation
4. Fact-Check Embedding Verification & Contradiction Scoring
5. Multimodal Risk Scoring (0 to 100)
6. Explainable AI (LIME word contributions + red/green highlighted HTML + plain-English rationale)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from src.preprocess import clean_text_bert, extract_style_signals, validate_input, clean_text_baseline
from src.bert_model import get_bert_classifier
from src.baselines import get_baseline_manager
from src.credibility import evaluate_source
from src.factcheck import verify_factcheck
from src.risk import calculate_risk_score
from src.explain import get_text_explainer, render_highlighted_html, generate_plain_english_reason

logger = logging.getLogger("TruthLens-Pipeline")


def analyze(text: str, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Main analysis pipeline function for TruthLens.
    Accepts raw article text and optional source (URL or name).
    Returns complete structured results for UI rendering or report generation.
    """
    # 1. Validation
    is_valid, validation_msg = validate_input(text)
    if not is_valid:
        return {
            "status": "error",
            "error_message": validation_msg,
            "prediction": "N/A",
            "confidence_pct": 0.0,
            "risk_score": 0.0,
            "risk_label": "N/A"
        }

    cleaned_text = clean_text_bert(text)

    # 2. Extract Stylistic Signals
    style_signals = extract_style_signals(cleaned_text)

    # 3. Source Credibility Evaluation
    credibility_info = evaluate_source(source)

    # 4. Fact-Check Database Verification
    factcheck_info = verify_factcheck(cleaned_text)

    # 5. Model Inference (BERT or Calibrated SVM Baseline Fallback)
    bert_clf = get_bert_classifier()
    baseline_mgr = get_baseline_manager()

    is_baseline_fallback = False
    model_name = "DistilBERT (Fine-Tuned)"

    if bert_clf.is_trained:
        try:
            pred_label, fake_prob = bert_clf.predict(cleaned_text)
            predict_fn = bert_clf.predict_proba_batch
        except Exception as e:
            logger.warning("BERT prediction error: %s. Falling back to SVM.", e)
            is_baseline_fallback = True
            model_name = "SVM (Linear Baseline Fallback)"
            pred_label, fake_prob = baseline_mgr.predict_svm(cleaned_text)
            predict_fn = lambda texts: baseline_mgr.svm_model.predict_proba(
                baseline_mgr.vectorizer.transform([clean_text_baseline(t) for t in texts])
            )
    else:
        logger.info("BERT model not yet trained. Using baseline SVM.")
        is_baseline_fallback = True
        model_name = "SVM (Linear Baseline Fallback)"
        if not baseline_mgr.is_fitted():
            if config.TRAIN_DATA_PATH.exists():
                import pandas as pd
                train_df = pd.read_csv(config.TRAIN_DATA_PATH)
                baseline_mgr.train_all(train_df, quick=True)
            else:
                from data.prepare_data import prepare_data
                prepare_data()
                import pandas as pd
                train_df = pd.read_csv(config.TRAIN_DATA_PATH)
                baseline_mgr.train_all(train_df, quick=True)

        pred_label, fake_prob = baseline_mgr.predict_svm(cleaned_text)
        predict_fn = lambda texts: baseline_mgr.svm_model.predict_proba(
            baseline_mgr.vectorizer.transform([clean_text_baseline(t) for t in texts])
        )

    # Determine human-readable verdict and confidence %
    if pred_label == 1:
        verdict = "FAKE"
        confidence_pct = round(fake_prob * 100, 1)
    else:
        verdict = "REAL"
        confidence_pct = round((1.0 - fake_prob) * 100, 1)

    # 6. Multimodal Risk Scoring
    risk_info = calculate_risk_score(
        bert_fake_prob=fake_prob,
        source_credibility=credibility_info["credibility_score"],
        factcheck_contradiction=factcheck_info["contradiction_score"],
        style_score=style_signals["style_risk_score"]
    )

    # 7. Explainable AI (LIME contributions & highlighted text)
    explainer = get_text_explainer()
    word_weights = explainer.explain(cleaned_text, predict_fn=predict_fn)
    highlighted_html = render_highlighted_html(cleaned_text, word_weights)

    plain_english_reason = generate_plain_english_reason(
        prediction=verdict,
        confidence_pct=confidence_pct,
        word_weights=word_weights,
        style_signals=style_signals,
        credibility_info=credibility_info,
        factcheck_info=factcheck_info
    )

    return {
        "status": "success",
        "input_text": text,
        "source_input": source,
        "validation_msg": validation_msg,
        "prediction": verdict,
        "confidence_pct": confidence_pct,
        "fake_probability": round(fake_prob, 3),
        "model_used": model_name,
        "is_baseline_fallback": is_baseline_fallback,
        "risk_score": risk_info["risk_score"],
        "risk_label": risk_info["risk_label"],
        "risk_color": risk_info["risk_color"],
        "risk_description": risk_info["risk_description"],
        "risk_breakdown": risk_info["breakdown"],
        "risk_max_possible": risk_info["max_possible"],
        "word_weights": word_weights,
        "highlighted_html": highlighted_html,
        "plain_english_reason": plain_english_reason,
        "credibility": credibility_info,
        "factcheck": factcheck_info,
        "style_signals": style_signals
    }


if __name__ == "__main__":
    sample_text = (
        "SHOCKING: Doctors BANNED this miracle herbal tea because it cures all stage 4 cancer in 48 HOURS! "
        "Big Pharma executives are panicking and desperately trying to scrub this secret video off the internet."
    )
    sample_source = "naturalnews.com"
    res = analyze(sample_text, sample_source)
    print("\n=== TruthLens Analysis Result ===")
    print(f"Prediction: {res['prediction']} ({res['confidence_pct']}%)")
    print(f"Risk Score: {res['risk_score']} ({res['risk_label']})")
    print(f"Source: {res['credibility']['domain']} (Credibility: {res['credibility']['credibility_score']})")
    print(f"Reason: {res['plain_english_reason']}\n")
