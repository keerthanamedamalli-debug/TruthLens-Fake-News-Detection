"""
TruthLens Streamlit Dashboard.
Detecting Misinformation with Explainable and Credibility-Aware AI.
Features:
- Dark Navy & Teal Accent Aesthetics with Custom CSS
- Tabs: Detect, Model Comparison, Credibility Graph, How It Works
- Preloaded one-click examples (2 fake, 2 real)
- Highlighted text explanations (LIME), Plotly Risk Gauge & Component Breakdown
- Downloadable standalone HTML/PDF-ready analysis report
- Model caching with @st.cache_resource
"""

import sys
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from src.pipeline import analyze
from src.preprocess import clean_text_bert
from src.risk import create_risk_gauge_figure, create_risk_breakdown_figure
from src.credibility import get_credibility_manager
from src.evaluate import get_cached_results
from src.bert_model import get_bert_classifier
from data.prepare_data import get_dataset_metadata

# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="TruthLens – AI Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Custom Styling (Dark Navy #0B132B + Teal Accent #00F0FF)
# -------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main Background & Typography */
.stApp {
    background-color: #0b132b;
    color: #e0e6ed;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, #1c2541 0%, #0b132b 100%);
    border-bottom: 2px solid #00f0ff;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 240, 255, 0.08);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #48cae4;
    font-size: 1.05rem;
    margin-top: 6px;
    margin-bottom: 0;
}

/* Glassmorphism Cards */
.tl-card {
    background: #1c2541;
    border: 1px solid #3a506b;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.tl-card-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #00f0ff;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Verdict Badges */
.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg, #ff4b4b 0%, #b30000 100%);
    color: #ffffff;
    font-size: 1.7rem;
    font-weight: 800;
    padding: 8px 28px;
    border-radius: 30px;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
}

.badge-real {
    display: inline-block;
    background: linear-gradient(135deg, #00cc96 0%, #007a5a 100%);
    color: #ffffff;
    font-size: 1.7rem;
    font-weight: 800;
    padding: 8px 28px;
    border-radius: 30px;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(0, 204, 150, 0.4);
}

/* Category Badges */
.badge-cat-trusted { background-color: rgba(0, 240, 255, 0.2); color: #00f0ff; border: 1px solid #00f0ff; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
.badge-cat-mixed { background-color: rgba(255, 183, 3, 0.2); color: #ffb703; border: 1px solid #ffb703; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
.badge-cat-unreliable { background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; border: 1px solid #ff4b4b; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
.badge-cat-satire { background-color: rgba(199, 125, 255, 0.2); color: #c77dff; border: 1px solid #c77dff; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
.badge-cat-unknown { background-color: rgba(141, 153, 174, 0.2); color: #8d99ae; border: 1px solid #8d99ae; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600; }

/* Fact-check Table */
.fc-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}
.fc-table th {
    background-color: #0b132b;
    color: #00f0ff;
    padding: 10px;
    text-align: left;
    font-size: 0.9rem;
    border-bottom: 2px solid #3a506b;
}
.fc-table td {
    padding: 10px;
    border-bottom: 1px solid #1c2541;
    font-size: 0.9rem;
    color: #e0e6ed;
}

/* Custom button tweaks */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #00f0ff 0%, #0077b6 100%);
    color: #0b132b;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 4px 16px rgba(0, 240, 255, 0.4);
    transform: translateY(-1px);
    color: #ffffff;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1c2541;
    border-radius: 8px 8px 0 0;
    color: #8d99ae;
    padding: 10px 20px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #3a506b !important;
    color: #00f0ff !important;
    border-bottom: 2px solid #00f0ff !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Resources (Models & Data)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP & Credibility Engines...")
def load_cached_engines():
    """Caches pipeline resources so repeated inferences respond in milliseconds."""
    from src.baselines import get_baseline_manager
    from src.factcheck import get_factcheck_verifier
    from src.credibility import get_credibility_manager
    from src.bert_model import get_bert_classifier

    # Initialize singletons
    b_mgr = get_baseline_manager()
    fc_ver = get_factcheck_verifier()
    cred_mgr = get_credibility_manager()
    bert_clf = get_bert_classifier()

    return {
        "baseline_mgr": b_mgr,
        "factcheck_verifier": fc_ver,
        "credibility_mgr": cred_mgr,
        "bert_classifier": bert_clf
    }

engines = load_cached_engines()


# -------------------------------------------------------------
# Demo Examples
# -------------------------------------------------------------
SAMPLE_INPUTS = {
    "fake_1": {
        "title": "🔴 Fake: Miracle Cancer Tea (Health)",
        "text": (
            "SHOCKING: Doctors BANNED this miracle herbal tea because it cures all stage 4 cancer in 48 HOURS! "
            "Big Pharma executives are panicking and desperately trying to scrub this secret video off the internet. "
            "Just boiling two kitchen spices dissolves tumors completely with zero side effects! Share this before it gets deleted!"
        ),
        "source": "naturalnews.com"
    },
    "fake_2": {
        "title": "🔴 Fake: UNESCO Best Anthem Hoax (Viral India)",
        "text": (
            "UNESCO OFFICIAL DECLARATION: Indian National Anthem has just been awarded the Best Anthem in the Universe! "
            "All 193 country presidents gave a standing ovation in Paris assembly yesterday! "
            "Every proud citizen must forward this patriotic achievement to at least 10 groups right now!"
        ),
        "source": "postcard.news"
    },
    "real_1": {
        "title": "🟢 Real: WHO Influenza Advisory (Health)",
        "text": (
            "World Health Organization releases updated clinical management recommendations for viral seasonal influenza. "
            "Surveillance data collected across 48 reference laboratories indicates predominant circulating sub-strains "
            "remain susceptible to standard antiviral regimens. Public health authorities urge annual vaccination for geriatric populations."
        ),
        "source": "who.int"
    },
    "real_2": {
        "title": "🟢 Real: Chandrayaan-3 Landing (Science/Space)",
        "text": (
            "ISRO successfully landed the Chandrayaan-3 lander Vikram near the south pole of the Moon. "
            "On August 23, 2023, India became the first space agency in history to achieve a controlled soft touchdown in the lunar south polar region. "
            "Telemetry instruments deployed the Pragyan rover to conduct surface mineralogy spectrometry."
        ),
        "source": "thehindu.com"
    }
}


# -------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ **TruthLens Engine**")
    st.markdown("**Version**: 1.0.0 (College Demo Edition)")

    # Display active data source
    meta = get_dataset_metadata()
    data_source = meta.get("data_source", "Synthetic demo dataset")
    total_samples = meta.get("total_samples", 1400)

    st.markdown(f"""
    <div style='background: #1c2541; padding: 12px; border-radius: 8px; border: 1px solid #3a506b; font-size: 0.85rem; margin-bottom: 16px;'>
        <div style='color: #8d99ae;'>Training Dataset:</div>
        <div style='color: #00f0ff; font-weight: 700; margin-top: 2px;'>{data_source}</div>
        <div style='color: #e0e6ed; font-size: 0.8rem; margin-top: 4px;'>Samples: {total_samples} (Balanced 50/50)</div>
    </div>
    """, unsafe_allow_html=True)

    # Active Model Status
    bert_clf = engines["bert_classifier"]
    if bert_clf.is_trained:
        st.success("🤖 **Active Model**: DistilBERT Transformer (Fine-Tuned)")
    else:
        st.warning("⚡ **Active Model**: TF-IDF + SVM Baseline Fallback")
        st.caption("Run `python src/bert_model.py --quick` to fine-tune DistilBERT.")

    st.markdown("---")
    st.markdown("### 📐 **System Pipeline**")
    st.markdown("""
    1. **Text Preprocessing** & Style Signals
    2. **Transformer Classification** (DistilBERT)
    3. **Source Credibility** (Graph PageRank)
    4. **Fact-Check Verification** (Embeddings)
    5. **Explainable AI** (LIME Word Highlighting)
    6. **Multimodal Risk Scoring** (0-100 Gauge)
    """)

    st.markdown("---")
    st.markdown("### ⚖️ **Risk Weighting**")
    st.markdown(f"""
    - **BERT Fake Probability**: `{int(config.RISK_WEIGHT_BERT*100)}%`
    - **Source Unreliability**: `{int(config.RISK_WEIGHT_CREDIBILITY*100)}%`
    - **Fact-Check Contradiction**: `{int(config.RISK_WEIGHT_FACTCHECK*100)}%`
    - **Stylistic Anomalies**: `{int(config.RISK_WEIGHT_STYLE*100)}%`
    """)


# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>TruthLens – Detecting Misinformation with Explainable AI</h1>
    <p>A credibility-aware, explainable framework for deep linguistic analysis, graph-propagated source trust, and semantic fact-check corroboration.</p>
</div>
""", unsafe_allow_html=True)

# Warning banner if running on baseline fallback
if not engines["bert_classifier"].is_trained:
    st.info("ℹ️ **Demo Notice**: Currently operating in **Fast Baseline Mode (TF-IDF + Calibrated SVM)**. All explanation, fact-checking, credibility, and risk modules are fully active.")


# -------------------------------------------------------------
# Tabs
# -------------------------------------------------------------
tab_detect, tab_models, tab_graph, tab_about = st.tabs([
    "🔍 Detect Misinformation",
    "📊 Model Comparison",
    "🌐 Credibility Graph",
    "📖 How It Works"
])


# =============================================================
# TAB 1: DETECT
# =============================================================
with tab_detect:
    st.markdown("#### Paste Headline or Article Excerpt for Verification")

    # Example Buttons
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    # Initialize session state for input fields
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "input_source" not in st.session_state:
        st.session_state["input_source"] = ""

    with col_ex1:
        if st.button("Sample: Cancer Tea (Fake)", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_INPUTS["fake_1"]["text"]
            st.session_state["input_source"] = SAMPLE_INPUTS["fake_1"]["source"]
    with col_ex2:
        if st.button("Sample: UNESCO Anthem (Fake)", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_INPUTS["fake_2"]["text"]
            st.session_state["input_source"] = SAMPLE_INPUTS["fake_2"]["source"]
    with col_ex3:
        if st.button("Sample: WHO Advisory (Real)", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_INPUTS["real_1"]["text"]
            st.session_state["input_source"] = SAMPLE_INPUTS["real_1"]["source"]
    with col_ex4:
        if st.button("Sample: Chandrayaan-3 (Real)", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_INPUTS["real_2"]["text"]
            st.session_state["input_source"] = SAMPLE_INPUTS["real_2"]["source"]

    # Input Form
    with st.form("detection_form"):
        user_text = st.text_area(
            "News Article / Headline / Post Text:",
            value=st.session_state["input_text"],
            height=140,
            placeholder="Paste news headline, article excerpt, WhatsApp forward, or social media post here..."
        )
        col_src, col_btn = st.columns([3, 1])
        with col_src:
            user_source = st.text_input(
                "Source Publication / Domain / URL (Optional):",
                value=st.session_state["input_source"],
                placeholder="e.g., https://www.reuters.com, bbc.com, or 'The Hindu'"
            )
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("🚀 Analyze with TruthLens", use_container_width=True)

    # Trigger analysis on submit or if example was loaded and submit pressed
    if submit_btn:
        if not user_text.strip():
            st.error("⚠️ Please paste an article excerpt or headline before clicking Analyze.")
        else:
            with st.spinner("Analyzing linguistic semantics, style indicators, source reputation, and verified claims..."):
                result = analyze(user_text, user_source)

            if result["status"] == "error":
                st.error(f"⚠️ {result['error_message']}")
            else:
                # ---------------- RESULTS DISPLAY ----------------
                st.markdown("<hr style='border-color: #3a506b; margin: 24px 0;'>", unsafe_allow_html=True)

                # TOP SUMMARY ROW: Verdict & Risk Gauge
                col_verdict, col_gauge = st.columns([1.1, 1.2])

                with col_verdict:
                    st.markdown("""
                    <div class="tl-card" style="text-align: center; height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    """, unsafe_allow_html=True)

                    if result["prediction"] == "FAKE":
                        st.markdown("<div class='badge-fake'>🚨 FAKE NEWS</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='badge-real'>✅ REAL NEWS</div>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin-top: 18px; font-size: 1.3rem; color: #ffffff;">
                        Model Confidence: <span style="color: #00f0ff; font-weight: 700;">{result['confidence_pct']}%</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #8d99ae; margin-top: 6px;">
                        Architecture: {result['model_used']}
                    </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_gauge:
                    gauge_fig = create_risk_gauge_figure(result["risk_score"], result["risk_label"])
                    st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})

                # PLAIN-ENGLISH REASON CARD
                st.markdown(f"""
                <div class="tl-card" style="border-left: 4px solid #00f0ff;">
                    <div class="tl-card-header">🧠 Explainable AI Rationale (Plain English)</div>
                    <div style="font-size: 1.05rem; line-height: 1.7; color: #ffffff;">
                        {result['plain_english_reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # MIDDLE ROW: Highlighted LIME text & Risk Breakdown
                col_xai, col_risk_bar = st.columns([1.3, 1.0])

                with col_xai:
                    st.markdown("""
                    <div class="tl-card">
                        <div class="tl-card-header">
                            <span>🔍 Linguistic Feature Contributions (LIME)</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #8d99ae; margin-bottom: 10px;">
                            <span style="color: #ff4b4b; font-weight: bold;">■ Red</span> pushes toward FAKE &nbsp;|&nbsp;
                            <span style="color: #00cc96; font-weight: bold;">■ Green/Teal</span> pushes toward REAL (hover words for weight)
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(result["highlighted_html"], unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_risk_bar:
                    st.markdown("""<div class="tl-card">""", unsafe_allow_html=True)
                    breakdown_fig = create_risk_breakdown_figure(result["risk_breakdown"], result["risk_max_possible"])
                    st.plotly_chart(breakdown_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown("</div>", unsafe_allow_html=True)

                # BOTTOM ROW: Source Credibility & Fact-Check Verification
                col_cred, col_fc = st.columns([1.0, 1.4])

                with col_cred:
                    cred = result["credibility"]
                    cat = cred["category"]
                    cat_badge_class = {
                        "Trusted": "badge-cat-trusted",
                        "Mixed": "badge-cat-mixed",
                        "Unreliable": "badge-cat-unreliable",
                        "Satire": "badge-cat-satire"
                    }.get(cat, "badge-cat-unknown")

                    st.markdown(f"""
                    <div class="tl-card">
                        <div class="tl-card-header">🌐 Source Credibility Analysis</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div>
                                <span style="font-size: 1.2rem; font-weight: 700; color: #ffffff;">{cred['source_name']}</span>
                                <div style="font-size: 0.85rem; color: #8d99ae;">Domain: {cred['domain']}</div>
                            </div>
                            <span class="{cat_badge_class}">{cat.upper()}</span>
                        </div>
                        <div style="margin: 14px 0;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                                <span>Credibility Score:</span>
                                <strong style="color: #00f0ff;">{cred['credibility_score']}/100</strong>
                            </div>
                            <div style="background-color: #0b132b; border-radius: 6px; height: 8px; width: 100%; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #ff4b4b 0%, #ffb703 50%, #00f0ff 100%); height: 100%; width: {cred['credibility_score']}%;"></div>
                            </div>
                        </div>
                        <div style="font-size: 0.85rem; color: #8d99ae; line-height: 1.5; margin-top: 10px;">
                            <strong>Derivation:</strong> {cred['derivation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_fc:
                    fc = result["factcheck"]
                    matches = fc["matches"]

                    st.markdown("""
                    <div class="tl-card">
                        <div class="tl-card-header">📋 Fact-Check Database Corroboration</div>
                    """, unsafe_allow_html=True)

                    if not matches:
                        st.info("No matching verified claims found in database.")
                    else:
                        table_html = "<table class='fc-table'><thead><tr><th>Verified Claim</th><th>Verdict</th><th>Sim.</th><th>Source</th></tr></thead><tbody>"
                        for m in matches:
                            v = m["verdict"].upper()
                            v_color = "#ff4b4b" if "FALSE" in v else ("#00cc96" if "TRUE" in v else "#ffb703")
                            table_html += f"""
                            <tr>
                                <td style="max-width: 260px;"><strong>{m['claim']}</strong><br><span style="font-size: 0.78rem; color: #8d99ae;">{m['explanation'][:110]}...</span></td>
                                <td><span style="color: {v_color}; font-weight: 700;">{v}</span></td>
                                <td>{m['similarity_pct']}%</td>
                                <td><span style="font-size: 0.8rem; color: #48cae4;">{m['fact_checker']}</span></td>
                            </tr>
                            """
                        table_html += "</tbody></table>"
                        st.markdown(table_html, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # DOWNLOAD REPORT BUTTON
                st.markdown("<hr style='border-color: #3a506b; margin: 20px 0;'>", unsafe_allow_html=True)
                report_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>TruthLens Verification Report</title>
                    <style>
                        body {{ font-family: sans-serif; background: #0b132b; color: #e0e6ed; padding: 30px; }}
                        .card {{ background: #1c2541; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #3a506b; }}
                        h1 {{ color: #00f0ff; }}
                        .verdict {{ font-size: 24px; font-weight: bold; color: {'#ff4b4b' if result['prediction']=='FAKE' else '#00cc96'}; }}
                    </style>
                </head>
                <body>
                    <h1>TruthLens – Fact Verification & Risk Audit Report</h1>
                    <div class="card">
                        <div>Prediction: <span class="verdict">{result['prediction']} ({result['confidence_pct']}%)</span></div>
                        <div>Overall Risk Score: <strong>{result['risk_score']}/100 ({result['risk_label']})</strong></div>
                        <div>Architecture: {result['model_used']}</div>
                    </div>
                    <div class="card">
                        <h3>Plain English Reason</h3>
                        <p>{result['plain_english_reason']}</p>
                    </div>
                    <div class="card">
                        <h3>Analyzed Text with Feature Highlights</h3>
                        <div>{result['highlighted_html']}</div>
                    </div>
                    <div class="card">
                        <h3>Source Credibility</h3>
                        <p>Source: {result['credibility']['domain']} (Score: {result['credibility']['credibility_score']}/100)</p>
                        <p>{result['credibility']['derivation']}</p>
                    </div>
                </body>
                </html>
                """
                b64_report = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64_report}" download="truthlens_report.html" style="background: linear-gradient(135deg, #00f0ff 0%, #0077b6 100%); color: #0b132b; padding: 10px 22px; border-radius: 6px; text-decoration: none; font-weight: bold;">📥 Download Complete Verification Report (HTML/PDF)</a>'
                st.markdown(href, unsafe_allow_html=True)


# =============================================================
# TAB 2: MODEL COMPARISON
# =============================================================
with tab_models:
    st.markdown("### 📊 Empirical Performance Benchmark & Limitations of Existing Systems")
    st.markdown("""
    This benchmark directly compares the fine-tuned **DistilBERT transformer** against existing classical systems 
    (**Naive Bayes, Linear SVM, Random Forest, and Bi-LSTM**) evaluated on the identical holdout test split.
    """)

    results_df = get_cached_results()
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#1d4ed8")
                        .highlight_min(axis=0, subset=["False Positive Rate"], color="#047857"),
        use_container_width=True
    )

    col_chart, col_cm = st.columns([1.2, 0.8])

    with col_chart:
        chart_path = config.RESULTS_DIR / "model_comparison.png"
        if chart_path.exists():
            st.image(str(chart_path), caption="Grouped Bar Chart: Comparison Across All Evaluation Metrics", use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate the comparison chart.")

    with col_cm:
        cm_path = config.RESULTS_DIR / "bert_confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="DistilBERT Confusion Matrix", use_container_width=True)
        else:
            st.info("Fine-tune BERT (`python src/bert_model.py --quick`) to generate confusion matrix.")

    st.markdown("""
    <div class="tl-card" style="margin-top: 20px;">
        <div class="tl-card-header">🎓 Theoretical Analysis for Project Presentation</div>
        <p><strong>1. Why BERT Outperforms Existing Shallow Systems:</strong></p>
        <ul>
            <li><strong>Contextual Polysemy:</strong> Traditional Bag-of-Words (TF-IDF) models assign static weights to words regardless of syntax. Legitimate articles quoting sensational statements are frequently misclassified by Naive Bayes and SVM.</li>
            <li><strong>Bidirectional Self-Attention:</strong> DistilBERT processes tokens in context of the entire sentence, capturing negation ("not true"), sarcasm, and structural tone that sequential LSTMs lose over long texts.</li>
            <li><strong>False Positive Rate (FPR):</strong> High FPR in classical models means legitimate investigative journalism is mistakenly flagged. Transformers significantly suppress false alarms.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# TAB 3: CREDIBILITY GRAPH
# =============================================================
with tab_graph:
    st.markdown("### 🌐 Source Credibility & Influence Network")
    st.markdown("""
    The network graph models inter-source syndication, citation patterns, and audience overlap. 
    Credibility scores propagate from trusted seed authorities (e.g., Reuters, AP, WHO) through neighbor-weighted PageRank.
    """)

    cred_mgr = engines["credibility_mgr"]

    # Render Plotly interactive network
    plotly_fig = cred_mgr.render_plotly_figure()
    st.plotly_chart(plotly_fig, use_container_width=True)

    # Search & Table of Sources
    st.markdown("#### Curated Source Reputation Index (~60 Domains)")
    search_query = st.text_input("Filter Sources by Name or Domain:", placeholder="e.g., bbc, onion, infowars, hindu...")

    df_sources = cred_mgr.sources_df.copy()
    if search_query:
        df_sources = df_sources[
            df_sources["domain"].str.contains(search_query, case=False, na=False) |
            df_sources["name"].str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(
        df_sources[["domain", "name", "credibility_score", "category", "country", "notes"]],
        use_container_width=True,
        hide_index=True
    )


# =============================================================
# TAB 4: HOW IT WORKS & PPT ALIGNMENT
# =============================================================
with tab_about:
    st.markdown("### 📖 Architecture, Datasets, and System Modules")

    st.markdown("""
    <div class="tl-card">
        <div class="tl-card-header">📐 7-Stage Architectural Pipeline</div>
        <div style="font-family: monospace; background: #0b132b; padding: 14px; border-radius: 6px; color: #00f0ff; font-size: 0.95rem; line-height: 1.6;">
            [News Article / Post] ➡️ [1. Text Preprocessing] ➡️ [2. DistilBERT Transformer] ➡️ [3. Credibility Analysis] <br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⬇️<br>
            [Prediction: FAKE / REAL + Risk Score] ⬅️ [6. Explainable AI (LIME)] ⬅️ [5. Risk Scoring Engine] ⬅️ [4. Fact-Check Verification]
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_desc1, col_desc2 = st.columns(2)

    with col_desc1:
        st.markdown("""
        <div class="tl-card">
            <div class="tl-card-header">📦 Core Subsystems</div>
            <ul>
                <li><strong>1. Preprocessing:</strong> URL stripping, HTML normalization, stylistic signal computation (ALL-CAPS ratio, clickbait phrasing, emotional extremity).</li>
                <li><strong>2. DistilBERT:</strong> 6-layer 66M-parameter bidirectional transformer fine-tuned for sequence classification.</li>
                <li><strong>3. Credibility Graph:</strong> NetworkX directed graph modeling media citation and syndication networks with PageRank propagation.</li>
                <li><strong>4. Fact-Checking:</strong> <code>all-MiniLM-L6-v2</code> sentence embeddings matching input text against 170+ verified claims with disk caching.</li>
                <li><strong>5. Risk Engine:</strong> Weighted fusion formula scaled 0-100 (Low: 0-33, Medium: 34-66, High: 67-100).</li>
                <li><strong>6. Explainability:</strong> LIME perturbation analysis generating word-level contribution weights and plain-English rationales.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_desc2:
        st.markdown("""
        <div class="tl-card">
            <div class="tl-card-header">⚠️ Limitations & Future Scope</div>
            <p><strong>Current Project Limitations:</strong></p>
            <ul>
                <li><strong>Satire vs Malice:</strong> Satirical news (e.g., The Onion) often mimics fake news linguistic cues and relies on source domain flags.</li>
                <li><strong>Zero-Day Breaking Events:</strong> Unseen breaking events not yet present in fact-checking databases rely on text and credibility features.</li>
                <li><strong>Monolingual Focus:</strong> Primarily tuned for English news items.</li>
            </ul>
            <p><strong>Future Research Scope:</strong></p>
            <ul>
                <li>Multimodal deepfake detection (image & video forensic verification).</li>
                <li>Browser extension for real-time social media feed inspection.</li>
                <li>Multilingual support using XLM-RoBERTa for regional Indian languages.</li>
                <li>Blockchain-backed cryptographic provenance tracking for digital journalism.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
