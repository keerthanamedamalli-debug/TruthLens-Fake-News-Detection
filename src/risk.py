"""
Multimodal Risk Scoring Module for TruthLens.
Computes an overall Risk Score (0 to 100) fusing:
1. BERT / Transformer Fake Probability (Weight: 45%)
2. Source Unreliability [ (100 - Credibility)/100 ] (Weight: 25%)
3. Fact-Check Contradiction Score (Weight: 20%)
4. Stylistic & Clickbait Signal Score (Weight: 10%)

Provides categorical risk classification (Low / Medium / High) and Plotly visualization builders.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

logger = logging.getLogger("TruthLens-Risk")


def calculate_risk_score(
    bert_fake_prob: float,
    source_credibility: float,
    factcheck_contradiction: float,
    style_score: float
) -> Dict[str, Any]:
    """
    Calculates weighted multimodal risk score (0-100) and component contributions.
    """
    # Clamp inputs to safe normalized ranges
    p_bert = min(max(float(bert_fake_prob), 0.0), 1.0)
    source_cred = min(max(float(source_credibility), 0.0), 100.0)
    p_source_risk = (100.0 - source_cred) / 100.0
    p_factcheck = min(max(float(factcheck_contradiction), 0.0), 1.0)
    p_style = min(max(float(style_score), 0.0), 100.0)
    if p_style > 1.0:
        p_style = p_style / 100.0

    # Calculate weighted contributions (scale 0-100)
    w_bert = config.RISK_WEIGHT_BERT
    w_cred = config.RISK_WEIGHT_CREDIBILITY
    w_fc = config.RISK_WEIGHT_FACTCHECK
    w_style = config.RISK_WEIGHT_STYLE

    contrib_bert = w_bert * p_bert * 100.0
    contrib_cred = w_cred * p_source_risk * 100.0
    contrib_fc = w_fc * p_factcheck * 100.0
    contrib_style = w_style * p_style * 100.0

    raw_total = contrib_bert + contrib_cred + contrib_fc + contrib_style
    total_score = round(min(max(raw_total, 0.0), 100.0), 1)

    # Assign risk label and color
    if total_score <= config.RISK_LOW_MAX:
        label = "Low Risk"
        color = "#00cc96"
        description = "Content displays high credibility, factual consistency, and measured journalistic tone."
    elif total_score <= config.RISK_MED_MAX:
        label = "Medium Risk"
        color = "#ffb703"
        description = "Contains mixed signals: unverified source, subtle sensationalism, or incomplete factual corroboration."
    else:
        label = "High Risk"
        color = "#ff4b4b"
        description = "Severe misinformation indicators: high model fake probability, debunked claim match, or low-credibility source."

    breakdown = {
        "AI Model Prediction (BERT)": round(contrib_bert, 1),
        "Source Unreliability": round(contrib_cred, 1),
        "Fact-Check Contradiction": round(contrib_fc, 1),
        "Stylistic & Clickbait Anomalies": round(contrib_style, 1)
    }

    max_possible = {
        "AI Model Prediction (BERT)": round(w_bert * 100.0, 1),
        "Source Unreliability": round(w_cred * 100.0, 1),
        "Fact-Check Contradiction": round(w_fc * 100.0, 1),
        "Stylistic & Clickbait Anomalies": round(w_style * 100.0, 1)
    }

    return {
        "risk_score": total_score,
        "risk_label": label,
        "risk_color": color,
        "risk_description": description,
        "breakdown": breakdown,
        "max_possible": max_possible
    }


def create_risk_gauge_figure(risk_score: float, risk_label: str):
    """Creates a high-contrast dark theme Plotly gauge indicator for the Risk Score."""
    import plotly.graph_objects as go

    if risk_score <= config.RISK_LOW_MAX:
        bar_color = "#00cc96"
    elif risk_score <= config.RISK_MED_MAX:
        bar_color = "#ffb703"
    else:
        bar_color = "#ff4b4b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Overall Risk: {risk_label}", 'font': {'size': 20, 'color': '#e0e6ed'}},
        number={'suffix': "/100", 'font': {'color': '#ffffff', 'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#3a506b", 'tickfont': {'color': '#8d99ae'}},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "#1c2541",
            'borderwidth': 1,
            'bordercolor': "#3a506b",
            'steps': [
                {'range': [0, config.RISK_LOW_MAX], 'color': 'rgba(0, 204, 150, 0.2)'},
                {'range': [config.RISK_LOW_MAX, config.RISK_MED_MAX], 'color': 'rgba(255, 183, 3, 0.2)'},
                {'range': [config.RISK_MED_MAX, 100], 'color': 'rgba(255, 75, 75, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': risk_score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260
    )
    return fig


def create_risk_breakdown_figure(breakdown: Dict[str, float], max_possible: Dict[str, float]):
    """Creates a stacked horizontal bar chart displaying component risk contributions."""
    import plotly.graph_objects as go

    categories = list(breakdown.keys())
    actual_values = [breakdown[k] for k in categories]
    max_values = [max_possible[k] for k in categories]
    colors = ["#00f0ff", "#ffb703", "#ff4b4b", "#c77dff"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=categories,
        x=actual_values,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#0b132b', width=1)),
        text=[f"{v:.1f} pts" for v in actual_values],
        textposition='auto',
        hoverinfo='x+y'
    ))

    fig.update_layout(
        title=dict(text="Risk Score Component Breakdown", font=dict(color="#00f0ff", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Contributed Risk Points (Max Total: 100)", font=dict(color="#8d99ae", size=11)),
            range=[0, 50],
            showgrid=True,
            gridcolor="#1c2541",
            tickfont=dict(color="#8d99ae")
        ),
        yaxis=dict(
            tickfont=dict(color="#e0e6ed", size=12),
            autorange="reversed"
        ),
        margin=dict(l=10, r=20, t=40, b=30),
        height=240
    )
    return fig
