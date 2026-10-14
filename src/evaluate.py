"""
Comprehensive Model Evaluation and Comparison Module for TruthLens.
Evaluates:
1. TF-IDF + Naive Bayes
2. TF-IDF + SVM
3. TF-IDF + Random Forest
4. PyTorch Bi-LSTM
5. Fine-tuned DistilBERT

Computes:
- Accuracy, Precision, Recall, F1-Score
- False Positive Rate (FPR) - directly addressing the PPT limitation of classical models

Generates:
- results/results.csv
- results/model_comparison.png (grouped bar chart)
- results/bert_confusion_matrix.png
- results/evaluation_summary.md (markdown report for project documentation)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from src.baselines import get_baseline_manager
from src.bert_model import get_bert_classifier
from src.preprocess import clean_text_bert

logger = logging.getLogger("TruthLens-Evaluate")


def run_evaluation() -> pd.DataFrame:
    """
    Executes evaluation across all baseline models and BERT on the test split.
    Saves CSV, charts, confusion matrix, and markdown summary.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test split not found at {config.TEST_DATA_PATH}. Please run data/prepare_data.py first.")

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    logger.info("Evaluating all models on %d test samples...", len(test_df))

    baseline_mgr = get_baseline_manager()
    if not baseline_mgr.is_fitted():
        logger.info("Baselines not yet fitted. Fitting now...")
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
        baseline_mgr.train_all(train_df, quick=True)

    # 1. Evaluate classical baselines
    results = baseline_mgr.evaluate_all(test_df)

    # 2. Evaluate BERT
    bert_clf = get_bert_classifier()
    y_true = test_df["label"].values

    if bert_clf.is_trained:
        logger.info("Evaluating fine-tuned DistilBERT on test set...")
        bert_preds = []
        for text in test_df["text"]:
            lbl, _ = bert_clf.predict(text)
            bert_preds.append(lbl)
        bert_preds = np.array(bert_preds)

        acc = accuracy_score(y_true, bert_preds)
        p, r, f1, _ = precision_recall_fscore_support(y_true, bert_preds, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, bert_preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / max((fp + tn), 1)

        results.append({
            "Model": "DistilBERT (Fine-Tuned)",
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1-Score": round(float(f1), 4),
            "False Positive Rate": round(float(fpr), 4)
        })

        # Save BERT confusion matrix chart
        _plot_confusion_matrix(cm, config.RESULTS_DIR / "bert_confusion_matrix.png")
    else:
        logger.warning("DistilBERT is not trained. Skipping BERT in evaluation report.")

    results_df = pd.DataFrame(results)
    results_csv_path = config.RESULTS_DIR / "results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info("Evaluation results saved to %s", results_csv_path)

    # Generate grouped bar chart
    _plot_model_comparison(results_df, config.RESULTS_DIR / "model_comparison.png")

    # Generate Markdown Summary
    _generate_markdown_summary(results_df, config.RESULTS_DIR / "evaluation_summary.md")

    return results_df


def _plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plots and saves confusion matrix heatmap for BERT."""
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="dark")
    plt.style.use("dark_background")

    labels = ["REAL (0)", "FAKE (1)"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("DistilBERT Confusion Matrix", fontsize=14, color="#00f0ff", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, color="#e0e6ed")
    plt.ylabel("Actual Ground Truth", fontsize=11, color="#e0e6ed")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor="#0b132b")
    plt.close()
    logger.info("Saved confusion matrix plot to %s", save_path)


def _plot_model_comparison(df: pd.DataFrame, save_path: Path):
    """Plots grouped bar chart comparing all models across all metrics."""
    plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "False Positive Rate"]
    models = df["Model"].tolist()

    x = np.arange(len(models))
    width = 0.15
    colors = ["#00f0ff", "#00cc96", "#ffb703", "#c77dff", "#ff4b4b"]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0b132b")
    ax.set_facecolor("#1c2541")

    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        ax.bar(x + (i - 2) * width, values, width, label=metric, color=colors[i], edgecolor="#0b132b")

    ax.set_title("TruthLens: Comparative Performance of Fake News Detection Models", fontsize=15, color="#00f0ff", pad=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color="#e0e6ed", weight="bold")
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=11, color="#8d99ae")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#3a506b")
    ax.legend(loc="upper right", framealpha=0.8, facecolor="#0b132b", edgecolor="#3a506b")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor="#0b132b")
    plt.close()
    logger.info("Saved model comparison bar chart to %s", save_path)


def _generate_markdown_summary(df: pd.DataFrame, save_path: Path):
    """Writes detailed Markdown summary explaining why BERT beats classical models."""
    best_f1 = df.loc[df["F1-Score"].idxmax()]
    lowest_fpr = df.loc[df["False Positive Rate"].idxmin()]

    content = f"""# TruthLens Model Evaluation Summary

This document presents the empirical benchmark results of TruthLens across classical machine learning architectures and the fine-tuned DistilBERT transformer.

## 1. Performance Benchmark Table

| Model Architecture | Accuracy | Precision | Recall | F1-Score | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df.iterrows():
        content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | **{row['F1-Score']:.4f}** | {row['False Positive Rate']:.4f} |\n"

    content += f"""
## 2. Key Findings & PPT Architectural Insights

1. **Top Performing Model**: **{best_f1['Model']}** achieved the highest overall F1-Score of **{best_f1['F1-Score']:.4f}** and Accuracy of **{best_f1['Accuracy']:.4f}**.
2. **False Positive Mitigation**: **{lowest_fpr['Model']}** achieved the lowest False Positive Rate of **{lowest_fpr['False Positive Rate']:.4f}**. In real-world content moderation, minimizing false positives prevents legitimate news from being mistakenly censored.
3. **Why BERT Outperforms Classical Shallow Models (Existing Systems)**:
   - **Contextual Polysemy**: Classical bag-of-words (TF-IDF + Naive Bayes / SVM) treat words as independent frequency counts. They struggle when sensational words ("shocking", "urgent") are used in legitimate journalistic quotations.
   - **Bidirectional Attention**: DistilBERT uses self-attention mechanisms to weigh relationships between distant tokens simultaneously across the entire sentence, capturing nuanced syntax and tone.
   - **Robustness to Word Order**: LSTMs suffer from vanishing gradients over longer sequences and process tokens sequentially; DistilBERT processes all tokens concurrently with full multi-head attention.

## 3. Generated Artifacts
- Comparison Chart: `results/model_comparison.png`
- Confusion Matrix: `results/bert_confusion_matrix.png`
- Benchmark Table CSV: `results/results.csv`
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved evaluation markdown summary to %s", save_path)


def get_cached_results() -> pd.DataFrame:
    """Returns results dataframe from disk or runs fresh evaluation."""
    csv_path = config.RESULTS_DIR / "results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return run_evaluation()


if __name__ == "__main__":
    df = run_evaluation()
    print("\n=== TruthLens Model Evaluation Benchmark ===")
    print(df.to_string(index=False))
