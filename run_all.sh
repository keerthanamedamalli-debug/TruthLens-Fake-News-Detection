#!/usr/bin/env bash
# =====================================================================
# TruthLens: Fake News Detection Using Explainable AI
# Complete automated setup, training, evaluation, and dashboard launch
# =====================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================================="
echo "       TruthLens: Fake News Detection Using Explainable AI           "
echo "====================================================================="

echo ""
echo "[Step 1/5] Installing Dependencies..."
pip install -r requirements.txt || true

echo ""
echo "[Step 2/5] Preparing Datasets & Fact-Checks..."
python data/generate_factchecks.py
python data/prepare_data.py

echo ""
echo "[Step 3/5] Training Classical Baseline Models..."
python -c "import pandas as pd, config; from src.baselines import get_baseline_manager; df=pd.read_csv(config.TRAIN_DATA_PATH); get_baseline_manager().train_all(df, quick=True)"

echo ""
echo "[Step 4/5] Fine-Tuning DistilBERT Transformer (Quick Demo Mode)..."
python src/bert_model.py --quick

echo ""
echo "[Step 5/5] Running Model Evaluation..."
python src/evaluate.py

echo ""
echo "====================================================================="
echo "[SUCCESS] Launching Streamlit Dashboard..."
echo "====================================================================="
streamlit run app.py
