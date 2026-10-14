"""
Configuration module for TruthLens.
Centralizes paths, model hyperparameters, feature flags, and risk scoring weights.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"

# Sub-paths for Models & Artifacts
BERT_MODEL_DIR = MODELS_DIR / "bert"
BASELINES_DIR = MODELS_DIR / "baselines"
CACHE_DIR = BASE_DIR / ".cache"

# Data File Paths
PROCESSED_DATA_PATH = DATA_DIR / "news_dataset.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
VAL_DATA_PATH = DATA_DIR / "val.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SOURCE_CREDIBILITY_PATH = DATA_DIR / "source_credibility.csv"
SOURCE_GRAPH_PATH = DATA_DIR / "source_graph.json"
FACTCHECKS_PATH = DATA_DIR / "factchecks.csv"
FACTCHECK_EMBEDDINGS_CACHE = CACHE_DIR / "factcheck_embeddings.pt"

# Create directories if they do not exist
for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR, BERT_MODEL_DIR, BASELINES_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# BERT / Transformer Hyperparameters
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 128  # Keep at 128 for snappy CPU inference & training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
QUICK_NUM_EPOCHS = 1
QUICK_TRAIN_SAMPLES = 400
FULL_TRAIN_SAMPLES = 3200
RANDOM_SEED = 42

# Sentence Transformer for Fact-checking
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
TOP_K_FACTCHECKS = 3
FACTCHECK_SIMILARITY_THRESHOLD = 0.40

# Explainability (LIME) Settings
LIME_NUM_SAMPLES = 200  # Fast enough on CPU for live demo (~1-2 seconds)
LIME_NUM_FEATURES = 8

# Risk Score Configuration
# Formula: 0.45 * BERT_prob + 0.25 * (100 - credibility)/100 + 0.20 * factcheck_contradiction + 0.10 * style_score
RISK_WEIGHT_BERT = 0.45
RISK_WEIGHT_CREDIBILITY = 0.25
RISK_WEIGHT_FACTCHECK = 0.20
RISK_WEIGHT_STYLE = 0.10

RISK_LOW_MAX = 33
RISK_MED_MAX = 66

# Clickbait & Sensational Phrases for Style Analysis
CLICKBAIT_PHRASES = [
    "you won't believe", "you wont believe",
    "shocking", "bombshell", "secret uncovered",
    "miracle cure", "doctors hate", "government doesn't want you to know",
    "what happened next will blow your mind", "unbelievable truth",
    "breaking:", "breaking news:", "urgent alert",
    "banned video", "100% proof", "secret leaked",
    "scientists are stunned", "they don't want you to see",
    "the real truth behind", "mind-blowing", "share before deleted"
]

# Google Fact Check Tools API Key (Optional)
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
