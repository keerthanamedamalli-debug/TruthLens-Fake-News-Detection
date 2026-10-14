"""
Baseline Machine Learning Models Module for TruthLens.
Implements:
1. TF-IDF + Multinomial Naive Bayes
2. TF-IDF + Support Vector Machine (LinearSVC with Platt Calibration for probabilities)
3. TF-IDF + Random Forest Classifier
4. PyTorch Text LSTM Classifier (Word2Vec / Learned Embeddings + Bidirectional LSTM)

These represent the existing classical systems to compare against the BERT transformer.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from src.preprocess import clean_text_baseline

logger = logging.getLogger("TruthLens-Baselines")


# -------------------------------------------------------------
# PyTorch LSTM Model
# -------------------------------------------------------------
class TextLSTM(nn.Module):
    """Simple 2-layer Bidirectional LSTM for text classification."""
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (hn, cn) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        feat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        logits = self.fc(self.dropout(feat))
        return logits


class SimpleVocab:
    """Builds a small token dictionary for the LSTM model."""
    def __init__(self, max_vocab: int = 10000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: List[str]):
        freq = {}
        for text in texts:
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:self.max_vocab - 2]
        for idx, (word, _) in enumerate(sorted_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        tokens = [self.word2idx.get(w, 1) for w in text.split()[:max_len]]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return tokens


class LSTMDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: SimpleVocab, max_len: int = 64):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# -------------------------------------------------------------
# Baselines Trainer & Evaluator
# -------------------------------------------------------------
class BaselineModels:
    """
    Handles fitting, serializing, and querying classical baselines
    (Naive Bayes, SVM, Random Forest, LSTM).
    """
    def __init__(self):
        self.models_dir = config.BASELINES_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = None
        self.nb_model = None
        self.svm_model = None
        self.rf_model = None
        self.lstm_model = None
        self.vocab = None

        self._load_saved_models()

    def _load_saved_models(self):
        """Loads serialized models if they exist on disk."""
        vec_path = self.models_dir / "tfidf_vectorizer.joblib"
        nb_path = self.models_dir / "naive_bayes.joblib"
        svm_path = self.models_dir / "svm.joblib"
        rf_path = self.models_dir / "random_forest.joblib"
        lstm_path = self.models_dir / "lstm_model.pt"
        vocab_path = self.models_dir / "vocab.joblib"

        if vec_path.exists():
            self.vectorizer = joblib.load(vec_path)
        if nb_path.exists():
            self.nb_model = joblib.load(nb_path)
        if svm_path.exists():
            self.svm_model = joblib.load(svm_path)
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
        if vocab_path.exists():
            self.vocab = joblib.load(vocab_path)
        if lstm_path.exists() and self.vocab:
            try:
                self.lstm_model = TextLSTM(vocab_size=len(self.vocab.word2idx), embed_dim=48, hidden_dim=48)
                self.lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
                self.lstm_model.eval()
            except Exception as e:
                logger.warning("Could not load LSTM weights: %s", e)

    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.svm_model is not None

    def train_all(self, train_df: pd.DataFrame, quick: bool = False):
        """
        Trains TF-IDF vectorizer and classical models + LSTM.
        """
        logger.info("Training classical baselines (samples: %d, quick=%s)...", len(train_df), quick)

        # Preprocess text for baselines
        clean_train = [clean_text_baseline(t) for t in train_df["text"]]
        labels = train_df["label"].values

        # 1. TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english")
        X_train_tfidf = self.vectorizer.fit_transform(clean_train)
        joblib.dump(self.vectorizer, self.models_dir / "tfidf_vectorizer.joblib")

        # 2. Naive Bayes
        logger.info("Training Naive Bayes...")
        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_train_tfidf, labels)
        joblib.dump(self.nb_model, self.models_dir / "naive_bayes.joblib")

        # 3. SVM with Probability Calibration
        logger.info("Training SVM (Calibrated LinearSVC)...")
        base_svm = LinearSVC(C=1.0, random_state=config.RANDOM_SEED, dual="auto")
        self.svm_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
        self.svm_model.fit(X_train_tfidf, labels)
        joblib.dump(self.svm_model, self.models_dir / "svm.joblib")

        # 4. Random Forest
        logger.info("Training Random Forest...")
        n_estimators = 50 if quick else 100
        self.rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=config.RANDOM_SEED, n_jobs=-1)
        self.rf_model.fit(X_train_tfidf, labels)
        joblib.dump(self.rf_model, self.models_dir / "random_forest.joblib")

        # 5. Small PyTorch LSTM
        logger.info("Training PyTorch LSTM...")
        self.vocab = SimpleVocab(max_vocab=5000)
        self.vocab.fit(clean_train)
        joblib.dump(self.vocab, self.models_dir / "vocab.joblib")

        lstm_dataset = LSTMDataset(clean_train, labels, self.vocab, max_len=64)
        loader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

        self.lstm_model = TextLSTM(vocab_size=len(self.vocab.word2idx), embed_dim=48, hidden_dim=48)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.003)

        epochs = 2 if quick else 4
        self.lstm_model.train()
        for ep in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.lstm_model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.lstm_model.eval()
        torch.save(self.lstm_model.state_dict(), self.models_dir / "lstm_model.pt")

        logger.info("Classical baselines trained and saved to %s", self.models_dir)

    def predict_svm(self, text: str) -> Tuple[int, float]:
        """Inference helper for SVM fallback: returns (label, fake_probability)."""
        if not self.is_fitted():
            return 1, 0.5
        clean = clean_text_baseline(text)
        vec = self.vectorizer.transform([clean])
        proba = self.svm_model.predict_proba(vec)[0]
        label = int(np.argmax(proba))
        fake_prob = float(proba[1])
        return label, fake_prob

    def evaluate_all(self, test_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluates all classical baselines on the test split.
        Returns metrics dictionaries with Accuracy, Precision, Recall, F1, and False Positive Rate.
        """
        if not self.is_fitted():
            raise RuntimeError("Baseline models are not fitted. Call train_all() first.")

        clean_test = [clean_text_baseline(t) for t in test_df["text"]]
        y_true = test_df["label"].values
        X_test_tfidf = self.vectorizer.transform(clean_test)

        results = []

        # 1. Naive Bayes
        preds_nb = self.nb_model.predict(X_test_tfidf)
        results.append(self._calculate_metrics("Naive Bayes", y_true, preds_nb))

        # 2. SVM
        preds_svm = self.svm_model.predict(X_test_tfidf)
        results.append(self._calculate_metrics("SVM (Linear)", y_true, preds_svm))

        # 3. Random Forest
        preds_rf = self.rf_model.predict(X_test_tfidf)
        results.append(self._calculate_metrics("Random Forest", y_true, preds_rf))

        # 4. PyTorch LSTM
        if self.lstm_model is not None and self.vocab is not None:
            self.lstm_model.eval()
            lstm_dataset = LSTMDataset(clean_test, y_true, self.vocab, max_len=64)
            loader = DataLoader(lstm_dataset, batch_size=32, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for batch_x, _ in loader:
                    logits = self.lstm_model(batch_x)
                    preds = torch.argmax(logits, dim=1).numpy()
                    all_preds.extend(preds)
            results.append(self._calculate_metrics("Bi-LSTM (PyTorch)", y_true, np.array(all_preds)))

        return results

    def _calculate_metrics(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculates Accuracy, Precision, Recall, F1, and False Positive Rate (FPR)."""
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        # Confusion matrix: [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / max((fp + tn), 1)
        else:
            fpr = 0.0

        return {
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1-Score": round(float(f1), 4),
            "False Positive Rate": round(float(fpr), 4)
        }


# Singleton instance
_baseline_manager = None

def get_baseline_manager() -> BaselineModels:
    global _baseline_manager
    if _baseline_manager is None:
        _baseline_manager = BaselineModels()
    return _baseline_manager
