"""
Transformer Classification Module for TruthLens.
Fine-tunes and queries DistilBERT (distilbert-base-uncased) for binary fake news classification.
Includes:
- Standard PyTorch fine-tuning loop with AdamW and linear lr scheduler.
- Quick training mode (--quick) for rapid demo initialization in ~2 minutes on CPU.
- Batch inference compatibility for LIME / SHAP explainability.
- Graceful fallback detection when the model is not yet trained.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
from src.preprocess import clean_text_bert

logger = logging.getLogger("TruthLens-BERT")


class NewsDataset(Dataset):
    """PyTorch Dataset for tokenized news text and binary labels."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertClassifier:
    """
    Manages loading, fine-tuning, and inference for DistilBERT.
    """
    def __init__(self, model_dir: Path = config.BERT_MODEL_DIR):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_trained = False

        self._try_load_model()

    def _try_load_model(self):
        """Attempts to load fine-tuned model and tokenizer from disk."""
        if (self.model_dir / "config.json").exists() and (
            (self.model_dir / "model.safetensors").exists() or (self.model_dir / "pytorch_model.bin").exists()
        ):
            try:
                logger.info("Loading fine-tuned DistilBERT from %s onto %s...", self.model_dir, self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                self.model.to(self.device)
                self.model.eval()
                self.is_trained = True
                logger.info("DistilBERT loaded successfully.")
            except Exception as e:
                logger.warning("Could not load fine-tuned BERT: %s", e)
                self.is_trained = False
        else:
            self.is_trained = False

    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, quick: bool = False):
        """
        Fine-tunes distilbert-base-uncased on the provided dataset.
        If quick=True, trains for 1 epoch on a small sample subset (~400) for instant demo preparation.
        """
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        if quick:
            num_samples = min(len(train_df), config.QUICK_TRAIN_SAMPLES)
            num_epochs = config.QUICK_NUM_EPOCHS
            logger.info("⚡ Quick Training Mode: Subsampling to %d samples, %d epoch.", num_samples, num_epochs)
            train_subset = train_df.sample(num_samples, random_state=config.RANDOM_SEED)
        else:
            num_samples = min(len(train_df), config.FULL_TRAIN_SAMPLES)
            num_epochs = config.NUM_EPOCHS
            logger.info("Full Training Mode: Subsampling to %d samples, %d epochs.", num_samples, num_epochs)
            train_subset = train_df.sample(num_samples, random_state=config.RANDOM_SEED)

        logger.info("Initializing base tokenizer and model '%s'...", config.BERT_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.BERT_MODEL_NAME, num_labels=2)
        self.model.to(self.device)

        # Preprocess and tokenize text
        train_texts = [clean_text_bert(t) for t in train_subset["text"]]
        train_labels = train_subset["label"].tolist()

        logger.info("Tokenizing %d training texts...", len(train_texts))
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_TOKEN_LENGTH,
            return_tensors="pt"
        )

        train_dataset = NewsDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

        # PyTorch training loop
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"BERT Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            logger.info("Epoch %d completed. Average Loss: %.4f", epoch + 1, avg_loss)

        # Save to disk
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        self.model.eval()
        self.is_trained = True
        logger.info("Fine-tuned BERT model saved to %s", self.model_dir)

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Classifies a single text string.
        Returns:
            label (int): 0 for REAL, 1 for FAKE
            fake_probability (float): probability of class 1 (0.0 to 1.0)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("BERT model is not trained or loaded.")

        clean = clean_text_bert(text)
        inputs = self.tokenizer(
            clean,
            truncation=True,
            padding=True,
            max_length=config.MAX_TOKEN_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        label = int(np.argmax(probs))
        fake_prob = float(probs[1])
        return label, fake_prob

    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch probability predictor designed for LIME explainer.
        Accepts a list of N strings, returns an (N, 2) numpy array of [P(Real), P(Fake)].
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("BERT model is not trained.")

        cleaned = [clean_text_bert(t) for t in texts]
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=config.MAX_TOKEN_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

        return probs


# Singleton instance
_bert_instance = None

def get_bert_classifier() -> BertClassifier:
    global _bert_instance
    if _bert_instance is None:
        _bert_instance = BertClassifier()
    return _bert_instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune BERT on TruthLens dataset")
    parser.add_argument("--quick", action="store_true", help="Quick 1-epoch demo fine-tuning")
    args = parser.parse_args()

    # Load training data
    if not config.TRAIN_DATA_PATH.exists():
        from data.prepare_data import prepare_data
        prepare_data()

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH) if config.VAL_DATA_PATH.exists() else None

    clf = get_bert_classifier()
    clf.train(train_df, val_df, quick=args.quick)
