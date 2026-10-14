"""
Fact-Check Verification Module for TruthLens.
- Embeds verified claims from data/factchecks.csv using sentence-transformers (all-MiniLM-L6-v2).
- Caches precomputed claim embeddings to disk for instant offline retrieval.
- Computes cosine similarity to find top matching claims, their verdicts, and explanations.
- Calculates a contradiction score (0.0 to 1.0) indicating whether the input matches debunked claims.
- Gracefully checks the Google Fact Check Tools API if an API key is provided, falling back silently offline.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch

# Ensure parent directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config

logger = logging.getLogger("TruthLens-FactCheck")


class FactCheckVerifier:
    """
    Verifies input text against a local database of 170+ verified claims using semantic embeddings,
    with an optional fallback to Google Fact Check Tools API.
    """
    def __init__(self):
        self.df = self._load_database()
        self.encoder = None
        self.embeddings = None
        self._init_encoder_and_embeddings()

    def _load_database(self) -> pd.DataFrame:
        if config.FACTCHECKS_PATH.exists():
            df = pd.read_csv(config.FACTCHECKS_PATH)
            logger.info("Loaded %d verified claims from %s", len(df), config.FACTCHECKS_PATH)
            return df
        logger.warning("Factcheck database not found at %s. Creating empty fallback.", config.FACTCHECKS_PATH)
        return pd.DataFrame(columns=["claim_id", "claim", "verdict", "category", "explanation", "fact_checker", "reference_url"])

    def _init_encoder_and_embeddings(self):
        """Loads sentence-transformers model and generates/loads cached embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model '%s'...", config.SENTENCE_TRANSFORMER_MODEL)
            self.encoder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

            # Check for disk cache
            if config.FACTCHECK_EMBEDDINGS_CACHE.exists():
                logger.info("Loading cached claim embeddings from %s", config.FACTCHECK_EMBEDDINGS_CACHE)
                cached_data = torch.load(config.FACTCHECK_EMBEDDINGS_CACHE, weights_only=False)
                if isinstance(cached_data, dict) and "embeddings" in cached_data:
                    # Validate length
                    if len(cached_data["embeddings"]) == len(self.df):
                        self.embeddings = cached_data["embeddings"]
                        logger.info("Successfully loaded %d cached embeddings.", len(self.embeddings))
                        return

            # Compute and cache embeddings
            claims = self.df["claim"].tolist()
            if claims:
                logger.info("Computing embeddings for %d claims...", len(claims))
                emb = self.encoder.encode(claims, convert_to_tensor=True, show_progress_bar=False)
                self.embeddings = emb
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"embeddings": emb, "count": len(claims)}, config.FACTCHECK_EMBEDDINGS_CACHE)
                logger.info("Saved embeddings cache to %s", config.FACTCHECK_EMBEDDINGS_CACHE)

        except Exception as e:
            logger.warning("SentenceTransformer failed to initialize: %s. Using TF-IDF fallback.", e)
            self.encoder = None
            self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        """Fallback lightweight TF-IDF cosine similarity if sentence-transformers is offline."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        claims = self.df["claim"].fillna("").tolist()
        if claims:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(claims)
        else:
            self.tfidf_matrix = None

    def verify(self, query_text: str, top_k: int = config.TOP_K_FACTCHECKS) -> Dict[str, Any]:
        """
        Retrieves top_k closest verified claims for the given input query and computes a contradiction score.
        """
        if not query_text or not query_text.strip() or len(self.df) == 0:
            return {
                "matches": [],
                "top_match": None,
                "contradiction_score": 0.0,
                "summary": "No matching fact-checks found in database."
            }

        matches = []

        # 1. Semantic Embedding Retrieval
        if self.encoder is not None and self.embeddings is not None:
            try:
                from sentence_transformers import util
                query_emb = self.encoder.encode(query_text, convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, self.embeddings)[0]

                top_results = torch.topk(cos_scores, k=min(top_k, len(self.df)))
                for score, idx in zip(top_results[0], top_results[1]):
                    i = int(idx)
                    row = self.df.iloc[i]
                    sim_val = float(score)
                    matches.append({
                        "claim_id": row.get("claim_id", f"FC-{i:04d}"),
                        "claim": row.get("claim", ""),
                        "verdict": row.get("verdict", "Unknown"),
                        "category": row.get("category", "General"),
                        "similarity_pct": round(max(sim_val, 0.0) * 100, 1),
                        "similarity_raw": round(sim_val, 3),
                        "explanation": row.get("explanation", ""),
                        "fact_checker": row.get("fact_checker", "Fact Checker"),
                        "reference_url": row.get("reference_url", "")
                    })
            except Exception as e:
                logger.error("Embedding similarity failed: %s. Using TF-IDF.", e)

        # 2. TF-IDF Fallback if embeddings were unavailable
        if not matches and hasattr(self, "tfidf_matrix") and self.tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.tfidf_vectorizer.transform([query_text])
            sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]
            for idx in top_indices:
                row = self.df.iloc[idx]
                sim_val = float(sims[idx])
                matches.append({
                    "claim_id": row.get("claim_id", f"FC-{idx:04d}"),
                    "claim": row.get("claim", ""),
                    "verdict": row.get("verdict", "Unknown"),
                    "category": row.get("category", "General"),
                    "similarity_pct": round(max(sim_val, 0.0) * 100, 1),
                    "similarity_raw": round(sim_val, 3),
                    "explanation": row.get("explanation", ""),
                    "fact_checker": row.get("fact_checker", "Fact Checker"),
                    "reference_url": row.get("reference_url", "")
                })

        # 3. Optional Online Google Fact Check API check
        if config.GOOGLE_FACTCHECK_API_KEY:
            try:
                online_matches = self._query_google_factcheck_api(query_text)
                if online_matches:
                    matches = (online_matches + matches)[:top_k]
            except Exception as e:
                logger.debug("Google Fact Check API query skipped: %s", e)

        # 4. Calculate Contradiction Score (0.0 to 1.0)
        # If input has high similarity to a debunked claim (Verdict=False), contradiction is high.
        # If input has high similarity to a confirmed claim (Verdict=True), contradiction is 0.
        top_match = matches[0] if matches else None
        contradiction_score = 0.0
        summary = "No verified claims strongly match this text."

        if top_match:
            sim = top_match["similarity_raw"]
            verdict = top_match["verdict"].strip().lower()

            if sim >= config.FACTCHECK_SIMILARITY_THRESHOLD:
                if verdict == "false":
                    # Input closely matches a debunked falsehood -> high contradiction penalty
                    contradiction_score = min(0.60 + (sim - config.FACTCHECK_SIMILARITY_THRESHOLD) * 1.0, 1.0)
                    summary = (
                        f"Contradicts verified fact-check: matches a debunked claim "
                        f"('{top_match['claim']}') evaluated as FALSE by {top_match['fact_checker']} "
                        f"({top_match['similarity_pct']}% match)."
                    )
                elif verdict == "misleading":
                    contradiction_score = min(0.35 + (sim - config.FACTCHECK_SIMILARITY_THRESHOLD) * 0.7, 0.70)
                    summary = (
                        f"Partially matches claim evaluated as MISLEADING by {top_match['fact_checker']} "
                        f"({top_match['similarity_pct']}% match)."
                    )
                elif verdict == "true":
                    contradiction_score = 0.0
                    summary = (
                        f"Corroborates verified fact-check: matches fact evaluated as TRUE by "
                        f"{top_match['fact_checker']} ({top_match['similarity_pct']}% match)."
                    )
            else:
                contradiction_score = max(sim * 0.15, 0.0)
                summary = f"Weak match to known fact-checks (best match: {top_match['similarity_pct']}%). No direct contradiction."

        return {
            "matches": matches,
            "top_match": top_match,
            "contradiction_score": round(contradiction_score, 3),
            "summary": summary
        }

    def _query_google_factcheck_api(self, query: str) -> List[Dict[str, Any]]:
        """Optional query to Google Fact Check Tools API."""
        import requests
        api_key = config.GOOGLE_FACTCHECK_API_KEY
        if not api_key:
            return []

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": query[:120], "key": api_key, "languageCode": "en"}
        resp = requests.get(url, params=params, timeout=3.0)
        if resp.status_code != 200:
            return []

        data = resp.json()
        claims = data.get("claims", [])
        results = []
        for c in claims[:2]:
            claim_text = c.get("text", "")
            review = (c.get("claimReview") or [{}])[0]
            publisher = review.get("publisher", {}).get("name", "Google Fact Check")
            rating = review.get("textualRating", "Unrated")
            url = review.get("url", "")
            results.append({
                "claim_id": "API-GFC",
                "claim": claim_text,
                "verdict": rating,
                "category": "Google Fact Check API",
                "similarity_pct": 88.0,
                "similarity_raw": 0.88,
                "explanation": f"Google Fact Check API result from {publisher}: {rating}",
                "fact_checker": publisher,
                "reference_url": url
            })
        return results


# Global singleton instance
_verifier_instance = None

def get_factcheck_verifier() -> FactCheckVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = FactCheckVerifier()
    return _verifier_instance


def verify_factcheck(query_text: str, top_k: int = 3) -> Dict[str, Any]:
    """Exposed module function for verifying text against factcheck database."""
    return get_factcheck_verifier().verify(query_text, top_k=top_k)
