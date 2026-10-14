# 🎓 TruthLens: College Project Viva & Presentation Master Guide

This guide is written in clear, student-friendly language to help you confidently present your project to external examiners and answer viva questions.

---

## 1. Project Overview & Elevator Pitch

> **Elevator Pitch (30 seconds):**
> *"TruthLens is an explainable, credibility-aware fake news detection system. Unlike traditional black-box classifiers that only give a 'Fake' or 'Real' label without justification, TruthLens combines deep transformer language modeling (DistilBERT) with network-propagated source credibility, semantic fact-check verification, and Explainable AI (LIME). It highlights the exact misleading words in red, verifies claims against certified databases, and outputs a calibrated 0-100 Risk Score so everyday users and moderators can understand why a decision was made."*

---

## 2. System Architecture & Module Walkthrough

Our pipeline matches the 7-stage architectural diagram presented in our project slides:

```
[News Article / Post] 
          ⬇️
[1. Text Preprocessing & Style Extraction] (src/preprocess.py)
          ⬇️
[2. DistilBERT Transformer Model] (src/bert_model.py)
          ⬇️
[3. Source Credibility Graph] (src/credibility.py)
          ⬇️
[4. Semantic Fact-Check Verification] (src/factcheck.py)
          ⬇️
[5. Explainable AI Module - LIME] (src/explain.py)
          ⬇️
[6. Multimodal Risk Scoring Engine] (src/risk.py)
          ⬇️
[7. Final Verdict Badge + Risk Gauge + Explanatory Card] (app.py)
```

### Detailed Breakdown of Each Module:

### Module 1: Text Preprocessing & Style Analysis (`src/preprocess.py`)
- **What it does:** Cleans the input text while preserving punctuation, question marks, and capitalization (needed by transformers). For classical baselines, it converts to lowercase. It also extracts stylistic heuristics:
  - **ALL-CAPS Ratio:** Disinformation frequently uses capitalized words for emotional urgency.
  - **Exclamation Count:** Tabloid clickbait uses multiple exclamation marks.
  - **Clickbait Phrases:** Detects phrases like *"you won't believe"*, *"miracle cure"*, *"secret uncovered"*, *"doctors hate"*.
  - **Sensational Emotion Extremity:** Measures the density of extreme buzzwords (*"shocking"*, *"banned"*, *"apocalyptic"*).

### Module 2: Deep Transformer Classification (`src/bert_model.py`)
- **What it does:** Uses `distilbert-base-uncased` fine-tuned for sequence classification (binary: 0=Real, 1=Fake).
- **Why DistilBERT?** DistilBERT retains 97% of BERT's language understanding while being 40% smaller and 60% faster, allowing it to run smoothly on standard laptop CPUs without needing expensive GPU servers.

### Module 3: Source Credibility Network Graph (`src/credibility.py`)
- **What it does:** Extracts the domain from URLs or publication names and references a database of ~60 news sources categorized as *Trusted*, *Mixed*, *Satire*, or *Unreliable*.
- **How the Graph Works:** Sources and topical categories form nodes in a directed graph (`networkx`). Edges represent citation, syndication, or audience-sharing relationships. Credibility is calculated using **personalized PageRank propagation**: high-authority seeds (e.g., Reuters, AP, WHO) pass trust to connected nodes, while untrusted or unindexed sources receive appropriate scrutiny.

### Module 4: Semantic Fact-Check Verification (`src/factcheck.py`)
- **What it does:** Matches user claims against an indexed database of 170+ verified fact-checks across politics, health, science, tech, and viral Indian social media claims.
- **How Retrieval Works:** Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to convert claims into 384-dimensional dense semantic vectors. Cosine similarity compares the user query with the vector database.
- **Contradiction Scoring:** If the input closely matches a claim that certified checkers (Snopes, Alt News, BOOM Live, WHO) marked as **FALSE**, the contradiction score rises sharply, heavily penalizing the text in the risk formula.

### Module 5: Explainable AI with LIME (`src/explain.py`)
- **What it does:** Black-box AI models cannot be trusted blindly in media moderation. We use **LIME** (*Local Interpretable Model-agnostic Explanations*).
- **How it works:** LIME perturbs the input by removing random words and observing how the model's output probability shifts. Words whose removal causes the fake probability to plummet are deemed the primary drivers of the fake prediction.
- **Visual Output:** Red highlights indicate words pushing the prediction toward FAKE; green/teal highlights indicate words pushing toward REAL.

### Module 6: Multimodal Risk Scoring Engine (`src/risk.py`)
- **What it does:** Rather than relying solely on raw model probability (which can suffer from dataset bias), TruthLens computes a balanced Risk Score (0 to 100):
  $$\text{Risk Score} = 100 \times \left( 0.45 \cdot P_{\text{BERT}} + 0.25 \cdot \frac{100 - \text{Credibility}}{100} + 0.20 \cdot \text{Contradiction} + 0.10 \cdot \text{Style} \right)$$
- **Categorical Risk Tiers:**
  - **0 – 33 (Low Risk - Green):** Credible source, factual consistency, calm tone.
  - **34 – 66 (Medium Risk - Amber):** Unverified source, clickbait tone, or uncorroborated claim.
  - **67 – 100 (High Risk - Red):** Model flags fake, known debunked claim, or unreliable source.

---

## 3. Deep Dive: Key Technical Concepts

### Why BERT over Classical Models (SVM, Naive Bayes, Random Forest, LSTM)?
1. **Bag-of-Words Limitation in SVM & Naive Bayes:** Classical models count word frequencies. If a legitimate BBC article quotes a false claim: *"The politician falsely stated that drinking bleach cures infection"*, Naive Bayes sees words like *"bleach"* and *"infection"* and mistakenly flags the article as fake.
2. **Sequential Limitation in LSTMs:** LSTMs read tokens left-to-right. Over long paragraphs, earlier context degrades due to vanishing gradients, despite gating mechanisms.
3. **Bidirectional Multi-Head Attention in BERT:** BERT reads the entire sentence in both directions simultaneously. It understands that *"falsely stated"* negates the sensational claim that follows.

### Why F1-Score and False Positive Rate (FPR) Matter More Than Accuracy:
- **Accuracy Paradox:** If 90% of news on a platform is real and 10% is fake, a dumb model that predicts "Real" for everything gets 90% accuracy but catches zero misinformation!
- **F1-Score:** The harmonic mean of Precision and Recall. It ensures we catch fake news (high recall) without falsely accusing real journalists (high precision).
- **False Positive Rate (FPR):** $\text{FPR} = \frac{FP}{FP + TN}$. A high false-positive rate censors legitimate investigative journalism, destroying trust in fact-checking tools. Our evaluation table proves that DistilBERT achieves a significantly lower FPR than classical baselines.

---

## 4. Limitations and Future Scope

### Current Limitations:
1. **Satire Detection:** Parody sites like *The Onion* or *The Babylon Bee* intentionally mimic fake news rhetoric. Without source domain identification, pure NLP struggles to distinguish humor from malicious disinformation.
2. **Zero-Day Breaking Events:** If an event occurred 10 minutes ago, fact-checking databases have not yet indexed it. TruthLens must then rely on linguistic and source credibility signals.
3. **Language Scope:** Current fine-tuning is optimized for English text.

### Future Scope:
1. **Multimodal Deepfake Detection:** Expanding to analyze manipulated images and synthetic voice cloning using CLIP and MesoNet.
2. **Multilingual Regional Models:** Utilizing `XLM-RoBERTa` or `IndicBERT` to verify claims in Hindi, Tamil, Telugu, and other regional Indian languages.
3. **Browser Extension:** Creating a Chrome/Firefox extension that automatically checks social media feeds (Twitter/X, Facebook) in real time.
4. **Blockchain Provenance:** Storing cryptographic hashes of verified journalistic dispatches on an immutable ledger to guarantee origin authenticity.

---

## 5. Top 15 Likely Viva Questions & Model Answers

### Q1: What is the main objective of TruthLens?
**Answer:** The objective of TruthLens is to detect misinformation through an explainable, credibility-aware system that combines deep transformer text classification (DistilBERT), network-propagated source credibility, semantic fact-check verification, and LIME-based word contributions into a single interpretable Risk Score.

### Q2: Why did you choose DistilBERT instead of larger models like BERT-Large or GPT-4?
**Answer:** DistilBERT retains 97% of BERT's contextual language understanding while reducing parameter count by 40% and running 60% faster. This allows TruthLens to run completely offline on a standard laptop CPU without latency or GPU dependency, making it practical for real-time applications.

### Q3: How does LIME explain the model's prediction?
**Answer:** LIME (Local Interpretable Model-agnostic Explanations) creates perturbations of the input text by selectively omitting words and querying the model on these variants. It fits a simple, interpretable linear surrogate model locally around the prediction to quantify how much each specific word increased or decreased the probability of the news being fake.

### Q4: Why is a simple binary classification (Fake/Real) insufficient for real-world moderation?
**Answer:** Binary labels lack nuance and trust. A genuine article might report on a rumor, or an unverified rumor might appear on a trusted site. By providing a multimodal 0-100 Risk Score, highlighted words, and source credibility breakdown, human moderators can make informed decisions rather than trusting a black-box label.

### Q5: How does the credibility graph work?
**Answer:** We represent news domains and topics as nodes in a directed graph using NetworkX. Edges represent citation, syndication, and audience overlap. Using personalized PageRank starting from verified seed authorities (like Reuters, AP, WHO), credibility scores propagate through the network. Unknown sources receive a neutral default score.

### Q6: How does fact-check verification work without an SQL exact keyword match?
**Answer:** We use dense vector embeddings generated by `sentence-transformers` (`all-MiniLM-L6-v2`). The user input and verified claims are embedded into 384-dimensional vector space. Using cosine similarity, we find claims that have the same semantic meaning even if different words or sentence structures were used.

### Q7: What is the difference between Precision and Recall in this project?
**Answer:** 
- **Precision** is: of all articles we flagged as FAKE, how many were actually fake? High precision prevents falsely flagging genuine news.
- **Recall** is: of all actual fake news articles, how many did our system catch? High recall ensures dangerous hoaxes do not slip through undetected.

### Q8: What is False Positive Rate (FPR) and why did you measure it?
**Answer:** $\text{FPR} = \frac{FP}{FP + TN}$. It measures the proportion of genuine real news incorrectly flagged as fake. In our presentation, we noted that high FPR is a major flaw of existing shallow systems (like Naive Bayes). DistilBERT achieves a much lower FPR.

### Q9: What happens if an offline user inputs an unknown source?
**Answer:** The system identifies that the domain is unindexed, applies a neutral baseline credibility score of 50/100, flags it as "Unknown Source", and derives the risk score primarily from the text semantics, style cues, and fact-checking database.

### Q10: How do you handle clickbait style in the text?
**Answer:** In `src/preprocess.py`, we calculate an ALL-CAPS ratio, exclamation frequency, sensational vocabulary density, and match against a dictionary of known clickbait trigger phrases (e.g., *"you won't believe"*, *"miracle cure"*). This is fed as a 10% weighted component into the overall risk score.

### Q11: What loss function and optimizer did you use to fine-tune DistilBERT?
**Answer:** We used **Cross-Entropy Loss** for binary classification and the **AdamW optimizer** with a learning rate of $2 \times 10^{-5}$ and weight decay of 0.01 to prevent overfitting.

### Q12: How are embeddings cached for fact-checking?
**Answer:** On the first run, `sentence-transformers` encodes all 170+ verified claims into a PyTorch tensor and saves them to `.cache/factcheck_embeddings.pt`. On subsequent runs, TruthLens loads this tensor directly from disk, completing cosine similarity checks in under 10 milliseconds.

### Q13: Can TruthLens detect fake news in languages other than English?
**Answer:** Currently, TruthLens is trained on English datasets. To support multilingual news (such as Hindi or regional Indian languages), we would replace DistilBERT with `IndicBERT` or `XLM-RoBERTa` and index multilingual fact-checks from portals like Vishvas News and Boom Hindi.

### Q14: How does the system handle satire websites like The Onion?
**Answer:** Satirical articles often use sensational linguistic styles intentionally. TruthLens flags these publications in its source credibility database under the category **Satire** with a dedicated low-credibility/parody label so the user understands the piece is humorous rather than malicious propaganda.

### Q15: What is the single biggest advantage of your proposed system over existing systems?
**Answer:** Existing systems are isolated, unimodal, black-box classifiers. TruthLens is **multimodal, credibility-aware, and explainable**: it fuses deep linguistic transformers with knowledge-graph reputation and verified fact-checking databases, explaining every verdict visually and textually so users can verify the evidence for themselves.
