# 🛡️ TruthLens: Fake News Detection Using Explainable AI
### Credibility-Aware, Multimodal Misinformation Audit Framework for College Presentations

TruthLens is a complete, demo-ready fake news detection system that pairs deep transformer contextual representations with knowledge-graph source credibility, semantic fact-checking, and Explainable AI (LIME).

---

## 🚀 Key Highlights & Architectural Features

- **Runs 100% Offline**: Works on standard laptop CPUs without needing expensive GPU cloud servers.
- **7-Stage Architectural Pipeline**:
  `Raw Article ➡️ Text Preprocessing ➡️ DistilBERT Transformer ➡️ Source Credibility Graph ➡️ Fact-Check Retrieval ➡️ Explainable AI (LIME) ➡️ Multimodal Risk Score (0-100)`
- **Explainable AI (LIME)**: Highlights exact words pushing predictions toward FAKE (red) and REAL (green/teal) with plain-English reasoning.
- **Source Credibility Graph**: NetworkX PageRank propagation across ~60 domains with interactive visualizers.
- **Fact-Check Corroboration**: Fast vector search over 170+ verified claims with disk caching.
- **Risk Score Breakdown**: Plotly gauge (Low / Medium / High) with stacked bar component analysis.

---

## 📁 Project Structure

```
truthlens/
├── README.md               # Complete setup, usage, and demo guide
├── VIVA_NOTES.md           # Master viva preparation guide with 15 examiner Q&As
├── requirements.txt        # Pinned, tested dependencies
├── config.py               # Central paths, hyperparameters, and risk weights
├── app.py                  # Dark Navy & Teal interactive Streamlit dashboard
├── run_all.bat             # Automated Windows setup & launcher
├── run_all.sh              # Automated Unix/macOS launcher
├── data/
│   ├── prepare_data.py     # Downloads HF datasets or generates 1,400 synthetic samples
│   ├── generate_factchecks.py # Generates 170+ verified fact-checks
│   ├── source_credibility.csv # Curated index of 60 media sources
│   ├── source_graph.json   # Citation & syndication network for PageRank
│   └── factchecks.csv      # Indexed verified claims
├── src/
│   ├── preprocess.py       # BERT/Baseline text cleaners & style signal extractors
│   ├── baselines.py        # Naive Bayes, Linear SVM, Random Forest, Bi-LSTM
│   ├── bert_model.py       # DistilBERT fine-tuning loop and batch predictors
│   ├── credibility.py      # Domain extraction, PageRank propagation, PyVis/Plotly
│   ├── factcheck.py        # SentenceTransformer embeddings + Cosine similarity
│   ├── explain.py          # LIME explainer & red/green HTML renderer
│   ├── risk.py             # Multimodal risk formula & Plotly gauge builder
│   ├── evaluate.py         # Benchmark suite (Accuracy, Precision, Recall, F1, FPR)
│   └── pipeline.py         # Unified analyze(text, source) single entrypoint
├── models/                 # Saved model weights
└── results/                # Evaluation benchmark CSVs, confusion matrices, charts
```

---

## ⚡ Quickstart Guide (Step-by-Step)

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows, macOS, or Linux

### 1. Automated One-Click Launch
On Windows, simply run:
```cmd
run_all.bat
```
On macOS / Linux:
```bash
chmod +x run_all.sh
./run_all.sh
```

---

### 2. Manual Step-by-Step Execution

#### Step A: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step B: Prepare Datasets & Fact-Checks
```bash
python data/generate_factchecks.py
python data/prepare_data.py
```

#### Step C: Train Classical Baseline Models
```bash
python -c "import pandas as pd, config; from src.baselines import get_baseline_manager; df=pd.read_csv(config.TRAIN_DATA_PATH); get_baseline_manager().train_all(df, quick=True)"
```

#### Step D: Fine-Tune DistilBERT Transformer (Quick Demo Mode)
```bash
# Takes ~2 minutes on a normal laptop CPU:
python src/bert_model.py --quick

# (Optional: for full 2-epoch training, run without --quick):
# python src/bert_model.py
```

#### Step E: Run Model Evaluation Benchmark
```bash
python src/evaluate.py
```
This will generate:
- `results/results.csv`
- `results/model_comparison.png`
- `results/bert_confusion_matrix.png`
- `results/evaluation_summary.md`

#### Step F: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🎯 5 Recommended Live Demo Inputs for Your Presentation

Use these 5 inputs in your live demo to showcase the full power of TruthLens:

### 1. Fake News (Health Conspiracy)
- **Input Text**:
  > *"SHOCKING: Doctors BANNED this miracle herbal tea because it cures all stage 4 cancer in 48 HOURS! Big Pharma executives are panicking and desperately trying to scrub this secret video off the internet. Just boiling two kitchen spices dissolves tumors completely with zero side effects! Share this before it gets deleted!"*
- **Source**: `naturalnews.com`
- **What it shows**: Flagged as **FAKE (High Risk, 85-95/100)**. Words like *"SHOCKING"*, *"BANNED"*, *"miracle"*, *"cancer"* are highlighted in vivid red. Source credibility flagged as Unreliable (10/100). Contradicts medical fact-checks.

### 2. Fake News (Viral Indian WhatsApp Hoax)
- **Input Text**:
  > *"UNESCO OFFICIAL DECLARATION: Indian National Anthem has just been awarded the Best Anthem in the Universe! All 193 country presidents gave a standing ovation in Paris assembly yesterday! Every proud citizen must forward this patriotic achievement to at least 10 groups right now!"*
- **Source**: `postcard.news`
- **What it shows**: Flagged as **FAKE (High Risk, 80-90/100)**. Fact-check module finds a direct match with UNESCO anthem hoax debunked by BOOM Live. High ALL-CAPS style penalty.

### 3. Real News (Official Public Health Advisory)
- **Input Text**:
  > *"World Health Organization releases updated clinical management recommendations for viral seasonal influenza. Surveillance data collected across 48 reference laboratories indicates predominant circulating sub-strains remain susceptible to standard antiviral regimens. Public health authorities urge annual vaccination for geriatric populations."*
- **Source**: `who.int`
- **What it shows**: Flagged as **REAL (Low Risk, 8-15/100)**. Objective, measured words highlighted in teal. Source credibility score is 98/100 (Trusted).

### 4. Real News (Space Exploration Achievement)
- **Input Text**:
  > *"ISRO successfully landed the Chandrayaan-3 lander Vikram near the south pole of the Moon. On August 23, 2023, India became the first space agency in history to achieve a controlled soft touchdown in the lunar south polar region. Telemetry instruments deployed the Pragyan rover to conduct surface mineralogy spectrometry."*
- **Source**: `thehindu.com`
- **What it shows**: Flagged as **REAL (Low Risk, 5-12/100)**. Corroborates verified factual space database. High source trust rating (92/100).

### 5. Tricky / Borderline Case (Sensational Style from Reputable Source)
- **Input Text**:
  > *"BREAKING: Central Bank governor delivers urgent warning regarding volatile global currency markets and inflation pressures during emergency summit."*
- **Source**: `reuters.com`
- **What it shows**: Flagged as **Medium Risk (35-50/100)**. Illustrates multimodal balance: even though words like *"BREAKING"* and *"urgent"* push toward sensationalism, the high source credibility of Reuters (96/100) and absence of fact-check contradictions prevent a false alarm!

---

## 🛠️ Troubleshooting

1. **Model taking too long to train?**
   Make sure you used the `--quick` flag: `python src/bert_model.py --quick`. This subsamples the training set to ~400 items and runs 1 epoch in ~2 minutes.
2. **Streamlit port conflict?**
   If port 8501 is occupied, launch on another port:
   `streamlit run app.py --server.port 8502`
3. **Running fully offline?**
   Once models and dependencies are downloaded once, TruthLens runs completely locally. All claims, embeddings, graphs, and transformers are cached on your hard drive.
