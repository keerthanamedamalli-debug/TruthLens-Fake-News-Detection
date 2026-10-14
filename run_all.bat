@echo off
TITLE TruthLens - Fake News Detection AI Pipeline
echo =====================================================================
echo       TruthLens: Fake News Detection Using Explainable AI
echo          College Project Complete Pipeline Execution
echo =====================================================================
echo.

cd /d "%~dp0"

echo [Step 1/5] Checking and Installing Dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Dependency installation encountered warnings or errors. Continuing with existing packages...
)

echo.
echo [Step 2/5] Preparing Dataset Splits and Verified Claims...
python data\generate_factchecks.py
python data\prepare_data.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Data preparation failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Step 3/5] Training Classical Baseline Models (Naive Bayes, SVM, RF, LSTM)...
python -c "import pandas as pd, config; from src.baselines import get_baseline_manager; df=pd.read_csv(config.TRAIN_DATA_PATH); get_baseline_manager().train_all(df, quick=True)"

echo.
echo [Step 4/5] Fine-Tuning DistilBERT Transformer (Quick Demo Mode)...
python src\bert_model.py --quick

echo.
echo [Step 5/5] Running Comprehensive Model Evaluation Benchmark...
python src\evaluate.py

echo.
echo =====================================================================
echo [SUCCESS] TruthLens pipeline training and evaluation complete!
echo Launching Interactive Streamlit Dashboard...
echo =====================================================================
streamlit run app.py
pause
