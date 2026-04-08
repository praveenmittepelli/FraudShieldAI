# FraudShield AI: Enterprise System Documentation

## Overview
FraudShield AI is an enterprise-grade, hybrid ensemble fraud detection pipeline. It detects fraudulent credit transactions by unifying classical tree-based machine learning with sequential deep-learning topologies, ensuring high recall on extremely imbalanced financial data streams.

---

## 1. Step-by-Step Setup and Execution

### Local Environment Setup
1. **Clone & Install Dependencies:** Ensure Python and `uv` (or pip) are installed.
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install pandas numpy scikit-learn xgboost tensorflow seaborn matplotlib gradio joblib
   ```
2. **Execute the Training Pipeline:**
   To train models, generate sequences, and output evaluation charts, run the core orchestrator:
   ```bash
   uv run python src/pipeline.py
   ```
   *This saves computed `.pkl` / `.keras` state dictionaries to `/saved_models/` and charts directly into `/plots/`.*
3. **Launch Web Interface:**
   Once trained, initiate the interactive Gradio dashboard to upload inference CSVs and interrogate the analyst chatbot.
   ```bash
   uv run python app.py
   ```
   *Access the web UI at http://0.0.0.0:7860.*

### Cloud Deployment Strategy (e.g. Hugging Face Spaces / AWS EC2)
1. Export the active environment:
   ```bash
   uv pip freeze > requirements.txt
   ```
2. Provision a cloud instance (or Space) running a container with Python 3.9+.
3. During the CI/CD pull, ensure the host fetches `/saved_models/` safely (e.g., via DVC, Git LFS, or S3).
4. Expose TCP port **7860**. Execute `python app.py` as the system process command. 

---

## 2. System Architecture & Data Flow

### The Data Flow
1. **Data Ingestion:** Large tabular sets (e.g., `fraudTrain.csv`) are intercepted.
2. **Feature Engineering** (`FeatureEngineer`): Creates complex derivative signals like dynamic "customer velocity" (trailing counts) and geospatial markers via Haversine distance tracking to predict account takeovers.
3. **Temporal Mapping** (`create_sequences`): Tabular rows are grouped independently by `cc_num` (avoiding data leakage). The sliding window logic constructs contiguous memory blocks (size $N=5$) enabling predictive sequence classification.
4. **Ensemble Execution:** Processed features feed into independent deep-learning components and standard threshold ML algorithms simultaneously.
5. **Score Aggregation** (`HybridEnsemble`): Produces weighted uniform arrays mapping a terminal probability `[0.0 - 1.0]` across divergent sequences and formats.

### Deployed Components
* **Offline Pipeline (`pipeline.py`)**: Batches computation entirely headless for automated cron scheduling or massive multi-GPU processing arrays.
* **Gradio Web application (`app.py`)**:
    * **Bulk Tab:** Provides CSV Upload tools to map unverified transactions returning a visual table of rows crossing a 50% threat threshold.
    * **Chatbot Tab:** Rule-based generative module answering analyst inquiries on algorithmic functions, feature thresholds, and model states.

---

## 3. Models and Hyperparameters

To accurately score data lacking positive variance (0.5% fraud rate), the architecture leverages divergent topologies:

### A. Classical Supervised Topologies
* **Logistic Regression:**
  * **Config:** `max_iter=1000`, `class_weight='balanced'`.
  * **Role:** Linear separator ensuring rapid classification of heavily distant outliers. Weight impact: **10%**.
* **Random Forest Classifier:**
  * **Config:** `n_estimators=100`, `max_depth=10`, `n_jobs=-1`, `class_weight='balanced'`.
  * **Role:** Overfitting resistant bagging topology. Weight impact: **20%**.
* **XGBoost Classifier:**
  * **Config:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `scale_pos_weight=99` *(to compensate 99:1 split)*.
  * **Role:** Primary gradient-boosted tabular arbiter. Weight impact: **30%**.

### B. Deep Learning Sequences
* **Fully Connected Feed-Forward Network (FNN):**
  * **Config:** 3 Epochs, `batch_size=256`, `pos_weight=10.0`.
  * **Role:** Evaluates global complex non-linear combinations across tabular outputs. Weight impact: **20%**.
* **Long Short-Term Memory Network (LSTM):**
  * **Config:** 3 Epochs, `batch_size=256`, `pos_weight=10.0`, Sequence Length = 5.
  * **Role:** Retains structural memory of individual `cc_num` activity over short chronological horizons to identify compromised burst purchases. Weight impact: **20%**.

---

## 4. Evaluation Metrics & Threat Scoring

Traditional evaluation methodologies heavily fail under imbalanced sets (predicting 0 continually results in 99.5% accuracy). This system evaluates success natively through rigorous anomaly metrics implemented in `evaluation.py`:

* **Precision-Recall Area Under Curve (PR-AUC):** The primary benchmark tracking the exact trade-off threshold of true positive catch rates relative to false-positive nuisance flags.
* **F1-Score / Threshold Bounds:** Harmonic mean of prediction accuracy. Determines final hard cutoff limits for UI mapping (`predict()` > 0.5 binary triggers).
* **Confusion Matrices:** Visualizes classification leakage natively mapping True Positives / Negatives via a serialized seaborn matrix output (`/plots/cm_{model}.png`).
