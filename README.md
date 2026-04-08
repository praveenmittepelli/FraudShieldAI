---
title: FraudShieldAI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🛡️ FraudShieldAI

FraudShieldAI is an end-to-end Machine Learning pipeline and web application for real-time credit card fraud detection. It uses a hybrid ensemble of Classical Machine Learning models (Logistic Regression, Random Forest, XGBoost) alongside Deep Learning networks (FNN, LSTM) to combat extreme class imbalance and identify complex fraudulent sequential and static data patterns.

## 🚀 Local Installation & Usage

This project utilizes [uv](https://github.com/astral-sh/uv) as a lightning-fast Python package manager.

### Prerequisites
- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv` on macOS)

### 1. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/praveenmittepelli/FraudShieldAI.git
cd FraudShieldAI

# Install dependencies and sync virtual environment
uv sync
```

### 2. Prepare Data
Ensure you have the training dataset `fraudTrain.csv` placed inside the `src/data/` directory.
Download the transactional dataset (e.g., Kaggle's Credit Card Fraud Detection Dataset - https://www.kaggle.com/datasets/kartik2112/fraud-detection) to simulate real-world financial scenarios.

### 3. Run the Training Pipeline
Train all supervised and deep learning models, evaluate them, and generate metrics/plots:
```bash
uv run python src/pipeline.py
```
*This will output the compiled binaries to `src/saved_models/` and performance benchmarks to `src/plots/`.*

### 4. Launch the Web Interface
Start the interactive Gradio dashboard to make live predictions:
```bash
cd src
uv run python app.py
```
Navigate your browser to `http://0.0.0.0:7860/`.

---

## ☁️ Deploying to a Hugging Face Private Space

Deploying your Gradio app to Hugging Face Spaces securely allows you to host the interface without exposing it, your models, or your data to the public. 

### Prerequisites
- Git LFS installed (`brew install git-lfs`)
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv` on macOS)

### Step 1: Export Dependencies

Hugging Face defaults to reading a standard `requirements.txt` file when spinning up the environment container. Export your `uv` environment so Hugging Face can install the correct packages:
```bash
use below script to conver pyproject.toml to requirements.txt
uv pip compile pyproject.toml -o requirements.txt
# Alternatively: uv export --no-hashes > requirements.txt
```
*(Make sure this `requirements.txt` is moved to the root of your Hugging Face pushing directory).*

### Step 2: Create a Private Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click your profile icon -> **New Space**.
3. Name your space (e.g., `FraudShieldAI`).
4. Select **Gradio** as the Space SDK.
5. Under `Space hardware`, choose your desired compute (CPU Basic is free).
6. **IMPORTANT**: Under `Space Visibility`, ensure you choose **Private or Public**.
7. Click **Create Space**.

### Step 3: Setup Git LFS for Large Models
Since your models (`.keras`, `.joblib`, `.pkl`) and data are massive, you **must** track them with Git Large File Storage (`git-lfs`) before pushing.
```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.joblib"
git lfs track "*.pkl"
git lfs track "*.csv"
git add .gitattributes
```

### Step 4: Directory Structure Preparation
Step4 is optional if you are deploying to Hugging Face Space because already app.py is at the root of the repository.

When Hugging Face spins up, it automatically executes the `app.py` script located at the **ROOT** of the repository. Move your contents outside of `src/` to the root, or duplicate your `src/app.py` out to the root layer so Hugging Face can find it. 

Your root Hugging Face directory should look like this:
```
/
├── app.py
├── requirements.txt
├── feature_engineering.py  (and other auxiliary codes your app.py imports)
├── saved_models/
│   ├── sup_xgboost.joblib
│   └── dl_lstm.keras...
```

### Step 5: Push the Project
Now you just need to `git push` to your Space repository.
```bash
# Add Hugging Face as a remote origin
git remote add huggingface https://huggingface.co/spaces/<your-username>/FraudShieldAI

ex: git remote add huggingface https://huggingface.co/spaces/praveenmittepelli/FraudShieldAI

# Stage and Commit
git add .
git commit -m "Initial deploy to HF Space"

# Push
git push huggingface main
```

*(Note: If Git asks for credentials when pushing, treat your username as normal but use a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `write` privileges as your password).*

    1. Go to your Hugging Face Account Settings -> Access Tokens and create a new token with Write access.
    
    Run the following command, replacing <YOUR_HF_TOKEN> with the token you just created:
    git push https://<YOUR_HF_TOKEN>@huggingface.co/spaces/<your-username>/FraudShieldAI main
    
### Step 6: Verification
1. Go to your Hugging Face Space repository.
2. At the top of the UI, an indicator will say "**Building**". This step installs everything in your `requirements.txt`.
3. Once it says "**Running**", your private Gradio app is live! Because you set visibility to Private, only you (and teammates in your HF Organization) can view or interact with it.

My Hugging space link: https://huggingface.co/spaces/mittepellip34/FraudShieldAI