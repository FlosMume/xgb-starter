# XGBoost Starter (GPU‑Ready with Conda + WSL2)

> **Purpose:** A minimal, beginner‑friendly machine learning project demonstrating GPU‑accelerated gradient boosting with **XGBoost**, including **early stopping (ES)** and **5‑fold cross‑validation (CV)** training options. Designed to run on **Windows 11 + WSL2 + RTX 4070 SUPER** or any CUDA‑capable GPU.

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
# clone the repo
git clone https://github.com/FlosMume/xgb-starter.git
cd xgb-starter

# create and activate the conda environment
conda env create -f env.yml
conda activate xgb-starter

# verify GPU
nvidia-smi
```

If XGBoost is missing:
```bash
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost==3.1.1
```

---

### 2️⃣ Train the Model

**Early Stopping (default)**  
Trains a model with a train/validation split, reports AUC + F1, and saves `xgb_model.joblib`:

```bash
python train.py
```

**Cross‑Validation**  
5‑fold CV (AUC + F1) and retrains on full data:

```bash
python train.py --mode cv
```

---

### 3️⃣ Predict on New Data

Provide a CSV with identical feature columns to the training data:

```bash
python predict.py your_input.csv
```

Output → `predictions.csv`:

| prediction | proba |
|-------------|--------|
| 1 | 0.973 |
| 0 | 0.114 |

Optional custom threshold:

```bash
python predict.py your_input.csv --threshold 0.35
```

---

## 📁 Project Structure

```
xgb-starter/
├── train.py          # training script (ES + CV modes, GPU support)
├── predict.py        # prediction script with rich comments
├── env.yml           # Conda environment definition
├── requirements.txt  # optional pip requirements
└── README.md         # this guide
```

---

## ⚙️ Environment Details

| Component | Example Version |
|------------|----------------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| GPU | NVIDIA RTX 4070 SUPER |
| CUDA | 12.8 (runtime toolkit) |
| Python | 3.11 |
| XGBoost | 3.1.1 (GPU) |
| Scikit‑Learn | 1.4 + |
| Pandas | 2.2 + |

---

## 💡 Notes for Beginners

- `train.py` is heavily commented—open it to learn how early stopping and cross‑validation work.  
- `predict.py` shows safe CSV handling and schema validation for deployment‑ready prediction workflows.  
- For hyperparameter tuning, add [`Optuna`](https://optuna.org/) or `sklearn.model_selection.GridSearchCV` later.  
- All code runs identically on CPU—simply change `device="cuda"` to `"cpu"`.

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)
- [Scikit‑Learn API Guide](https://scikit-learn.org/stable/documentation.html)

---

© 2025 FlosMume. MIT License.
