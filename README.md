# XGBoost Starter (GPU-Ready + Auto GPU/CPU Inference)

> **Purpose:**  
> A reproducible, beginner-friendly machine learning project demonstrating **GPU-accelerated gradient boosting with XGBoost**, featuring **Early Stopping (ES)**, **5-Fold Cross-Validation (CV)**, and **automatic GPU/CPU detection** during both training and inference.  
> Designed for **Windows 11 + WSL 2 + RTX 4070 SUPER**, but runs on any CUDA-capable or CPU-only system.

---

## 🚀 Quick Start

### 1️⃣ Clone and Create Environment

```bash
git clone https://github.com/FlosMume/xgb-starter.git
cd xgb-starter

conda env create -f env.yml
conda activate xgb-starter
```

Verify GPU availability in WSL 2:
```bash
nvidia-smi
```

If XGBoost is missing:
```bash
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost==3.1.1
```

---

### 2️⃣ Train the Model

**Early Stopping (default)**  
Trains with a train/validation split, reports **AUC + F1**, and saves `xgb_model.joblib`:

```bash
python train.py
```

**Cross-Validation**  
Performs 5-fold CV (AUC + F1) and retrains on all data:

```bash
python train.py --mode cv
```

---

### 3️⃣ Predict on New Data (with Auto GPU/CPU)

`predict.py` automatically detects your environment:
- If model → `device="cuda"` **and** CuPy is installed → GPU inference  
- Otherwise → CPU inference (no warnings)

```bash
python predict.py your_data.csv
```

Example output:
```
✅ Wrote predictions.csv (threshold=0.50, inference=GPU)
```

Adjust threshold:
```bash
python predict.py your_data.csv --threshold 0.35
```

Output file → `predictions.csv`:
| prediction | proba |
|-------------|--------|
| 1 | 0.973 |
| 0 | 0.114 |

---

## 📁 Project Structure

```
xgb-starter/
├── train.py              # training script (ES + CV, GPU support)
├── predict.py            # prediction script (auto GPU/CPU inference)
├── xgb_starter_demo.ipynb# richly commented Jupyter demo notebook
├── env.yml               # Conda environment definition
├── requirements.txt      # optional pip requirements
└── README.md             # this guide
```

---

## ⚙️ Environment Details

| Component | Example Version |
|------------|----------------|
| OS | Windows 11 + WSL 2 (Ubuntu 22.04) |
| GPU | NVIDIA RTX 4070 SUPER |
| CUDA | 12.8 (runtime toolkit) |
| Python | 3.11 |
| XGBoost | 3.1.1 (GPU build) |
| CuPy | Optional (for GPU inference) |
| scikit-learn | 1.4 + |
| pandas | 2.2 + |
| matplotlib | 3.8 + (for notebook plots) |

---

## 🧠 Key Features

### 🏋️‍♂️ Training (`train.py`)
- Dual modes: **Early Stopping** and **5-Fold Cross-Validation**
- Reports **AUC** and **F1**
- GPU training (`device="cuda"`) or CPU fallback
- Saves `xgb_model.joblib` (bundle: model + feature names)
- Reproducible random seed control

### 🔍 Inference (`predict.py`)
- Loads model and validates schema
- Auto-detects GPU/CPU and sets predictor accordingly
- Converts data to **CuPy** arrays when using GPU
- Safe CSV I/O and clear warnings
- Outputs `predictions.csv` with probabilities and class labels

### 📓 Notebook (`xgb_starter_demo.ipynb`)
- Interactive, richly commented step-by-step demo
- Shows ROC Curve and Feature-Importance plots
- Uses identical workflow as scripts for reproducibility
- Ideal for GitHub visualization (GitHub renders outputs)

---

## ⚠️ Notes

- Do **not** push `.joblib` files to GitHub — binary artifacts are large and non-portable.  
  `.gitignore` already handles this.
- CuPy installation (optional for GPU prediction):  
  ```bash
  pip install cupy-cuda12x
  # or
  conda install -c conda-forge cupy
  ```
- To manually force CPU inference and silence warnings:
  ```python
  model.set_params(predictor="cpu_predictor")
  ```
- For hyperparameter tuning, extend with [`Optuna`](https://optuna.org/) or `sklearn.model_selection.GridSearchCV`.

---

## 💡 Tips for Beginners
- Open `train.py` to see commented examples of Early Stopping and Cross-Validation.  
- Open `predict.py` to learn how to handle schema mismatch and device selection.  
- The notebook visualizes the same processes in an interactive way.

---

## 📚 References
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)  
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)  
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)  
- [scikit-learn Metrics Guide](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

---

© 2025 FlosMume. MIT License.

---

📦 **Summary:**  
This project provides a complete GPU-ready XGBoost pipeline—from environment setup to training, evaluation, auto-inference, and visualization—ideal for learning, portfolio demos, and benchmarks.
