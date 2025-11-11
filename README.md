# XGBoost Starter (GPU/CPU Auto Inference)

> **Purpose:** Demonstrates a reproducible, beginner‑friendly workflow for GPU‑accelerated XGBoost classification with automatic GPU/CPU detection during both training and inference.

---

## 🚀 Quick Start

### 1️⃣ Clone and Create Environment

```bash
git clone https://github.com/FlosMume/xgb-starter.git
cd xgb-starter

conda env create -f env.yml
conda activate xgb-starter
```

Verify GPU access in WSL 2:

```bash
nvidia-smi
```

If XGBoost is missing:
```bash
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost==3.1.1
```

---

### 2️⃣ Train the Model

**Early Stopping (default):**

```bash
python train.py
```

**Cross‑Validation:**

```bash
python train.py --mode cv
```

Both create `xgb_model.joblib` — a serialized bundle with the model and feature names.

---

### 3️⃣ Predict with GPU/CPU Auto Detection

The new `predict.py` automatically decides:
- If model → `device="cuda"` **and** CuPy is available → GPU inference  
- Otherwise → CPU inference (no warnings)

```bash
python predict.py your_data.csv
```

Output example:

```
✅ Wrote predictions.csv (threshold=0.50, inference=GPU)
```

To adjust threshold:
```bash
python predict.py your_data.csv --threshold 0.35
```

---

## ⚙️ Environment Summary

| Component | Example Version |
|------------|----------------|
| Python | 3.11 |
| CUDA | 12.8 |
| XGBoost | 3.1.1 (GPU build) |
| CuPy | Optional (only for GPU inference) |
| Matplotlib | 3.8 + (for plots in notebook) |
| scikit‑learn | 1.4 + |
| pandas | 2.2 + |

---

## 🧠 Key Features

### Training (`train.py`)
- Two modes: **Early Stopping** and **5‑fold Cross‑Validation**
- Reports **AUC** and **F1**
- GPU training (`device="cuda"`) or CPU fallback
- Saves `xgb_model.joblib`

### Inference (`predict.py`)
- Loads model + feature names safely
- Checks for missing/extra columns
- Auto‑detects GPU and sets predictor accordingly
- Converts data to **CuPy** array when on GPU
- Writes `predictions.csv` with `prediction` and `proba` columns

### Notebook (`xgb_starter_demo.ipynb`)
- Richly commented example with ROC curve and Feature Importance plots
- Perfect for GitHub visualization

---

## ⚠️ Notes

- Do **not** push `.joblib` files to GitHub — they are binary artifacts.  
  Already ignored via `.gitignore`.
- CuPy installation (optional for GPU prediction):  
  ```bash
  pip install cupy-cuda12x
  # or conda install -c conda-forge cupy
  ```
- To silence device mismatch warnings manually: set  
  ```python
  model.set_params(predictor="cpu_predictor")
  ```
  in `predict.py`.

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)
- [scikit‑learn Metrics Guide](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

---

© 2025 FlosMume. MIT License.
