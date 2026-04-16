# 🛠️ Utilities

Reusable modules for preprocessing, evaluation, and visualization across ML pipelines.

---

## 📦 Modules

### `preprocessing.py`

* Handle missing values (mean/mode imputation)
* Encode categorical variables (label + one-hot encoding)
* Feature engineering for:

  * Titanic dataset (FamilySize, IsAlone)
  * Housing dataset (binary mapping + one-hot encoding)

---

### `evaluation.py`

* Classification metrics:

  * Accuracy, Precision, Recall, F1 Score
* Confusion Matrix
* ROC-AUC computation
* Returns structured metrics for model comparison

---

### `visualization.py`

* ROC Curve plotting
* Residual plots (regression)
* Actual vs Predicted visualization

---

## 🎯 Purpose

* Centralize reusable logic
* Keep pipelines clean and modular
* Ensure consistency across different ML tasks
