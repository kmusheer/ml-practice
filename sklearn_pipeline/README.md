# ⚙️ Scikit-Learn Pipelines

This folder contains end-to-end Machine Learning workflows built using **scikit-learn pipelines**, covering classification, regression, and clustering tasks.

---

## 📌 Purpose

* Build production-style ML pipelines
* Compare multiple models on real-world datasets
* Apply proper evaluation and validation techniques

---

## 🔍 Features

### 🔹 Data Processing

* Missing value handling
* Categorical encoding (label + one-hot)
* Feature engineering

---

### 🔹 Model Training

#### Classification

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

#### Regression

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

#### Clustering

* K-Means

---

### 🔹 Evaluation

* Classification metrics:

  * Accuracy, Precision, Recall, F1 Score
  * ROC Curve and AUC

* Regression metrics:

  * MSE, MAE, R²

---

### 🔹 Validation

* Cross-validation (5-fold)
* Model stability analysis using standard deviation

---

## 📊 Datasets Used

* Titanic → Binary classification
* Iris → Multi-class classification
* House Prices → Regression
* Mall Customers → Clustering

---

## ▶️ How to Run

```bash
python main.py
```

Then in `main.py`:

```python
mode = "sklearn"
```

---

## 🧠 Key Takeaways

* Pipelines ensure **consistent preprocessing + training**
* Proper scaling is applied only where required
* Cross-validation prevents overfitting and measures stability
* Model selection is based on both **performance and consistency**
