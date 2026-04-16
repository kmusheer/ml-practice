
# 🚀 Machine Learning Practice — From Scratch to Scikit-Learn

End-to-end Machine Learning implementations covering multiple algorithms, from mathematical foundations (NumPy) to production-style pipelines (scikit-learn).

> ⚡ This project demonstrates both **deep algorithm understanding (from scratch)** and **practical ML engineering (pipelines, evaluation, and comparison)**.

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Practice-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)


---

## 🔍 What This Project Covers

* ✅ ML algorithms implemented from scratch (NumPy)
* ✅ End-to-end pipelines using scikit-learn
* ✅ Classification, Regression, and Clustering workflows
* ✅ Model comparison across multiple algorithms
* ✅ ROC-AUC, regression metrics, and clustering evaluation
* ✅ Cross-validation to ensure model stability and avoid overfitting

---

## 📂 Project Structure

```
ml-practice/
│
├── assets/  
│
├── data/                  # Datasets (Titanic, Housing, Iris, etc.)
│
├── scratch/               # From-scratch implementations (NumPy)
│   ├── linear_regression_numpy.py
│   ├── logistic_regression_numpy.py
│   ├── knn_numpy.py
│   ├── kmeans_numpy.py
│   ├── pca_numpy.py
│   └── README.md
│
├── sklearn_pipeline/      # Practical ML pipelines (scikit-learn)
│   ├── titanic_pipeline.py       # Classification
│   ├── house_prices.py           # Regression
│   ├── iris_models.py            # Multi-class classification
│   ├── clustering.py             # Unsupervised learning
│   └── README.md
│
├── utils/                 # Shared utilities
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── README.md
│
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

---

## 🧠 Algorithms Covered

### 🔹 Supervised Learning

* Logistic Regression (classification)
* Linear Regression (regression)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest

---

### 🔹 Unsupervised Learning

* K-Means Clustering
* Principal Component Analysis (PCA)

---

## 📊 Example: Classification (Titanic Dataset)

| Model               | Accuracy | Precision | Recall | F1 Score | AUC    |
| ------------------- | -------- | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.7982   | 0.7619    | 0.7191 | 0.7399   | 0.8751 |
| KNN                 | 0.8251   | 0.7976    | 0.7528 | 0.7746   | 0.8701 |
| Decision Tree       | 0.8117   | 0.7831    | 0.7303 | 0.7558   | 0.8280 |
| Random Forest       | 0.8206   | 0.8000    | 0.7191 | 0.7574   | 0.8698 |

---

## 📈 ROC Curve

---

## ⚙️ Workflow

The machine learning pipeline follows these steps:

1. Data Loading
2. Preprocessing (missing values, encoding, feature engineering)
3. Train-Test Split
4. Feature Scaling
5. Model Training (multiple algorithms)
6. Evaluation (classification / regression / clustering metrics)
7. Model Comparison
8. Cross-Validation
9. Model Selection

---

## ▶️ How to Run

```bash
git clone https://github.com/kmusheer/ml-practice.git
cd ml-practice
pip install -r requirements.txt
python main.py
```

---

## 🧠 Key Learnings

* Implemented ML algorithms from scratch using NumPy
* Built reusable ML pipelines using scikit-learn
* Compared multiple models across different tasks
* Understood importance of feature scaling and preprocessing
* Applied proper evaluation metrics for different problem types
* Used cross-validation for robust model evaluation

---

## 📌 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Feature selection techniques
* Model deployment (Flask / FastAPI)
* Advanced models (Boosting, XGBoost, etc.)

---
