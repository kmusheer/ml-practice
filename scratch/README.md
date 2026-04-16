# 🧠 Scratch Implementations (NumPy)

This folder contains Machine Learning algorithms implemented **from scratch using NumPy**, without relying on ML libraries.

---

## 📌 Purpose

* Understand how ML algorithms work internally
* Implement core mathematics (not just use APIs)
* Build strong intuition for optimization and model behavior

---

## ⚙️ Implemented Algorithms

### 🔹 Linear Regression (`linear_regression_numpy.py`)

* Normal Equation (closed-form solution)
* Handling singular matrices (pseudoinverse)
* Evaluation metrics:

  * MSE, MAE, R²
* Manual prediction and residual analysis

---

### 🔹 Logistic Regression (`logistic_regression_numpy.py`)

* Sigmoid (logistic) function
* Binary Cross-Entropy loss
* Gradient Descent optimization
* Classification metrics (Accuracy, Precision, Recall, F1)
* ROC Curve and AUC computation from scratch
* Threshold tuning (Youden Index)

👉 See implementation: 

---

### 🔹 K-Nearest Neighbors (`knn_numpy.py`)

* Distance-based classification (Euclidean)
* No training phase (lazy learning)
* Majority voting

---

### 🔹 K-Means Clustering (`kmeans_numpy.py`)

* Random centroid initialization
* Iterative cluster assignment
* Convergence based on centroid movement

---

### 🔹 Principal Component Analysis (`pca_numpy.py`)

* Covariance matrix computation
* Eigen decomposition
* Dimensionality reduction

---

## 🧩 Key Concepts Covered

* Gradient Descent
* Cost Functions (MSE, Binary Cross-Entropy)
* Linear Algebra (matrix operations, eigenvalues)
* Optimization techniques
* Distance metrics

---

## ▶️ How to Run

```bash
python main.py
```

Then in `main.py`:

```python
mode = "scratch"
```

---

## 🧠 Why This Matters

Most ML projects only use libraries.

This section shows:

* You understand **how models actually work**
* You can implement algorithms without abstraction
* You can debug and reason about model behavior
  
---
### 📌 add gradient descent version to linear regression
