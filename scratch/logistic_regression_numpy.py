import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import (
    fill_missing,
    encode_features,
    feature_engineering
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def logistic(x):
    """Sigmoid function: converts linear output to probability"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Log loss
def log_loss(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = - (y * np.log(y_hat)) - ((1-y) * np.log(1-y_hat))
    return loss

def cost_func(y, y_hat):
    """Binary Cross-Entropy Loss"""
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    # return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return np.mean(log_loss(y, y_hat))

def standardize(X):
    """
    Standardize features (Z-score normalization)

    Formula:
        X_scaled = (X - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # avoid division by zero
    std = np.where(std == 0, 1,std)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

# Function to compute gradients of the cost function with respect to model parameters - using vectorization
def grad_logreg_vec(X, y, w, b):
    """Gradient of loss w.r.t weights and bias"""
    m = X.shape[0]
    y_hat = logistic(np.dot(X, w) + b)
    grad_w = np.dot(X.T, (y_hat - y)) / m
    grad_b = np.mean(y_hat - y)
    return grad_w, grad_b

# Gradient descent algorithm for logistic regression
def grad_desc(X, y, w, b, alpha, n_iter):
    """Train logistic regression using gradient descent"""
    cost_history = []

    for i in range(n_iter):
        y_hat = logistic(np.dot(X, w) + b)

        cost = cost_func(y, y_hat)
        cost_history.append(cost)

        #  — Compute Gradient
        grad_w, grad_b = grad_logreg_vec(X, y, w, b)

        w -= alpha * grad_w
        b -= alpha * grad_b

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return w, b, cost_history


def predict(X, w, b):
    """Convert probabilities to class labels"""
    probs = logistic(np.dot(X, w) + b)
    return (probs >= 0.5).astype(int)

def confusion_matrix_np(y_true, y_pred):
    """
        FP: wrongly predicted positives
        FN: wrongly predicted negatives
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def classification_metrics(y_true, y_pred):
    """
    Computes key classification metrics.
    Returns:
        accuracy  : overall correctness
        precision : TP / (TP + FP)
        recall    : TP / (TP + FN)
        f1_score  : harmonic mean of precision and recall
    """
    TP, TN, FP, FN = confusion_matrix_np(y_true, y_pred)
    precision = TP / (TP + FP + 1e-15)
    recall = TP / (TP + FN + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    accuracy = (TP + TN) / len(y_true) 
    return accuracy, precision, recall, f1

def roc_curve_np(y_true, probs, steps=100):
    """
    Compute ROC curve points.

    Returns:
        fpr_list: False Positive Rate
        tpr_list: True Positive Rate
    """
    thresholds = np.linspace(0, 1, steps)

    tpr_list = []
    fpr_list = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)

        TP, TN, FP, FN = confusion_matrix_np(y_true, y_pred)

        tpr = TP / (TP + FN + 1e-15)   # Recall
        fpr = FP / (FP + TN + 1e-15)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list)

def auc_np(fpr, tpr):
    """
    Compute AUC using trapezoidal rule
    """
    # sort by fpr -- Proper left → right movement
    sorted_idx = np.argsort(fpr)
    fpr = fpr[sorted_idx]
    tpr = tpr[sorted_idx]
    return np.trapz(tpr,fpr)

def best_threshold(y_true, probs):
    thresholds = np.linspace(0, 1, 100)

    best_t = 0
    best_score = -1

    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        TP, TN, FP, FN = confusion_matrix_np(y_true, y_pred)

        tpr = TP / (TP + FN + 1e-15)
        fpr = FP / (FP + TN + 1e-15)

        score = tpr - fpr   # Youden Index

        if score > best_score:
            best_score = score
            best_t = t

    return best_t


def run_titanic_scratch():
    """
    Train Logistic Regression from scratch and evaluate.

    Steps:
    1. Load and preprocess data
    2. Train model using gradient descent
    3. Predict on test data
    4. Evaluate using classification metrics
    5. Compare with sklearn model
    """

    # Load data
    # df = pd.read_csv("./data/Titanic.csv")
    df = pd.read_csv("./data/Titanic.csv")
    print(df.head())

    # Preprocessing
    df = fill_missing(df)
    df = feature_engineering(df)

    y = df["Survived"]
    X = encode_features(df.drop("Survived", axis=1))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Convert to numpy
    X_train = X_train.values.astype(np.float64)
    X_test = X_test.values.astype(np.float64)
    y_train = y_train.values.astype(np.float64)
    y_test = y_test.values.astype(np.float64)

    # Scale
    X_train, mean, std = standardize(X_train)
    X_test = (X_test - mean) / std

    # Initialize
    w = np.zeros(X_train.shape[1])
    b = 0

    # Train
    w, b, cost_history = grad_desc(X_train, y_train, w, b, alpha=0.001, n_iter=1000)
    
    print("\n===== Feature importance =====")
    feature_names = X.columns
    importance = np.abs(w)

    for name, val in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"{name}: {val:.4f}")

    probs = logistic(np.dot(X_test, w) + b)
    # Predict
    y_pred = predict(X_test, w, b)

    # Evaluate
    acc, prec, rec, f1 = classification_metrics(y_test, y_pred)

    print("\n===== SCRATCH MODEL =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Plot loss
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Loss")
    plt.show()


    fpr, tpr = roc_curve_np(y_test, probs)
    roc_auc = auc_np(fpr, tpr)

    print(f"\nAUC Score: {roc_auc:.4f}")
    print("\nInterpretation:")
    if roc_auc > 0.8:
        print("Strong model")
    elif roc_auc > 0.7:
        print("Decent model")
    else:
        print("Weak model")

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')  # random model
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


    t = best_threshold(y_test, probs)
    print(f"Best Threshold: {t:.3f}")

    y_pred_opt = (probs >= t).astype(int)

    acc, prec, rec, f1 = classification_metrics(y_test, y_pred_opt)

    print("\n===== OPTIMIZED THRESHOLD =====")
    print("\nThreshold Impact:")
    print("Default threshold = 0.5")
    print("Optimized threshold =", round(t, 3))

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Sklearn comparison
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("\nSklearn Accuracy:", model.score(X_test, y_test))

