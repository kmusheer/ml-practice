import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
 
from utils.preprocessing import feature_engineering_house_price
from utils.visualization import plot_predictions


np.set_printoptions(precision=2, suppress=True)
pd.set_option("display.float_format", "{:.2f}".format)

def run_regression_scratch() : 
    # ===============================
    # 1. Load Data
    # ===============================
    df = pd.read_csv("./data/Housing.csv")

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns)
    print("Final columns:", df.drop("price", axis=1).columns)

    # ===============================
    # 2. Split Features & Target
    # ===============================
    y = df["price"].values
    X = df.drop("price", axis=1)

    # Feature Engineering
    X_df = feature_engineering_house_price(X)
    X_columns = X_df.columns
    X = X_df.values   # convert to numpy
    print("After FE columns:", X.shape)
    print("Columns after FE:", list(X_columns))

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ===============================
    # 3. Train-Test Split
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===============================
    # 4. Add Bias Term
    # ===============================
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)
    print("X_train shape (with bias):", X_train.shape)


    # ===============================
    # 5. Train Model (Normal Equation)
    # ===============================
    if is_full_rank(X_train):
        print("OLS solution exists ✅")
        weights = normal_equation(X_train, y_train)
    else:
        print("Singular matrix → using pseudoinverse ⚠️")
        weights = normal_equation_pinv(X_train, y_train)

    print("Weights:", weights)

    # ===============================
    # 6. Predictions
    # ===============================
    y_pred = X_test @ weights

    # ===============================
    # 7. Visualization
    # ===============================
    plot_predictions(y_test, y_pred)

    # ===============================
    # 8. Single Prediction
    # ===============================
    sample = get_sample_input(X_columns)
    sample_pred = sample @ weights

    # ===============================
    # 9. Evaluation
    # ===============================

    print("\n***===== TRAIN PERFORMANCE =====***")
    y_train_pred = X_train @ weights
    evaluate_regression(y_train, y_train_pred)

    print("\n***===== TEST PERFORMANCE =====***")
    y_test_pred = X_test @ weights
    evaluate_regression(y_test, y_test_pred)

    print(f"Predicted Price: ₹{sample_pred[0]:,.0f}")



def add_bias(X):
    ones = np.ones((X.shape[0],1))
    return np.hstack((ones,X))

def is_full_rank(X):
    rank = np.linalg.matrix_rank(X)
    return rank == X.shape[1]

def normal_equation(X,y) :
    XT = X.T
    return np.linalg.solve(XT @ X, XT @ y)

def normal_equation_pinv(X, y):
    # Moore–Penrose pseudoinverse X+
    return np.linalg.pinv(X) @ y

def evaluate_regression(y_true, y_pred) :

    residuals = y_true - y_pred
    n = len(y_true)
    sse = np.sum(residuals ** 2)        # sum_of_square_error
    mse = sse / n
    mae = np.mean(np.abs(residuals))

    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (sse / ss_tot)

    print("\n===== Evaluation =====")
    print(f"MSE: {mse:.2e}")
    print(f"MAE: {mae:.2e}")
    print(f"R2 : {r2:.4f}")

def get_sample_input(X_columns) :
    sample_dict = {
        "area": 7420,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 3,
        "mainroad": 1,
        "guestroom": 0,
        "basement": 1,
        "hotwaterheating": 1,
        "airconditioning": 0,
        "parking": 1,
        "prefarea": 1,
        "furnish_furnished": 1,
        "furnish_semi-furnished": 0,
        "furnish_unfurnished": 0
    }      # furnishingstatus (encoded)  

    sample_df = pd.DataFrame([sample_dict])
    # sample_df = sample_df[X_columns]  # ensure order

    sample_df = sample_df[X_columns]
    sample = sample_df.values
    sample = add_bias(sample)

    return sample
    
