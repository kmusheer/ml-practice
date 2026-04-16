import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from utils.preprocessing import feature_engineering_house_price
from utils.visualization import plot_residuals


np.set_printoptions(precision=2, suppress=True)
pd.set_option("display.float_format", "{:.2f}".format)

def run_regression() : 
    df = pd.read_csv("./data/Housing.csv")
    print(len(df))
    print(df.shape)
    print(df.columns)
    print(type(df.columns))

    # Target & Features
    y = df["price"]
    X = df.drop("price", axis=1)


    # Feature Engineering
    X = feature_engineering_house_price(X)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ ADD BASELINE HERE
    baseline = np.mean(y_train)
    y_pred_base = np.full_like(y_test, baseline)

    print("\n===== BASELINE =====")
    print("MSE:", mean_squared_error(y_test, y_pred_base))
    print("MAE:", mean_absolute_error(y_test, y_pred_base))
    print("R2 :", r2_score(y_test, y_pred_base))

    print("\n===== CROSS-VALIDATION =====")
    cross_validate_regression(LinearRegression(), X, y, "Linear Regression")
    cross_validate_regression(Ridge(alpha=1.0), X, y, "Ridge")
    cross_validate_regression(RandomForestRegressor(n_estimators=100), X, y, "Random Forest")
    cross_validate_regression(XGBRegressor(n_estimators=100), X, y, "XGBoost")

    # Scaling (ONLY for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    results = []
    # Linear models (scaled)
    results.append({
        "Model": "Baseline",
        "MSE": mean_squared_error(y_test, y_pred_base),
        "MAE": mean_absolute_error(y_test, y_pred_base),
        "R2": r2_score(y_test, y_pred_base)
    })
    results.append(train_and_evaluate_regression(LinearRegression(), X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression"))
    results.append(train_and_evaluate_regression(Ridge(alpha=1.0), X_train_scaled, X_test_scaled, y_train, y_test, "Ridge"))

    # Tree models (NO scaling)
    results.append(train_and_evaluate_regression(RandomForestRegressor(n_estimators=100, random_state=42),X_train, X_test, y_train, y_test, "Random Forest"))

    results.append(train_and_evaluate_regression(XGBRegressor(n_estimators=100, learning_rate=0.1),X_train, X_test, y_train, y_test, "XGBoost"))


    print("\n===== Feature importance =====")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
    )

    feature_importance = feature_importance.sort_values(ascending=False)

    feature_importance.plot(kind='bar')
    plt.title("Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Compare
    print("\n===== MODEL COMPARISON =====")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by="R2", ascending=False))

    # Interpretation
    print("\n===== INTERPRETATION =====")
    best_model = results_df.sort_values(by="R2", ascending=False).iloc[0]
    print(f"Best model based on R2: {best_model['Model']}")
    print("R2 measures how well variance is explained.")


def cross_validate_regression(model, X, y, name) :
    if isinstance(model, (LinearRegression, Ridge)):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
    else:
        pipeline = Pipeline([
            ("model", model)
        ])

    scores = cross_val_score(pipeline,X,y, cv=5, scoring="r2")
    print(f"\n===== {name} Cross-Validation =====")
    print("Scores:", scores)
    print(f"Mean R2: {scores.mean():.4f}")
    print(f"Std Dev: {scores.std():.4f}")

    return scores.mean()

def train_and_evaluate_regression(model, X_train, X_test, y_train, y_test, name) :
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 :", r2)

     # Residual plot (IMPORTANT)
    plot_residuals(y_test,y_pred, name)


    return {
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }
