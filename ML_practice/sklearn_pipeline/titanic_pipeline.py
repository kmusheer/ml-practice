import pandas as pd

from utils.preprocessing import (
    fill_missing,
    encode_features,
    feature_engineering
)
from utils.evaluation import evaluate_model
from utils.visualization import plot_roc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_titanic():
     # Load data
    df = pd.read_csv("./data/Titanic.csv")

    print(df.shape)
    print(df.columns)
    print("Missing Age values:", df["Age"].isnull().sum())

    # Preprocessing
    df = fill_missing(df)
    print("Missing Age after fill:", df["Age"].isnull().sum())

    y = df["Survived"]

    df = feature_engineering(df)
    X = df.drop("Survived", axis=1)
    X = encode_features(X)
                # 
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("df type:", type(df))
    print("X type:", type(X))
     
    # train models
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
    print("X_Train : ", X_train.shape)
    print("X_Test : ", X_test.shape)
    print("y_Train : ", y_train.shape)
    print("y_Test : ", y_test.shape)
    print(type(y_test))

    print("\n===== CROSS-VALIDATION =====")
    cross_validate_model(LogisticRegression(max_iter=200), X, y, "Logistic Regression")
    cross_validate_model(KNeighborsClassifier(n_neighbors=5), X, y, "KNN")
    cross_validate_model(DecisionTreeClassifier(max_depth=5), X, y, "Decision Tree")
    cross_validate_model(RandomForestClassifier(n_estimators=100), X, y, "Random Forest")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    results.append(train_and_evaluate(LogisticRegression(max_iter=200),X_train, X_test, y_train, y_test,"Logistic Regression"))
    results.append(train_and_evaluate(KNeighborsClassifier(n_neighbors=5),X_train, X_test, y_train, y_test,"KNN"))
    results.append(train_and_evaluate(DecisionTreeClassifier(max_depth=5),X_train, X_test, y_train, y_test,"Decision Tree"))
    results.append(train_and_evaluate(RandomForestClassifier(n_estimators=100),X_train, X_test, y_train, y_test,"Random Forest"))
    
    print("\n===== MODEL COMPARISON =====")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by="Accuracy", ascending=False))

    results_df["AUC"] = results_df["AUC"].fillna(0)
    
    print("\n===== INTERPRETATION =====")

    best_auc_model = results_df.sort_values(by="AUC", ascending=False).iloc[0]

    print(f"Best model based on AUC: {best_auc_model['Model']}")
    print("AUC reflects overall classification ability across thresholds.")


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    # [0, 1, 1, 0, 0]
    y_pred = model.predict(X_test)

    #  [
    #   [0.8, 0.2],
    #   [0.3, 0.7],
    #   [0.1, 0.9]
    #  ]
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = None

    print(f"\n===== {name} =====")
    metrics = evaluate_model(y_test, y_pred, y_prob)

    if y_prob is not None:
        plot_roc(y_test, y_prob)

    metrics["Model"] = name
    return metrics
    


def cross_validate_model(model, X, y, name):
    """
    Perform 5-fold cross-validation and return mean accuracy
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    print(f"\n===== {name} Cross-Validation =====")
    print("Scores:", scores)
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Std Dev      : {scores.std():.4f}")

    return scores.mean()