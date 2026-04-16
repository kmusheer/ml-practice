import pandas as pd

def fill_missing(df):
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def encode_features(df) : 
    df = df.copy()  # safe practice
    # encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).fillna(0)
    # encode Embarked
    embarked_dummies = pd.get_dummies(df["Embarked"], drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)

    # ❌ DROP NON-NUMERIC COLUMNS
    df = df.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)

    return df

# feature engineering — it creates new useful columns from existing ones.
def feature_engineering(df) : 
    """
        SibSp = number of siblings/spouses aboard
        Parch = number of parents/children aboard
        +1 = the person themselves
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def feature_engineering_house_price(X) : 
    X = X.copy() 
    reference_columns = X.columns

    # Numeric columns
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # yes/no → 1/0
    yes_no_map = {"yes" : 1, "no" : 0}
    yn_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating','airconditioning', 'prefarea']
    # X[yn_cols] = X[yn_cols].map(yes_no_map)
    for col in yn_cols:
        X[col] = X[col].map(yes_no_map)

    # 🔥 One-hot encode furnishingstatus
    X = pd.get_dummies(X, columns=['furnishingstatus'], prefix="furnish", dtype=int)
    # print("Refrence colums ",reference_columns)
    # print("colums after ",X.columns)

    return X