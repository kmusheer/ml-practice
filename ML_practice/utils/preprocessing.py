import pandas as pd

def fill_missing(df):
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


def encode_features(df) : 
    df = df.copy()  # safe practice
    # encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
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
