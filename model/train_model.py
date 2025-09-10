# model/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import dump
import os

# Data columns for the Pima Indians Diabetes dataset
COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

def load_data():
    # Public mirror of the dataset (header included)
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)
    # Ensure expected cols
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace biologically-impossible zeros with NaN, then impute median
    zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in zero_invalid_cols:
        df[c] = df[c].replace(0, np.nan)
        df[c] = df[c].fillna(df[c].median())

    return df

def main():
    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Validation ROC-AUC: {auc:.3f}")

    # Persist artifacts
    os.makedirs("model", exist_ok=True)
    dump(clf, "model/model.pkl")
    dump(scaler, "model/scaler.pkl")
    print("Saved model/model.pkl and model/scaler.pkl")

if __name__ == "__main__":
    main()
