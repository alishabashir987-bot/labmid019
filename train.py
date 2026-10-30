"""Train several regression and classification models on the loan dataset.

Regression     : predict loan_amount
                 Linear Regression, Decision Tree, Random Forest, SVM (SVR)
Classification : predict loan_status (0 = rejected, 1 = approved)
                 Logistic Regression, Decision Tree, Naive Bayes, Random Forest, SVM (SVC)

Naive Bayes is a classification-only algorithm, so it has no regression version.

Run (from the project folder):  python src/train.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    BASE_DIR = Path(__file__).resolve().parent.parent  # project root (this file lives in src/)
except NameError:  # running in a notebook: __file__ is not defined, use the current folder
    BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Dataset ready for modeling
CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_LOAN.csv"
if not CSV_PATH.exists():
    raise FileNotFoundError(f"cleaned_LOAN.csv not found at {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

REG_FEATURES = [
    "education", "self_employed", "income_annum",
    "loan_term", "cibil_score", "bank_asset_value",
]
CLF_FEATURES = [
    "education", "self_employed", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "bank_asset_value",
]

# Scaling matters for SVM: income/loan values are in the millions, cibil_score is ~300-900.
REG_MODELS = {
    "linear": ("Linear Regression", LinearRegression()),
    "decision_tree": ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=42)),
    "random_forest": ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    # SVR also needs the target scaled (loan_amount is in the tens of millions)
    "svm": ("SVM (SVR)", TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="linear", C=1.0)),
        transformer=StandardScaler())),
}
CLF_MODELS = {
    "linear": ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=42)),
    "naive_bayes": ("Naive Bayes", GaussianNB()),
    "random_forest": ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    "svm": ("SVM (SVC)", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))),
}

metrics = {"regression": {}, "classification": {}}

# ---------- 1. Regression: predict loan_amount ----------
X, y = df[REG_FEATURES], df["loan_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for key, (name, model) in REG_MODELS.items():
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    metrics["regression"][key] = {"name": name, "r2": round(r2, 4)}
    joblib.dump({"model": model, "features": REG_FEATURES, "name": name}, MODELS_DIR / f"regression_{key}.joblib")
    print(f"Regression     {name:<20} R2       : {r2:.4f}")

# ---------- 2. Classification: predict loan_status ----------
X, y = df[CLF_FEATURES], df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
for key, (name, model) in CLF_MODELS.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    metrics["classification"][key] = {"name": name, "accuracy": round(acc, 4)}
    joblib.dump({"model": model, "features": CLF_FEATURES, "name": name}, MODELS_DIR / f"classification_{key}.joblib")
    print(f"Classification {name:<20} accuracy : {acc:.4f}")

# ---------- 3. Save metrics so the API / frontend can show them ----------
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved models to models/")