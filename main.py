"""FastAPI backend: loads every saved model and exposes prediction endpoints.

The frontend menu picks a model with the `model` query parameter:
    POST /predict/regression?model=random_forest
    POST /predict/classification?model=svm
If `model` is left out, the linear / logistic regression model is used.

Run (either works):
    python main.py                 # press Run in VS Code
    python -m uvicorn main:app --reload

Needs:  pip install fastapi uvicorn scikit-learn pandas joblib
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_models():
    """Load every saved model: {"linear": bundle, "random_forest": bundle, ...} plus metrics."""
    reg = {p.stem.replace("regression_", ""): joblib.load(p)
           for p in sorted(MODELS_DIR.glob("regression_*.joblib"))}
    clf = {p.stem.replace("classification_", ""): joblib.load(p)
           for p in sorted(MODELS_DIR.glob("classification_*.joblib"))}
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return reg, clf, metrics


def train_models():
    """Run src/train.py (from the project folder) to (re)create the models/ folder."""
    print("Training models with src/train.py ...")
    subprocess.run([sys.executable, str(BASE_DIR / "src" / "train.py")], cwd=BASE_DIR, check=True)


try:
    REG_MODELS, CLF_MODELS, METRICS = load_models()
    if not REG_MODELS or not CLF_MODELS:
        raise FileNotFoundError("no saved models found")
except Exception as exc:  # missing files, or models saved by a different scikit-learn version
    print(f"Could not load saved models ({exc!r}); retraining.")
    train_models()
    REG_MODELS, CLF_MODELS, METRICS = load_models()

app = FastAPI(title="Loan ML App")

# Allows the HTML page to work even if opened as a file or served from elsewhere
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class RegressionInput(BaseModel):
    education: int = Field(..., ge=0, le=1)
    self_employed: int = Field(..., ge=0, le=1)
    income_annum: float = Field(..., ge=0)
    loan_term: int = Field(..., ge=1)
    cibil_score: int = Field(..., ge=300, le=900)
    bank_asset_value: float = Field(..., ge=0)


class ClassificationInput(RegressionInput):
    loan_amount: float = Field(..., ge=0)


def get_bundle(models: dict, model: str):
    if model not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not available for this task. Choose from: {list(models)}",
        )
    return models[model]


@app.get("/")
def index():
    """Serve the frontend page."""
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/models")
def list_models():
    """Which models exist for each task, plus their test scores (used by the frontend menu)."""
    return {
        "regression": {k: METRICS["regression"][k] for k in REG_MODELS},
        "classification": {k: METRICS["classification"][k] for k in CLF_MODELS},
    }


@app.post("/predict/regression")
def predict_regression(data: RegressionInput, model: str = "linear"):
    bundle = get_bundle(REG_MODELS, model)
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    prediction = float(bundle["model"].predict(row)[0])
    return {
        "model": model,
        "model_name": bundle["name"],
        "r2": METRICS["regression"][model]["r2"],
        "predicted_loan_amount": round(max(prediction, 0), 2),
    }


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput, model: str = "linear"):
    bundle = get_bundle(CLF_MODELS, model)
    row = pd.DataFrame([data.model_dump()])[bundle["features"]]
    clf = bundle["model"]
    label = int(clf.predict(row)[0])
    probability = float(clf.predict_proba(row)[0][label])
    return {
        "model": model,
        "model_name": bundle["name"],
        "accuracy": METRICS["classification"][model]["accuracy"],
        "loan_status": label,
        "label": "Approved" if label == 1 else "Rejected",
        "confidence": round(probability, 4),
    }


if __name__ == "__main__":
    # Lets you start the server with "Run Python File" in VS Code
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
