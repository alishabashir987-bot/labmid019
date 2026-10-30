# Loan ML App

A full-stack machine learning app built on a loan dataset. It trains several scikit-learn models, serves them through a FastAPI backend, and lets you try them from a simple web page.

The app answers two questions about a loan applicant:

1. **Regression:** how much loan (`loan_amount`) is this person likely to get?
2. **Classification:** will the loan be **approved or rejected** (`loan_status`)?

---

## What happens in this project

```
cleaned_LOAN.csv  ->  src/train.py  ->  models/*.joblib + metrics.json
                                              |
                                              v
                      frontend/index.html  <->  main.py (FastAPI)
```

1. **Data**: `data/processed/cleaned_LOAN.csv` is the final, already-cleaned dataset.
2. **Training**: `src/train.py` trains 4 regression models and 5 classification models, scores each on a held-out test set, and saves them with `joblib`.
3. **Backend**: `main.py` loads every saved model at startup and exposes prediction endpoints.
4. **Frontend**: `frontend/index.html` is a single page. You pick a model from a dropdown, fill in a form, and see the prediction.

---

## Dataset

- **File:** `data/processed/cleaned_LOAN.csv`
- **Size:** 4,269 rows, 8 columns, no missing values
- **Class balance:** 2,656 approved (about 62%) and 1,613 rejected (about 38%)

| Column | Meaning |
|---|---|
| `education` | 1 = Graduate, 0 = Not Graduate |
| `self_employed` | 1 = Yes, 0 = No |
| `income_annum` | Annual income |
| `loan_amount` | Loan amount requested or granted |
| `loan_term` | Loan term in years |
| `cibil_score` | Credit score (300 to 900) |
| `bank_asset_value` | Value of the applicant's bank assets |
| `loan_status` | **Target for classification:** 1 = Approved, 0 = Rejected |

---

## Models

Both tasks use an 80/20 train/test split (`random_state=42`). The classification split is stratified so both sets keep the same approved/rejected ratio.

### Regression: predict `loan_amount`
**Inputs:** education, self_employed, income_annum, loan_term, cibil_score, bank_asset_value

| Model | R² score |
|---|---|
| Linear Regression | 0.8513 |
| Decision Tree (max depth 8) | 0.8278 |
| Random Forest (200 trees) | 0.8443 |
| SVM (SVR, linear kernel) | 0.8511 |

### Classification: predict `loan_status`
**Inputs:** all of the above plus `loan_amount`

| Model | Accuracy |
|---|---|
| Logistic Regression | 91.22% |
| Decision Tree (max depth 6) | 97.31% |
| Naive Bayes (Gaussian) | 75.64% |
| Random Forest (200 trees) | **98.36%** |
| SVM (SVC, RBF kernel) | 94.96% |

**Notes**
- Naive Bayes is classification-only, so it has no regression version. The dropdown shows it only for the approval task.
- Linear, Logistic and SVM models use `StandardScaler`, because income and loan values are in the millions while the CIBIL score is only about 300 to 900. The SVR also scales the target.
- For regression, all models land at about 0.83 to 0.85 R². For classification, tree-based models do best.

---

## Project structure

```
ALISHA-ML/
├── data/
│   └── processed/cleaned_LOAN.csv   # dataset used for modeling
├── models/                          # saved models + metrics.json
├── src/
│   └── train.py                     # trains and saves all models
├── frontend/
│   └── index.html                   # web page served by main.py
├── main.py                          # FastAPI backend
├── requirements.txt
└── README.md
```

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) retrain all models and regenerate models/ and metrics.json
python src/train.py

# 3. Start the server
python main.py
```

Then open **http://127.0.0.1:8001** in your browser.

You can also start it with `python -m uvicorn main:app --reload`.

If the saved models are missing, or were saved by a different scikit-learn version and fail to load, `main.py` retrains them automatically on startup.

---

## Using the web page

1. Choose a model from the **Select model** dropdown (Linear/Logistic, Decision Tree, Naive Bayes, Random Forest, SVM).
2. **Regression form:** enter the applicant details and click **Predict amount** to get an estimated loan amount.
3. **Classification form:** enter the details plus the loan amount and click **Predict status** to get *Approved* or *Rejected* with a confidence score.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend page |
| GET | `/models` | Lists available models and their test scores |
| POST | `/predict/regression?model=...` | Predicts `loan_amount` |
| POST | `/predict/classification?model=...` | Predicts approval status |

`model` options:
- Regression: `linear`, `decision_tree`, `random_forest`, `svm`
- Classification: `linear`, `decision_tree`, `naive_bayes`, `random_forest`, `svm`

If `model` is left out, it defaults to `linear`. An unknown model name returns a 404 listing the valid choices.

### Example: regression

```bash
curl -X POST "http://127.0.0.1:8001/predict/regression?model=random_forest" \
  -H "Content-Type: application/json" \
  -d '{"education":1,"self_employed":0,"income_annum":9600000,"loan_term":12,"cibil_score":778,"bank_asset_value":8000000}'
```

Response:
```json
{ "model": "random_forest", "model_name": "Random Forest", "r2": 0.8443, "predicted_loan_amount": 29500000.0 }
```

### Example: classification

```bash
curl -X POST "http://127.0.0.1:8001/predict/classification?model=svm" \
  -H "Content-Type: application/json" \
  -d '{"education":1,"self_employed":0,"income_annum":9600000,"loan_amount":29900000,"loan_term":12,"cibil_score":778,"bank_asset_value":8000000}'
```

Response:
```json
{ "model": "svm", "model_name": "SVM (SVC)", "accuracy": 0.9496, "loan_status": 1, "label": "Approved", "confidence": 0.97 }
```

*(Example response values are illustrative.)*

Inputs are validated: `education` and `self_employed` must be 0 or 1, `cibil_score` must be between 300 and 900, and the numeric fields must be non-negative.

---

## Tech stack

- **Python**, **pandas**: data handling
- **scikit-learn**: models, pipelines, metrics
- **joblib**: saving and loading models
- **FastAPI** and **Uvicorn**: backend API
- **HTML, CSS, JavaScript**: frontend
