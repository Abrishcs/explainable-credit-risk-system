# Explainable Credit Risk Prediction System

A production-ready machine learning system that predicts loan default risk and provides transparent, regulation-aligned explanations for every decision using SHAP (SHapley Additive exPlanations).

## Problem Statement

Banks and microfinance institutions need to assess whether a loan applicant is likely to default. Traditional rule-based scoring is slow and misses complex patterns. This system uses XGBoost to achieve strong predictive performance, while SHAP makes every decision explainable to applicants, auditors, and regulators.

## Results

| Model               | Test ROC-AUC | Gini  | F1 (bad class) |
|---------------------|-------------|-------|----------------|
| Logistic Regression | 0.638       | 0.275 | 0.451          |
| Random Forest       | 0.753       | 0.507 | 0.564          |
| **XGBoost**         | **0.770**   | **0.540** | **0.574**  |

Dataset: German Credit (UCI/Kaggle) — 1,000 applicants, 9 features.

## Features

- Three-model comparison (LR → RF → XGBoost) with fair evaluation
- SHAP global feature importance (portfolio-level insight)
- SHAP waterfall plots (per-applicant explanation)
- Plain-English adverse action notices per applicant
- REST API with input validation and SHAP-powered responses
- Full preprocessing pipeline with zero data leakage

## Project Structure

```
credit-risk-system/
├── data/
│   └── raw/
│       └── german_credit_with_risk.csv
├── src/
│   ├── preprocessing.py   # Data loading, cleaning, encoding, splitting
│   ├── train.py           # Model training and comparison
│   └── explain.py         # SHAP global and local explanations
├── api/
│   ├── app.py             # Flask REST API & Web UI
│   ├── sample_request.json
│   └── templates/
│       └── index.html     # Interactive Web Interface
├── models/
│   ├── xgboost.pkl
│   ├── preprocessor.pkl
│   └── feature_names.pkl
├── tests/
│   └── test_api.py
├── reports/
│   ├── model_comparison.csv
│   ├── shap_global_bar.png
│   ├── shap_beeswarm.png
│   └── shap_waterfall_*.png
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models
python src/train.py

# 3. Generate SHAP plots
python src/explain.py

# 4. Start the API
python api/app.py
```

## Web UI & API Usage

### 1. Interactive Web Interface (Recommended)
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```
You will find a clean interactive form to enter applicant details, submit them, and immediately view:
- **Decision:** APPROVE (Green) or REJECT (Red)
- **Probability of Default bar** with risk classification (`LOW`, `MEDIUM`, `HIGH`, `VERY HIGH`)
- **Plain-English explanation** aligned with credit compliance guidelines
- **SHAP Breakdown:** Key Risk Factors (+SHAP) vs. Key Protective Factors (-SHAP)

### 2. REST API (curl / JSON)

**Start server:** `python api/app.py`

**Health check:**
```bash
curl http://127.0.0.1:5000/health
```

**Make a prediction:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 35, "Sex": "male", "Job": 2,
       "Housing": "own", "Saving accounts": "little",
       "Checking account": "little", "Credit amount": 5000,
       "Duration": 36, "Purpose": "car"
     }'
```

**Sample response:**
```json
{
  "decision": "APPROVE",
  "probability_of_default": 0.22,
  "risk_level": "LOW",
  "explanation": "Application APPROVED. Strongest positive factors: Checking account, Saving accounts, Job.",
  "risk_factors": [...],
  "protective_factors": [...]
}
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| XGBoost over Logistic Regression | +13% AUC gain; captures non-linear credit patterns |
| SHAP over LIME | Mathematically rigorous, stable, supports both global and local views |
| Class weights over SMOTE | Avoids synthetic data distorting SHAP explanations |
| Informative missing = "none" | 39% of checking account data is missing — deletion would lose 40% of data |
| Stratified split | Preserves 70/30 class ratio across all three sets |

## Responsible AI Notes

- `Sex` is included as a feature but monitored via SHAP for disproportionate influence
- The model is a **decision-support tool**, not an automatic rejection system
- All SHAP findings align with established financial domain knowledge
- Dataset is from 1994 Germany — not directly applicable to Ethiopian lending without local data

## Limitations

- 1,000 rows is a small dataset; AUC estimates have uncertainty
- German 1994 data; a production Ethiopian system needs local, recent data
- No temporal validation (model stability over time not tested)
- Threshold of 0.5 is a starting point; real deployment requires business calibration

## Tech Stack

Python · XGBoost · scikit-learn · SHAP · Flask · pandas · matplotlib
