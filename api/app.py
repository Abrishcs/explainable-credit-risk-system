"""
app.py — Credit Risk Prediction API
-------------------------------------
A Flask web service that accepts applicant data and returns:
  - A risk probability (0.0 to 1.0)
  - A decision: APPROVE or REJECT
  - The top reasons (from SHAP) explaining the decision

Endpoints:
  GET  /health      → confirms the API is running
  POST /predict     → accepts applicant JSON, returns prediction + explanation
  GET  /example     → shows a sample request body you can copy-paste

How to run:
  cd "C:\\Users\\Administrator\\Documents\\Main Folder\\tiktok\\project 1"
  python api/app.py

Then open another terminal and test:
  curl -X POST http://127.0.0.1:5000/predict \
       -H "Content-Type: application/json" \
       -d @api/sample_request.json
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import shap

# Add src/ to path so we can import our preprocessing module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ─── App Setup ─────────────────────────────────────────────────────────────

app = Flask(__name__)

# Paths — relative to the project root
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "xgboost.pkl")
PREPROC_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
FEAT_PATH   = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# ─── Load Model at Startup ─────────────────────────────────────────────────
# We load once when the server starts — NOT on every request.
# Loading a model takes ~0.5s. Loading per request = extremely slow.

print("Loading model artifacts...")
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(PREPROC_PATH, "rb") as f:
    PREPROCESSOR = pickle.load(f)
with open(FEAT_PATH, "rb") as f:
    FEATURE_NAMES = pickle.load(f)

# Build SHAP explainer once (also slow to build — do it once)
EXPLAINER = shap.TreeExplainer(MODEL)
print("Model loaded. API ready.")

# ─── Input Validation ──────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "Age":              (int,   "Applicant age in years (e.g. 35)"),
    "Sex":              (str,   "Gender: 'male' or 'female'"),
    "Job":              (int,   "Job skill level: 0=unskilled, 1=unskilled resident, 2=skilled, 3=highly skilled"),
    "Housing":          (str,   "Housing type: 'own', 'rent', or 'free'"),
    "Saving accounts":  (str,   "Savings level: 'none', 'little', 'moderate', 'quite rich', or 'rich'"),
    "Checking account": (str,   "Checking account: 'none', 'little', 'moderate', or 'rich'"),
    "Credit amount":    (int,   "Loan amount requested in DM (e.g. 5000)"),
    "Duration":         (int,   "Loan duration in months (e.g. 24)"),
    "Purpose":          (str,   "Loan purpose: 'car', 'radio/TV', 'furniture/equipment', 'business', 'education', 'repairs', 'domestic appliances', or 'vacation/others'"),
}

VALID_VALUES = {
    "Sex":              ["male", "female"],
    "Housing":          ["own", "rent", "free"],
    "Saving accounts":  ["none", "little", "moderate", "quite rich", "rich"],
    "Checking account": ["none", "little", "moderate", "rich"],
    "Purpose":          ["car", "radio/TV", "furniture/equipment", "business",
                         "education", "repairs", "domestic appliances", "vacation/others"],
    "Job":              [0, 1, 2, 3],
}


def validate_input(data: dict):
    """
    Check that all required fields are present and have valid values.
    Returns a list of error messages (empty = no errors).

    Good validation prevents confusing model errors and protects the API.
    """
    errors = []

    for field, (ftype, description) in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}' — {description}")
            continue

        val = data[field]
        # Check type
        if not isinstance(val, ftype):
            errors.append(
                f"Field '{field}' must be {ftype.__name__}, got {type(val).__name__}"
            )
            continue

        # Check allowed values for categorical fields
        if field in VALID_VALUES and val not in VALID_VALUES[field]:
            errors.append(
                f"Field '{field}' = '{val}' is not valid. "
                f"Allowed: {VALID_VALUES[field]}"
            )

    # Range checks for numeric fields
    if "Age" in data and isinstance(data["Age"], int):
        if not (18 <= data["Age"] <= 100):
            errors.append("Age must be between 18 and 100")
    if "Credit amount" in data and isinstance(data["Credit amount"], int):
        if data["Credit amount"] <= 0:
            errors.append("Credit amount must be a positive number")
    if "Duration" in data and isinstance(data["Duration"], int):
        if not (1 <= data["Duration"] <= 120):
            errors.append("Duration must be between 1 and 120 months")

    return errors


# ─── Prediction Logic ───────────────────────────────────────────────────────

def make_prediction(applicant: dict, threshold: float = 0.5, top_n: int = 3):
    """
    Core prediction function:
    1. Build a DataFrame from the applicant dict
    2. Run it through the fitted preprocessor
    3. Get XGBoost probability
    4. Compute SHAP values for this applicant
    5. Extract top reasons for the decision

    Returns a dict with all prediction details.
    """
    # Step 1: Build single-row DataFrame
    df = pd.DataFrame([applicant])

    # Step 2: Preprocess (same pipeline as training)
    X_processed = PREPROCESSOR.transform(df)

    # Step 3: Predict probability of default (bad = class 1)
    prob_bad = float(MODEL.predict_proba(X_processed)[0][1])
    decision = "REJECT" if prob_bad >= threshold else "APPROVE"

    # Step 4: SHAP values for this applicant
    X_df = pd.DataFrame(X_processed, columns=FEATURE_NAMES)
    shap_vals = EXPLAINER(X_df)
    sv = shap_vals.values[0]           # shape: (n_features,)
    base_value = float(shap_vals.base_values[0])

    # Step 5: Rank features by their contribution
    contributions = sorted(
        zip(FEATURE_NAMES, sv.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Separate into risk factors (positive SHAP) and protective factors (negative)
    risk_factors = [
        {"feature": name, "shap": round(val, 4), "direction": "increases_risk"}
        for name, val in contributions if val > 0
    ][:top_n]

    protective_factors = [
        {"feature": name, "shap": round(val, 4), "direction": "reduces_risk"}
        for name, val in contributions if val < 0
    ][:top_n]

    # Build a plain-English explanation
    if decision == "REJECT":
        top_reasons = [f["feature"] for f in risk_factors[:3]]
        explanation = (
            f"Application REJECTED. Primary risk factors: "
            f"{', '.join(top_reasons)}."
        )
    else:
        top_reasons = [f["feature"] for f in protective_factors[:3]]
        explanation = (
            f"Application APPROVED. Strongest positive factors: "
            f"{', '.join(top_reasons)}."
        )

    return {
        "decision":           decision,
        "probability_of_default": round(prob_bad, 4),
        "risk_level":         _risk_level(prob_bad),
        "explanation":        explanation,
        "risk_factors":       risk_factors,
        "protective_factors": protective_factors,
        "shap_base_value":    round(base_value, 4),
        "threshold_used":     threshold,
    }


def _risk_level(prob: float) -> str:
    """Convert probability to a human-readable risk tier."""
    if prob < 0.25:   return "LOW"
    if prob < 0.50:   return "MEDIUM"
    if prob < 0.75:   return "HIGH"
    return "VERY HIGH"


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Renders the interactive web interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Simple health check.
    Useful for deployment monitoring — tells you the server is alive.
    """
    return jsonify({
        "status": "healthy",
        "model":  "XGBoost Credit Risk v1.0",
        "features": len(FEATURE_NAMES),
    }), 200


@app.route("/example", methods=["GET"])
def example():
    """Returns a sample request body so you can test immediately."""
    return jsonify({
        "description": "Copy this body and POST it to /predict",
        "sample_applicant": {
            "Age":              35,
            "Sex":              "male",
            "Job":              2,
            "Housing":          "own",
            "Saving accounts":  "little",
            "Checking account": "little",
            "Credit amount":    5000,
            "Duration":         36,
            "Purpose":          "car"
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects: JSON body with applicant fields (see /example for structure)
    Returns: decision, probability, SHAP-based explanation

    HTTP status codes:
      200 = success
      400 = bad input (validation failed)
      500 = server error (model failed unexpectedly)
    """
    # Check Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Request must have Content-Type: application/json"
        }), 400

    data = request.get_json()

    # Validate input
    errors = validate_input(data)
    if errors:
        return jsonify({
            "error":   "Invalid input",
            "details": errors
        }), 400

    # Run prediction
    try:
        result = make_prediction(data)
        return jsonify(result), 200

    except Exception as e:
        # Log the real error server-side, return a safe message to client
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "error": "Prediction failed due to an internal error."
        }), 500


# ─── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # debug=True: auto-reloads when you save the file — useful during development
    # debug=False in production: never expose stack traces to users
    app.run(host="0.0.0.0", port=5000, debug=False)
