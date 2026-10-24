"""
train.py
--------
Trains and compares three credit-risk models:
  1. Logistic Regression  (baseline — simple, interpretable)
  2. Random Forest        (intermediate — ensemble of trees)
  3. XGBoost              (final — gradient boosting, best accuracy)

All models are trained on the same data splits and evaluated with the
same metrics so the comparison is fair.

Evaluation metrics used:
  - ROC-AUC     : primary metric (handles imbalance)
  - PR-AUC      : precision-recall area (good for minority class)
  - Recall (bad): how many bad applicants we correctly catch
  - Precision (bad): of those we flag as bad, how many truly are
  - F1 (bad)    : balance of precision and recall for bad class
  - Gini        : industry standard = 2 * AUC - 1
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import prepare_data

# ─── Config ────────────────────────────────────────────────────────────────

DATA_PATH  = "data/raw/german_credit_with_risk.csv"
MODEL_DIR  = "models"
SEED       = 42   # Fixed seed = reproducible results every run

# ─── Evaluation Helper ─────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_val, y_val, X_test, y_test):
    """
    Compute all evaluation metrics on both validation and test sets.

    We evaluate on VALIDATION during model comparison.
    We evaluate on TEST only once at the very end (final report).

    Returns a dict of metrics for the comparison table.
    """
    # Predict probabilities (needed for AUC)
    val_proba  = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # Predict class labels (threshold = 0.5 default)
    val_pred  = (val_proba  >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    # Compute metrics
    val_auc   = roc_auc_score(y_val, val_proba)
    test_auc  = roc_auc_score(y_test, test_proba)
    val_prauc = average_precision_score(y_val, val_proba)

    # Classification report gives precision, recall, F1 per class
    report = classification_report(y_test, test_pred,
                                   target_names=["good", "bad"],
                                   output_dict=True)

    bad_recall    = report["bad"]["recall"]
    bad_precision = report["bad"]["precision"]
    bad_f1        = report["bad"]["f1-score"]
    gini          = 2 * test_auc - 1

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Val  ROC-AUC  : {val_auc:.4f}")
    print(f"  Test ROC-AUC  : {test_auc:.4f}   (Gini: {gini:.4f})")
    print(f"  Val  PR-AUC   : {val_prauc:.4f}")
    print(f"  Test Recall(bad)    : {bad_recall:.4f}  <- catching bad applicants")
    print(f"  Test Precision(bad) : {bad_precision:.4f}  <- avoiding false alarms")
    print(f"  Test F1(bad)        : {bad_f1:.4f}")
    print(f"\n  Confusion Matrix (Test):")
    cm = confusion_matrix(y_test, test_pred)
    print(f"                Predicted Good  Predicted Bad")
    print(f"  Actual Good:       {cm[0][0]:>5}            {cm[0][1]:>5}")
    print(f"  Actual Bad:        {cm[1][0]:>5}            {cm[1][1]:>5}")

    return {
        "Model":          name,
        "Val AUC":        round(val_auc, 4),
        "Test AUC":       round(test_auc, 4),
        "Gini":           round(gini, 4),
        "Val PR-AUC":     round(val_prauc, 4),
        "Recall(bad)":    round(bad_recall, 4),
        "Precision(bad)": round(bad_precision, 4),
        "F1(bad)":        round(bad_f1, 4),
    }


# ─── Model Definitions ─────────────────────────────────────────────────────

def get_logistic_regression():
    """
    Logistic Regression — our baseline model.

    class_weight='balanced': automatically adjusts weights so the model
    doesn't ignore the minority class (bad applicants).

    max_iter=1000: more iterations to ensure convergence on this dataset.
    """
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED
    )


def get_random_forest():
    """
    Random Forest — intermediate model.

    n_estimators=200: build 200 trees and vote. More trees = more stable.
    class_weight='balanced': same reason as LR.
    max_depth=10: limit tree depth to prevent overfitting on small dataset.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1   # Use all CPU cores
    )


def get_xgboost():
    """
    XGBoost — our primary model.

    scale_pos_weight: tells XGBoost how imbalanced the classes are.
    = count(good) / count(bad) = 700/300 = 2.33
    This makes it pay more attention to bad (defaulting) applicants.

    eval_metric='auc': optimize for AUC during training, not accuracy.
    """
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=700 / 300,   # handle class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,                # L1 regularization (prevents overfitting)
        reg_lambda=1.0,               # L2 regularization
        eval_metric="auc",
        use_label_encoder=False,
        random_state=SEED,
        verbosity=0
    )


# ─── Main Training Flow ────────────────────────────────────────────────────

def train_all():
    print("Loading and preprocessing data...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     preprocessor, feature_names) = prepare_data(DATA_PATH)

    models = {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest":       get_random_forest(),
        "XGBoost":             get_xgboost(),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_val, y_val, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model

    # Print comparison table
    print(f"\n\n{'='*75}")
    print("  MODEL COMPARISON TABLE")
    print(f"{'='*75}")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Save all models and preprocessor to disk
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in trained_models.items():
        fname = name.lower().replace(" ", "_") + ".pkl"
        fpath = os.path.join(MODEL_DIR, fname)
        with open(fpath, "wb") as f:
            pickle.dump(model, f)
        print(f"\nSaved: {fpath}")

    # Save preprocessor (MUST be saved with model — same transformations at inference)
    preproc_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"Saved: {preproc_path}")

    # Save feature names
    feat_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"Saved: {feat_path}")

    # Save comparison results
    df_results.to_csv("reports/model_comparison.csv", index=False)
    print(f"Saved: reports/model_comparison.csv")

    return trained_models, preprocessor, feature_names, X_test, y_test


if __name__ == "__main__":
    train_all()
