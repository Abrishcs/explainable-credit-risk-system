"""
explain.py
----------
Generates SHAP explanations for the trained XGBoost credit-risk model.

Two types of explanations are produced:

GLOBAL (portfolio-level):
  - Bar chart: average importance of each feature across all applicants
  - Beeswarm plot: how high/low feature values push the risk score

LOCAL (per-applicant):
  - Waterfall plot: exactly why one specific applicant was approved/rejected
  - Force plot (saved as HTML): interactive version

All plots are saved to the reports/ folder.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import prepare_data, TARGET

# ─── Config ────────────────────────────────────────────────────────────────

DATA_PATH   = "data/raw/german_credit_with_risk.csv"
MODEL_PATH  = "models/xgboost.pkl"
PREPROC_PATH = "models/preprocessor.pkl"
FEAT_PATH   = "models/feature_names.pkl"
REPORT_DIR  = "reports"


# ─── Load Artifacts ────────────────────────────────────────────────────────

def load_artifacts():
    """Load the trained model, preprocessor, and feature names from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PREPROC_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(FEAT_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return model, preprocessor, feature_names


# ─── SHAP Core ─────────────────────────────────────────────────────────────

def compute_shap_values(model, X_array, feature_names):
    """
    Use TreeSHAP — the fast, exact SHAP method for tree-based models.

    TreeSHAP is O(TLD²) instead of the exponential brute-force approach.
    It gives the SAME mathematically correct answer but much faster.

    Returns:
        explainer   : the SHAP explainer object
        shap_values : matrix of shape (n_samples, n_features)
                      each value = how much that feature pushed this
                      prediction away from the average prediction
    """
    print("Computing SHAP values using TreeSHAP...")

    # Convert to DataFrame so SHAP can label features
    X_df = pd.DataFrame(X_array, columns=feature_names)

    # TreeExplainer is the correct explainer for XGBoost/RF/LightGBM
    explainer = shap.TreeExplainer(model)

    # shap_values[:,1] = SHAP values for class 1 (bad/default)
    # This is what we care about — what pushes an applicant toward default
    shap_values = explainer(X_df)

    print(f"SHAP values computed. Shape: {shap_values.values.shape}")
    print(f"  Rows    = {shap_values.values.shape[0]} applicants")
    print(f"  Columns = {shap_values.values.shape[1]} features")

    return explainer, shap_values, X_df


# ─── Plot 1: Global Bar Chart ───────────────────────────────────────────────

def plot_global_bar(shap_values, feature_names):
    """
    Bar chart of mean |SHAP| values.

    Mean |SHAP| = average absolute contribution of each feature
    across all applicants. This answers: 'Which features matter most
    for credit risk decisions overall?'
    """
    print("\nGenerating: Global Feature Importance Bar Chart...")

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Mean |SHAP|": mean_shap
    }).sort_values("Mean |SHAP|", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"],
                   color="#e05c5c", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value| (average impact on risk score)", fontsize=11)
    ax.set_title("Global Feature Importance\nXGBoost Credit Risk Model",
                 fontsize=13, fontweight="bold", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, importance_df["Mean |SHAP|"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#333")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "shap_global_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return importance_df


# ─── Plot 2: Beeswarm Summary Plot ─────────────────────────────────────────

def plot_beeswarm(shap_values):
    """
    Beeswarm plot — the most informative SHAP plot.

    Each dot = one applicant.
    X-axis = SHAP value (positive = pushed toward BAD/default)
    Color  = feature value (red = high, blue = low)

    This shows BOTH:
    - How important a feature is (spread on X)
    - Which direction the feature pushes risk (left = safer, right = riskier)
    """
    print("Generating: Beeswarm Summary Plot...")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot — All Applicants\nXGBoost Credit Risk Model",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Plot 3: Local Waterfall (Individual Applicant) ────────────────────────

def plot_waterfall(shap_values, X_df, y_test, applicant_index=None):
    """
    Waterfall plot for ONE applicant — answers 'Why was this person rejected?'

    We pick the most interesting case: a bad applicant the model correctly
    flagged (true positive) — this is the most useful demo for a portfolio.

    The waterfall shows:
    - E[f(X)]: the base rate (average risk across all applicants)
    - Each bar: how much each feature pushed this applicant's score up or down
    - f(X): the final risk score for this applicant
    """
    # Find a correctly-identified bad applicant (true positive)
    y_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    if applicant_index is None:
        # Get model predictions to find a clear true-positive case
        risk_scores = shap_values.values.sum(axis=1) + shap_values.base_values
        predicted_bad = (1 / (1 + np.exp(-risk_scores))) > 0.5
        true_bad = y_arr == 1
        true_positives = np.where(predicted_bad & true_bad)[0]

        if len(true_positives) == 0:
            applicant_index = np.where(true_bad)[0][0]
            print("  No clean true positive found, using first bad applicant.")
        else:
            # Pick the one with the highest risk score (most convincing demo)
            applicant_index = true_positives[np.argmax(risk_scores[true_positives])]

    print(f"\nGenerating: Waterfall Plot for Applicant #{applicant_index}...")
    print(f"  True label: {'BAD (default risk)' if y_arr[applicant_index] == 1 else 'GOOD'}")

    # Print applicant's raw feature values for teaching
    print(f"\n  Applicant #{applicant_index} raw features:")
    for col in X_df.columns:
        print(f"    {col:<25}: {X_df.iloc[applicant_index][col]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[applicant_index], max_display=15, show=False)
    plt.title(f"Why Was Applicant #{applicant_index} Flagged as High Risk?\n"
              f"True Label: {'BAD' if y_arr[applicant_index] == 1 else 'GOOD'}",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"shap_waterfall_applicant_{applicant_index}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return applicant_index


# ─── Plot 4: Approved Applicant Waterfall ──────────────────────────────────

def plot_approved_waterfall(shap_values, X_df, y_test):
    """Waterfall for a correctly approved good applicant — shows contrast."""
    y_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    risk_scores = shap_values.values.sum(axis=1) + shap_values.base_values
    predicted_good = (1 / (1 + np.exp(-risk_scores))) <= 0.5
    true_good = y_arr == 0
    true_negatives = np.where(predicted_good & true_good)[0]

    if len(true_negatives) == 0:
        print("  No true negative found — skipping approved waterfall.")
        return

    idx = true_negatives[np.argmin(risk_scores[true_negatives])]
    print(f"\nGenerating: Waterfall Plot for Approved Applicant #{idx}...")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f"Why Was Applicant #{idx} Approved?\nTrue Label: GOOD",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"shap_waterfall_approved_{idx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── SHAP Summary Text (for API / adverse action notice) ───────────────────

def generate_adverse_action_notice(shap_values, X_df, applicant_index,
                                   feature_names, top_n=3):
    """
    Generate a plain-English explanation for why an applicant was rejected.
    This is exactly what banks must provide under fair lending regulations.
    """
    sv = shap_values.values[applicant_index]

    # Positive SHAP = pushed toward BAD (rejection reason)
    # Negative SHAP = pushed toward GOOD (approval reason)
    factor_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP":    sv
    }).sort_values("SHAP", ascending=False)

    rejection_reasons = factor_df[factor_df["SHAP"] > 0].head(top_n)
    approval_factors  = factor_df[factor_df["SHAP"] < 0].head(top_n)

    print(f"\n{'-'*55}")
    print(f"  ADVERSE ACTION NOTICE -- Applicant #{applicant_index}")
    print(f"{'-'*55}")
    print("  Your loan application requires further review.")
    print("  Principal reasons for elevated risk assessment:\n")
    for _, row in rejection_reasons.iterrows():
        print(f"  [X] {row['Feature']:<25}  (risk impact: +{row['SHAP']:.4f})")
    print("\n  Factors in your favour:\n")
    for _, row in approval_factors.iterrows():
        print(f"  [+] {row['Feature']:<25}  (risk impact: {row['SHAP']:.4f})")
    print(f"{'-'*55}\n")

    return rejection_reasons, approval_factors


# ─── Main ──────────────────────────────────────────────────────────────────

def run_explanations():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("Loading model artifacts...")
    model, preprocessor, feature_names = load_artifacts()

    print("Loading and preprocessing data...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     _, _) = prepare_data(DATA_PATH)

    # Compute SHAP on the TEST set (what the model has never seen)
    explainer, shap_values, X_df = compute_shap_values(model, X_test, feature_names)

    # Plot 1: Global bar chart
    importance_df = plot_global_bar(shap_values, feature_names)

    # Plot 2: Beeswarm
    plot_beeswarm(shap_values)

    # Plot 3: Rejected applicant waterfall
    bad_idx = plot_waterfall(shap_values, X_df, y_test)

    # Plot 4: Approved applicant waterfall (contrast)
    plot_approved_waterfall(shap_values, X_df, y_test)

    # Generate plain-English adverse action notice
    generate_adverse_action_notice(
        shap_values, X_df, bad_idx, feature_names
    )

    print("\n" + "="*55)
    print("  TOP 5 MOST IMPORTANT FEATURES (Global SHAP)")
    print("="*55)
    top5 = importance_df.sort_values("Mean |SHAP|", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"  {row['Feature']:<28}  {row['Mean |SHAP|']:.4f}")

    print("\nAll SHAP plots saved to reports/")
    return shap_values, X_df, y_test, feature_names


if __name__ == "__main__":
    run_explanations()
