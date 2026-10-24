"""
preprocessing.py
----------------
Handles all data loading, cleaning, encoding, and train/val/test splitting
for the German Credit Risk dataset.

IMPORTANT RULE: Encoders and scalers are ALWAYS fit on training data only.
They are then applied (not re-fit) to validation and test data.
This prevents data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ─── Column Definitions ────────────────────────────────────────────────────

# Columns where missing = "no account" (informative missingness)
FILL_NONE_COLS = ["Saving accounts", "Checking account"]

# Ordered categories for ordinal encoding
# (little < moderate < quite rich < rich < none means no account)
SAVING_ORDER    = ["none", "little", "moderate", "quite rich", "rich"]
CHECKING_ORDER  = ["none", "little", "moderate", "rich"]

# Columns we one-hot encode (no natural order)
ONEHOT_COLS = ["Sex", "Housing", "Purpose"]

# Columns we ordinal encode (natural order exists)
ORDINAL_COLS = ["Saving accounts", "Checking account"]

# Numeric columns to scale
NUMERIC_COLS = ["Age", "Credit amount", "Duration"]

# Already numeric — treat as-is (0=unskilled, 3=highly skilled)
PASSTHROUGH_COLS = ["Job"]

# Target column
TARGET = "Risk"


# ─── Load & Clean ──────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV and perform basic cleaning:
    - Drop the unnamed index column
    - Fill informative missing values with 'none'
    - Encode the target as binary (0=good, 1=bad)
    """
    df = pd.read_csv(filepath, index_col=0)

    # Fill missing values in account columns with 'none'
    # Reason: missing likely means the applicant has no such account
    for col in FILL_NONE_COLS:
        df[col] = df[col].fillna("none")

    # Encode target: good -> 0, bad -> 1
    # We use 1 for the risky class (standard convention)
    df[TARGET] = df[TARGET].map({"good": 0, "bad": 1})

    return df


# ─── Split ─────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15,
               random_state: int = 42):
    """
    Split into train / validation / test sets.

    Stratified on TARGET to preserve the 70/30 class ratio in each split.
    random_state=42 ensures reproducibility — same split every time.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: from the remaining data, carve out validation
    # Adjust val_size relative to the remaining data size
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"Train size:      {len(X_train)} rows")
    print(f"Validation size: {len(X_val)} rows")
    print(f"Test size:       {len(X_test)} rows")
    print(f"Train bad%:      {y_train.mean():.1%}")
    print(f"Val   bad%:      {y_val.mean():.1%}")
    print(f"Test  bad%:      {y_test.mean():.1%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Preprocessing Pipeline ────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that applies:
    - OrdinalEncoder to account columns (ordered categories)
    - OneHotEncoder to nominal columns (no order)
    - StandardScaler to numeric columns
    - Passthrough for Job (already numeric 0-3)

    This transformer is fit ONLY on training data, then applied to all splits.
    """

    ordinal_transformer = OrdinalEncoder(
        categories=[SAVING_ORDER, CHECKING_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    onehot_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False   # Return dense array (easier to work with)
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal",  ordinal_transformer,  ORDINAL_COLS),
            ("onehot",   onehot_transformer,   ONEHOT_COLS),
            ("numeric",  numeric_transformer,  NUMERIC_COLS),
            ("passthru", "passthrough",        PASSTHROUGH_COLS),
        ],
        remainder="drop"    # Drop any column not listed above
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract readable feature names after transformation.
    Useful for SHAP plots and debugging.
    """
    feature_names = []

    # Ordinal columns keep their original names
    feature_names += ORDINAL_COLS

    # OneHot columns expand into multiple columns
    ohe = preprocessor.named_transformers_["onehot"]
    feature_names += list(ohe.get_feature_names_out(ONEHOT_COLS))

    # Numeric columns keep their names
    feature_names += NUMERIC_COLS

    # Passthrough columns keep their names
    feature_names += PASSTHROUGH_COLS

    return feature_names


# ─── Main convenience function ─────────────────────────────────────────────

def prepare_data(filepath: str):
    """
    Full pipeline: load -> clean -> split -> preprocess.

    Returns:
        X_train_proc, X_val_proc, X_test_proc : processed numpy arrays
        y_train, y_val, y_test                : target Series
        preprocessor                           : fitted transformer (save this!)
        feature_names                          : list of feature names
    """
    df = load_data(filepath)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor()

    # FIT on training data ONLY, then transform all sets
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc   = preprocessor.transform(X_val)
    X_test_proc  = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    print(f"\nFeature names after encoding ({len(feature_names)} total):")
    for i, name in enumerate(feature_names):
        print(f"  [{i:02d}] {name}")

    return (X_train_proc, X_val_proc, X_test_proc,
            y_train, y_val, y_test,
            preprocessor, feature_names)
