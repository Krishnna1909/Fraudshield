"""
FraudShield — utils/model.py
Loads the trained XGBoost model and exposes the API app.py expects:
    load_model()          -> dict of artefacts
    predict_transaction()  -> single-transaction prediction result dict
    predict_batch()         -> vectorized predictions for a DataFrame of transactions
    get_model_metrics()      -> real evaluation metrics computed at training time
    get_confusion_matrix()    -> real confusion matrix (for Tab 2)
    get_roc_curve()            -> real ROC curve points (for Tab 2)
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

# ── Numpy 2.0 / SHAP compatibility patch ─────────────────────────────────────
# Mirrors the same patch in train.py — must run before any shap import.
# (object/str are checked with warnings suppressed: on some numpy versions
# `hasattr(np, "object")` itself emits a harmless FutureWarning.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _alias, _target in [
        ("bool",    bool),
        ("int",     int),
        ("float",   float),
        ("complex", complex),
        ("object",  object),
        ("str",     str),
    ]:
        if not hasattr(np, _alias):
            setattr(np, _alias, _target)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_fraud.pkl"


class ModelNotTrainedError(RuntimeError):
    """Raised when the model .pkl file is missing."""
    pass


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """
    Load artefacts from models/xgboost_fraud.pkl.
    Cached with lru_cache — disk is only read once per process.
    """
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Run `python train.py` first (needs data/creditcard.csv)."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_explainer():
    """
    Build the SHAP TreeExplainer fresh from the loaded model, rather than
    unpickling a saved one. TreeExplainer construction is fast (well under
    a second for this model size), and building it fresh avoids a fragile
    cross-environment pickle of numba-compiled internals that can break
    when the numba/llvmlite version differs between where the model was
    trained and where the app is deployed. Cached with lru_cache so it's
    only built once per running process.
    """
    import shap
    return shap.TreeExplainer(load_model()["model"])


def is_model_loaded() -> bool:
    return MODEL_PATH.exists()


def _prepare_row(transaction: dict[str, Any], arts: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row, model-ready (encoded + scaled) DataFrame from a raw feature dict."""
    feat_names = arts["feature_names"]
    numeric_cols = arts["numeric_cols"]
    categorical_cols = arts["categorical_cols"]
    encoders = arts["encoders"]

    row = {}
    for col in feat_names:
        if col in categorical_cols:
            raw_val = transaction.get(col, encoders[col].classes_[0])
            # Unseen category -> fall back to the first known class rather than crashing
            if raw_val not in set(encoders[col].classes_):
                raw_val = encoders[col].classes_[0]
            row[col] = encoders[col].transform([raw_val])[0]
        else:
            row[col] = transaction.get(col, 0.0)

    row_df = pd.DataFrame([row])[feat_names]
    row_df[numeric_cols] = arts["scaler"].transform(row_df[numeric_cols])
    return row_df


def predict_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single transaction dict. Keys should match the Streamlit form fields:
    amount, hour, days_since_last, avg_amount_7d, num_transactions_24h,
    foreign_transaction, is_weekend, card_type, merchant_category.

    Returns
    -------
    dict with keys:
        fraud_probability, is_fraud, risk_level, confidence,
        shap_values, feature_names, top_features
    """
    arts       = load_model()
    model      = arts["model"]
    feat_names = arts["feature_names"]

    row_scaled = _prepare_row(transaction, arts)

    prob     = float(model.predict_proba(row_scaled.values)[0, 1])
    is_fraud = prob >= 0.5
    risk     = _risk_level(prob)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = get_explainer()
        shap_vals = explainer.shap_values(row_scaled.values)[0]

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [
        {
            "feature":   feat_names[i],
            "value":     transaction.get(feat_names[i], row_scaled.iloc[0][feat_names[i]]),
            "impact":    float(shap_vals[i]),
            "direction": "↑ fraud" if shap_vals[i] > 0 else "↓ fraud",
        }
        for i in top_idx
    ]

    return {
        "fraud_probability": prob,
        "is_fraud":          is_fraud,
        "risk_level":        risk,
        "confidence":        _confidence_label(prob),
        "shap_values":       shap_vals,
        "feature_names":     feat_names,
        "top_features":      top_features,
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of transactions using the REAL trained model (vectorized —
    no per-row Python loop, so this scales to large uploaded CSVs).

    Required columns: amount, hour, days_since_last, avg_amount_7d,
    num_transactions_24h, foreign_transaction, is_weekend, card_type,
    merchant_category. Missing columns are filled with safe defaults
    (0 for numeric/binary, the most common training category for categoricals)
    and reported back to the caller via the returned DataFrame's attrs.

    Returns the original DataFrame with two new columns:
        fraud_probability : float, 0-1
        prediction         : int, 1 if fraud_probability >= 0.5 else 0
    """
    arts             = load_model()
    model            = arts["model"]
    feat_names       = arts["feature_names"]
    numeric_cols     = arts["numeric_cols"]
    categorical_cols = arts["categorical_cols"]
    encoders         = arts["encoders"]

    work = df.copy()
    missing_cols = [c for c in feat_names if c not in work.columns]

    for col in feat_names:
        if col not in work.columns:
            if col in categorical_cols:
                work[col] = encoders[col].classes_[0]
            else:
                work[col] = 0.0

    X = pd.DataFrame(index=work.index)
    for col in feat_names:
        if col in categorical_cols:
            known = set(encoders[col].classes_)
            safe_col = work[col].where(work[col].isin(known), encoders[col].classes_[0])
            X[col] = encoders[col].transform(safe_col)
        else:
            X[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    X = X[feat_names]
    X[numeric_cols] = arts["scaler"].transform(X[numeric_cols])

    probs = model.predict_proba(X.values)[:, 1]

    result = df.copy()
    result["fraud_probability"] = probs
    result["prediction"] = (probs >= 0.5).astype(int)
    result.attrs["missing_columns"] = missing_cols
    return result


def get_model_metrics() -> dict[str, Any]:
    """Real evaluation metrics computed at training time (train.py)."""
    return load_model().get("metrics", {})


def get_confusion_matrix() -> np.ndarray:
    m = get_model_metrics()
    return np.array(m.get("confusion_matrix", [[0, 0], [0, 0]]))


def get_roc_curve() -> tuple[np.ndarray, np.ndarray]:
    m = get_model_metrics()
    return np.array(m.get("roc_fpr", [])), np.array(m.get("roc_tpr", []))


def get_categorical_options() -> dict[str, list[str]]:
    return load_model().get("categorical_options", {})


def get_shap_summary() -> dict[str, Any]:
    arts = load_model()
    return {
        "shap_values":   arts["shap_values"],
        "X_sample":      arts["X_sample"],
        "feature_names": arts["feature_names"],
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def _confidence_label(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance > 0.35:
        return "Very High"
    elif distance > 0.20:
        return "High"
    elif distance > 0.10:
        return "Moderate"
    else:
        return "Low"