"""
FraudShield — utils/model.py
Loads the trained XGBoost model and exposes the exact API that app.py expects:
    load_model()          → dict of artefacts
    predict_transaction() → prediction result dict
"""

from __future__ import annotations

import os
import joblib
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Numpy 2.0 / SHAP compatibility patch ─────────────────────────────────────
# Mirrors the same patch in train.py — must run before any shap import.
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

# Path to the pkl saved by train.py
MODEL_PATH = Path(__file__).parent.parent / "models" / "xgboost_fraud.pkl"


class ModelNotTrainedError(RuntimeError):
    """Raised when the model .pkl file is missing."""
    pass


@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """
    Load artefacts from models/xgboost_fraud.pkl.
    Cached with lru_cache — disk is only read once per process.

    Returns dict with keys:
        model, scaler, explainer, feature_names,
        shap_values, X_sample, metrics
    """
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Run `python train.py` first (needs data/creditcard.csv)."
        )
    artifacts = joblib.load(MODEL_PATH)
    return artifacts


def is_model_loaded() -> bool:
    """Return True if the model pkl exists on disk."""
    return MODEL_PATH.exists()


def predict_transaction(transaction: dict[str, float | int]) -> dict[str, Any]:
    """
    Score a single transaction dict.

    Parameters
    ----------
    transaction : dict
        Keys must include all feature names used during training
        (V1–V28, Amount, Time — same columns as creditcard.csv minus 'Class').

    Returns
    -------
    dict with keys:
        fraud_probability   float   0–1
        is_fraud            bool
        risk_level          str     "Low" | "Medium" | "High" | "Critical"
        confidence          str     human-readable confidence label
        shap_values         np.ndarray  shape (n_features,)
        feature_names       list[str]
        top_features        list[dict]  top-3 SHAP drivers
    """
    arts      = load_model()
    model     = arts["model"]
    scaler    = arts["scaler"]
    feat_names = arts["feature_names"]

    # Build input row — Amount & Time need scaling; V-cols are already scaled
    row = pd.DataFrame([{k: transaction.get(k, 0.0) for k in feat_names}])

    # Re-scale Amount and Time columns (same scaler fitted in train.py)
    row_scaled = row.copy()
    row_scaled[["Amount", "Time"]] = scaler.transform(row[["Amount", "Time"]])

    # Predict
    prob     = float(model.predict_proba(row_scaled.values)[0, 1])
    is_fraud = prob >= 0.5
    risk     = _risk_level(prob)

    # Per-row SHAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = arts["explainer"]
        shap_vals = explainer.shap_values(row_scaled.values)[0]

    # Top-3 drivers by absolute SHAP impact
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [
        {
            "feature": feat_names[i],
            "value":   float(row.iloc[0][feat_names[i]]),
            "impact":  float(shap_vals[i]),
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


def get_model_metrics() -> dict[str, Any]:
    """Return the evaluation metrics stored at training time."""
    return load_model().get("metrics", {})


def get_shap_summary() -> dict[str, Any]:
    """Return the pre-computed SHAP sample for global feature importance."""
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