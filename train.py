"""
FraudShield — Model Training Pipeline
======================================
1. Place creditcard.csv in the data/ folder
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Activate your venv
3. Run: python train.py
4. Model saved to models/xgboost_fraud.pkl

After training, utils/model.py will load the saved model automatically.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Numpy 2.0 / SHAP compatibility patch ─────────────────────────────────────
# Must happen BEFORE importing shap. Numpy 2.0 removed np.bool, np.int, etc.
# SHAP still references them internally — this restores the aliases safely.
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

import shap  # noqa: E402 — must come after the patch above
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("FraudShield — Model Training Pipeline")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH = "data/creditcard.csv"

if not os.path.exists(DATA_PATH):
    print(f"\n❌ Dataset not found at {DATA_PATH}")
    print("Please download from:")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("and place creditcard.csv in the data/ folder")
    exit(1)

print(f"\n📂 Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Fraud rate: {df['Class'].mean():.4%}")

# ── 2. EDA Plot — Amount Distribution ────────────────────────────────────────
os.makedirs("models", exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df[df["Class"] == 0]["Amount"].clip(upper=500).hist(bins=50, ax=ax[0], color="#6366f1", alpha=0.7)
ax[0].set_title("Legit Transaction Amounts")
ax[0].set_xlabel("Amount ($)")

df[df["Class"] == 1]["Amount"].clip(upper=500).hist(bins=50, ax=ax[1], color="#ef4444", alpha=0.7)
ax[1].set_title("Fraud Transaction Amounts")
ax[1].set_xlabel("Amount ($)")

plt.tight_layout()
plt.savefig("models/eda_amount_distribution.png", dpi=100, bbox_inches="tight")
plt.close()
print("   EDA plot saved → models/eda_amount_distribution.png")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("\n🔧 Preparing features ...")
feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols].copy()
y = df["Class"].copy()

# Scale Amount and Time (V1–V28 are already PCA-transformed)
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. SMOTE Oversampling ─────────────────────────────────────────────────────
print("\n⚖️  Applying SMOTE ...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"   After SMOTE — fraud: {(y_res == 1).sum():,}  |  legit: {(y_res == 0).sum():,}")

# ── 6. Train XGBoost ──────────────────────────────────────────────────────────
print("\n🚀 Training XGBoost ...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_res, y_res,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
print("\n📊 Evaluation:")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)
ap      = average_precision_score(y_test, y_proba)

print(f"   ROC-AUC : {auc:.4f}")
print(f"   Avg Precision: {ap:.4f}")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"], ax=ax)
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
ax.set_title(f"Confusion Matrix (AUC={auc:.4f})")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()
print("   Confusion matrix saved → models/confusion_matrix.png")

# ── 8. SHAP Explainer ─────────────────────────────────────────────────────────
print("\n🔍 Computing SHAP values (sample of 500 rows) ...")
sample_idx = X_test.sample(500, random_state=42).index
X_sample   = X_test.loc[sample_idx]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# ── 9. Save all artefacts ─────────────────────────────────────────────────────
print("\n💾 Saving model artefacts ...")
artifacts = {
    "model":         model,
    "scaler":        scaler,
    "explainer":     explainer,
    "feature_names": feature_cols,
    "shap_values":   shap_values,
    "X_sample":      X_sample,
    "metrics": {
        "auc":              auc,
        "avg_precision":    ap,
        "n_train":          len(X_res),
        "n_test":           len(X_test),
        "fraud_rate_test":  float(y_test.mean()),
    },
}

MODEL_OUT = "models/xgboost_fraud.pkl"
joblib.dump(artifacts, MODEL_OUT)
print(f"\n✅ Done! Model saved → {MODEL_OUT}")
print(f"   ROC-AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")
print("=" * 60)