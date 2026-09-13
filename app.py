import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.model import (
    load_model, predict_transaction, predict_batch,
    get_model_metrics, get_confusion_matrix, get_roc_curve,
    get_categorical_options, is_model_loaded,
)
from utils.claude_integration import generate_fraud_report
import warnings
warnings.filterwarnings('ignore')

# The Claude API key is read directly by utils/claude_integration.py from the
# ANTHROPIC_API_KEY environment variable (Streamlit Secrets sets this as an
# env var in deployment). We just check it's present so the UI can warn
# early instead of failing silently deep inside a report-generation call.
def _get_anthropic_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    try:
        section = st.secrets.get("anthropic")
        if section:
            key = section.get("api_key")
            if key:
                return str(key).strip()
    except Exception:
        pass

    return None


_HAS_CLAUDE_KEY = bool(_get_anthropic_key())
st.write("DEBUG: Anthropic secret detected =", _HAS_CLAUDE_KEY)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .fraud-alert {
        background-color: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .safe-alert {
        background-color: #2ed573;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .report-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=80)
    st.title("FraudShield")
    st.caption("AI-Powered Fraud Detection")
    st.divider()

    st.subheader("⚙️ Detection Settings")
    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Transactions above this threshold are flagged as fraud"
    )

    st.divider()
    st.markdown("**About**")
    st.caption(
        "FraudShield uses XGBoost + SHAP explainability "
        "combined with Claude AI to detect and investigate "
        "fraudulent credit card transactions."
    )

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ FraudShield</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Explainable AI Fraud Detection System powered by XGBoost + Claude AI</p>',
    unsafe_allow_html=True
)

if not is_model_loaded():
    st.error(
        "⚠️ Model file not found at `models/xgboost_fraud.pkl`. "
        "Run `python train.py` first (needs `data/creditcard.csv`)."
    )
    st.stop()

if not _HAS_CLAUDE_KEY:
    st.warning(
        "⚠️ ANTHROPIC_API_KEY is not set — the app will run and predictions "
        "will work, but the AI Investigation Report on the Transaction Analyzer "
        "tab will show an error instead of a generated report."
    )

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Transaction Analyzer", "📊 Model Insights", "📁 Batch Analysis"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Single Transaction Analyzer
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Single Transaction")
    st.caption("Enter transaction details below to check for fraud")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Details**")
        amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=150.0, step=10.0)
        hour = st.slider("Hour of Transaction", 0, 23, 14)
        days_since_last = st.number_input("Days Since Last Transaction", min_value=0, value=2)

    with col2:
        st.markdown("**Behavioral Features**")
        avg_amount_7d = st.number_input("Avg Transaction (Last 7 Days)", min_value=0.0, value=120.0)
        num_transactions_24h = st.number_input("Transactions in Last 24h", min_value=0, value=3)
        foreign_transaction = st.selectbox("Foreign Transaction?", ["No", "Yes"])

    cat_options = get_categorical_options()
    with col3:
        st.markdown("**Card Details**")
        card_type = st.selectbox("Card Type", cat_options.get("card_type", ["Visa", "Mastercard", "Amex", "RuPay"]))
        merchant_category = st.selectbox("Merchant Category", cat_options.get("merchant_category", [
            "Retail", "Food & Dining", "Travel", "Entertainment",
            "Electronics", "Online Shopping", "ATM Withdrawal"
        ]))
        is_weekend = st.checkbox("Weekend Transaction")

    st.divider()

    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction..."):

            # Build feature dict
            features = {
                "amount": amount,
                "hour": hour,
                "days_since_last": days_since_last,
                "avg_amount_7d": avg_amount_7d,
                "num_transactions_24h": num_transactions_24h,
                "foreign_transaction": 1 if foreign_transaction == "Yes" else 0,
                "card_type": card_type,
                "merchant_category": merchant_category,
                "is_weekend": 1 if is_weekend else 0,
            }

            # ── Get prediction (returns a dict) ──────────────────────────
            result          = predict_transaction(features)
            fraud_prob      = result["fraud_probability"]
            is_fraud        = fraud_prob >= threshold
            shap_vals_arr   = result["shap_values"]
            feature_names   = result["feature_names"]
            top_features    = result["top_features"]

            # Build a name→value dict for the SHAP bar chart
            shap_values = dict(zip(feature_names, shap_vals_arr.tolist()))
            # ─────────────────────────────────────────────────────────────

            # ── Result Banner ──
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                if is_fraud:
                    st.markdown(
                        f'<div class="fraud-alert">⚠️ FRAUD DETECTED — {fraud_prob*100:.1f}% Probability</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="safe-alert">✅ TRANSACTION SAFE — {fraud_prob*100:.1f}% Fraud Probability</div>',
                        unsafe_allow_html=True
                    )

            st.divider()

            # ── Metrics ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
            m2.metric("Amount", f"₹{amount:,.2f}")
            m3.metric("Risk Level", result["risk_level"])
            m4.metric("Decision", "BLOCK" if is_fraud else "APPROVE")

            # ── SHAP Feature Importance Chart ──
            st.subheader("🔬 Why did the model decide this?")
            fig_shap = go.Figure(go.Bar(
                x=list(shap_values.values()),
                y=list(shap_values.keys()),
                orientation='h',
                marker=dict(
                    color=['#ff4757' if v > 0 else '#2ed573'
                           for v in shap_values.values()]
                )
            ))
            fig_shap.update_layout(
                title="Feature Impact on Fraud Score (SHAP Values)",
                xaxis_title="Impact (Red = Increases Fraud Risk)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # ── Top 3 drivers ──
            st.subheader("🎯 Top Risk Drivers")
            for f in top_features:
                direction_color = "🔴" if f["impact"] > 0 else "🟢"
                st.markdown(
                    f"{direction_color} **{f['feature']}** = `{f['value']}` "
                    f"→ SHAP impact: `{f['impact']:+.4f}` {f['direction']}"
                )

            # ── Claude AI Investigation Report (always shown) ──
            st.subheader("📋 AI Investigation Report")
            with st.spinner("Claude is generating fraud investigation report..."):
                report = generate_fraud_report(
                    features=features,
                    fraud_prob=fraud_prob,
                    is_fraud=is_fraud,
                    shap_values=shap_values
                )
            st.markdown(report)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Performance & Insights")
    st.caption("All numbers below are computed live from the saved model artifact (models/xgboost_fraud.pkl) — nothing on this tab is hardcoded.")

    metrics = get_model_metrics()
    cm = get_confusion_matrix()
    fpr, tpr = get_roc_curve()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix** (real test-set results)")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Not Fraud", "Fraud"],
            y=["Not Fraud", "Fraud"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown(f"**ROC Curve** (AUC = {metrics.get('auc', 0):.4f})")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"XGBoost (AUC={metrics.get('auc', 0):.3f})", line=dict(color="#667eea", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=300,
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("**Model Performance Metrics** (default 0.5 threshold, real test-set evaluation)")
    metrics_df = pd.DataFrame({
        "Metric": ["AUC-ROC", "Precision", "Recall", "F1-Score", "Accuracy"],
        "Value": [
            f"{metrics.get('auc', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('accuracy', 0):.4%}",
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    rec_thresh = metrics.get("recommended_threshold")
    if rec_thresh is not None:
        st.info(
            f"💡 Because fraud is extremely rare in the test set "
            f"({metrics.get('fraud_rate_test', 0):.3%} of transactions), the default 0.5 "
            f"threshold trades away precision. The F1-optimal threshold found during training "
            f"is **{rec_thresh:.2f}**, giving Precision={metrics.get('precision_at_recommended', 0):.3f}, "
            f"Recall={metrics.get('recall_at_recommended', 0):.3f}, F1={metrics.get('f1_at_recommended', 0):.3f}. "
            f"Try setting the sidebar threshold near this value."
        )
    st.caption(
        f"Trained on {metrics.get('n_train', 0):,} rows (post-SMOTE) · "
        f"evaluated on {metrics.get('n_test', 0):,} untouched, real-world-imbalanced test rows."
    )

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📁 Batch Transaction Analysis")
    st.caption("Upload a CSV file with multiple transactions to analyze all at once")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV should contain transaction features"
    )

    st.caption(
        "Expected columns: amount, hour, days_since_last, avg_amount_7d, "
        "num_transactions_24h, foreign_transaction, is_weekend, card_type, "
        "merchant_category. Any missing column is filled with a safe default "
        "and flagged below — it isn't silently ignored."
    )
    try:
        sample_csv = pd.read_csv("models/sample_batch_transactions.csv").to_csv(index=False)
        st.download_button("📥 Download a sample CSV to try", sample_csv, "sample_batch_transactions.csv", "text/csv")
    except FileNotFoundError:
        pass

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} transactions")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🔍 Analyze All Transactions", type="primary"):
            with st.spinner("Analyzing batch..."):
                scored = predict_batch(df)
                missing = scored.attrs.get("missing_columns", [])
                fraud_probs = scored["fraud_probability"].values
                scored["prediction"] = (fraud_probs >= threshold).astype(int)
                scored["risk_level"] = pd.cut(
                    fraud_probs,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True,
                )

            if missing:
                st.warning(
                    f"⚠️ These expected columns were missing from your CSV and were "
                    f"filled with defaults, which reduces prediction accuracy: {', '.join(missing)}"
                )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{len(scored):,}")
            col2.metric("Flagged as Fraud", f"{scored['prediction'].sum():,}")
            col3.metric("Fraud Rate", f"{scored['prediction'].mean()*100:.2f}%")
            col4.metric("Safe Transactions", f"{(1-scored['prediction']).sum():,}")

            fig_dist = px.histogram(
                scored, x='fraud_probability',
                nbins=50,
                title="Distribution of Fraud Probabilities (real model output)",
                color_discrete_sequence=["#667eea"]
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.subheader("Flagged Transactions")
            flagged = scored[scored['prediction'] == 1].sort_values('fraud_probability', ascending=False)
            if flagged.empty:
                st.info("No transactions crossed the current fraud threshold.")
            else:
                st.dataframe(flagged.head(20), use_container_width=True)
                csv = flagged.to_csv(index=False)
                st.download_button(
                    "📥 Download Flagged Transactions",
                    csv,
                    "flagged_transactions.csv",
                    "text/csv"
                )
    else:
        st.info("👆 Upload a CSV file to get started with batch analysis")