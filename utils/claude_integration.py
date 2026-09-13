import os
from typing import Dict

import anthropic
import streamlit as st


def _get_api_key(explicit_key: str = None) -> str | None:
    """
    Get the Anthropic API key from:
    1. Explicitly provided key
    2. Streamlit Secrets root-level key
    3. Streamlit Secrets [anthropic] section
    4. Environment variable
    """

    # 1. Explicit key
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # 2. Root-level Streamlit secret
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")

        if key:
            return str(key).strip()

    except Exception:
        pass

    # 3. [anthropic] section
    try:
        section = st.secrets.get("anthropic")

        if section:
            key = section.get("api_key")

            if key:
                return str(key).strip()

    except Exception:
        pass

    # 4. Environment variable
    key = os.getenv("ANTHROPIC_API_KEY")

    if key:
        return key.strip()

    return None


def generate_fraud_report(
    features: Dict,
    fraud_prob: float,
    is_fraud: bool,
    shap_values: Dict,
    api_key: str = None,
) -> str:
    """
    Generate an AI-powered fraud analysis using Claude.
    """

    api_key = _get_api_key(api_key)

    if not api_key:
        return (
            "⚠️ Anthropic API key is not configured. "
            "Add ANTHROPIC_API_KEY in "
            "Streamlit Cloud → App settings → Secrets."
        )

    client = anthropic.Anthropic(
        api_key=api_key
    )

    risk_level = (
        "Critical"
        if fraud_prob >= 0.75
        else "High"
        if fraud_prob >= 0.50
        else "Medium"
        if fraud_prob >= 0.25
        else "Low"
    )

    feature_explanations = []

    for feature, value in shap_values.items():

        direction = (
            "increases"
            if value > 0
            else "decreases"
        )

        feature_explanations.append(
            f"- {feature}: contribution={value:.4f} "
            f"({direction} fraud risk)"
        )

    shap_text = "\n".join(
        feature_explanations
    )

    prompt = f"""
You are a financial fraud detection analyst.

Analyze the following transaction:

Transaction features:
{features}

Fraud probability: {fraud_prob:.2%}
Prediction: {"FRAUDULENT" if is_fraud else "LEGITIMATE"}
Risk level: {risk_level}

Feature contributions:
{shap_text}

Provide a concise professional fraud analysis containing:

1. Overall assessment
2. Main factors influencing the prediction
3. Why the transaction appears risky or safe
4. Recommended action

Do not invent information that is not present in the transaction data.
"""

    try:

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return message.content[0].text

    except anthropic.AuthenticationError:

        return (
            "❌ Invalid API key. "
            "Please check your Anthropic API key."
        )

    except anthropic.RateLimitError:

        return (
            "⚠️ Rate limit reached. "
            "Please wait a moment and try again."
        )

    except Exception as e:

        return (
            f"⚠️ Could not generate report: {str(e)}"
        )


def generate_batch_summary(
    total_transactions: int,
    fraudulent_transactions: int,
    fraud_rate: float,
    api_key: str = None,
) -> str:
    """
    Generate an AI-powered summary for batch analysis.
    """

    api_key = _get_api_key(api_key)

    if not api_key:
        return (
            "⚠️ Anthropic API key is not configured. "
            "Add ANTHROPIC_API_KEY in "
            "Streamlit Cloud → App settings → Secrets."
        )

    client = anthropic.Anthropic(
        api_key=api_key
    )

    prompt = f"""
You are a financial fraud analytics expert.

Summarize the following batch fraud detection results:

Total transactions: {total_transactions}
Fraudulent transactions: {fraudulent_transactions}
Fraud rate: {fraud_rate:.2%}

Provide:

1. Overall risk assessment
2. Key observations
3. Business implications
4. Recommended actions

Keep the response concise and professional.
Do not invent information.
"""

    try:

        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return message.content[0].text

    except anthropic.AuthenticationError:

        return (
            "❌ Invalid API key. "
            "Please check your Anthropic API key."
        )

    except anthropic.RateLimitError:

        return (
            "⚠️ Rate limit reached. "
            "Please wait a moment and try again."
        )

    except Exception as e:

        return (
            f"⚠️ Could not generate report: {str(e)}"
        )