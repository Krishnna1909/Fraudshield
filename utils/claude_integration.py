import anthropic
import os
from typing import Dict

# Your API key is used automatically — recruiters don't need to enter anything
_CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "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"))


def generate_fraud_report(
    features: Dict,
    fraud_prob: float,
    is_fraud: bool,
    shap_values: Dict,
    api_key: str = None,  # kept for backward compatibility, no longer used
) -> str:
    """
    Use Claude API to generate a detailed fraud investigation report.
    """

    shap_summary = "\n".join([
        f"  - {feature}: {value:+.3f} impact"
        for feature, value in shap_values.items()
    ])

    feature_summary = f"""
    - Amount: ₹{features['amount']:,.2f}
    - Time: {features['hour']:02d}:00 hours
    - Days since last transaction: {features['days_since_last']}
    - Average transaction (7 days): ₹{features['avg_amount_7d']:,.2f}
    - Transactions in last 24h: {features['num_transactions_24h']}
    - Foreign transaction: {'Yes' if features['foreign_transaction'] else 'No'}
    - Card type: {features['card_type']}
    - Merchant category: {features['merchant_category']}
    - Weekend transaction: {'Yes' if features['is_weekend'] else 'No'}
    """

    decision = "FLAGGED AS FRAUDULENT" if is_fraud else "CLEARED AS LEGITIMATE"

    prompt = f"""You are a senior fraud analyst at a major Indian bank.
An AI model has analyzed a credit card transaction and produced the following results.

TRANSACTION DETAILS:
{feature_summary}

MODEL OUTPUT:
- Fraud Probability: {fraud_prob*100:.1f}%
- Decision: {decision}

KEY RISK FACTORS (SHAP Analysis):
{shap_summary}

Write a professional fraud investigation report with the following sections:
1. Executive Summary - One sentence verdict
2. Risk Assessment - What specific patterns triggered the alert
3. Behavioral Analysis - How this transaction compares to normal behavior
4. Recommended Action - What the bank should do (approve/block/call customer)
5. Prevention Note - One tip to prevent similar fraud

Keep it concise, professional, and actionable. Use bullet points where appropriate.
Format in clean markdown. Do not use ** for bold — use plain text only.
Do not leave any incomplete bullet points or sentences."""

    try:
        message = _CLIENT.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

    except anthropic.AuthenticationError:
        return "❌ Invalid API key. Please check your Anthropic API key."
    except anthropic.RateLimitError:
        return "⚠️ Rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"⚠️ Could not generate report: {str(e)}"


def generate_batch_summary(
    total: int,
    flagged: int,
    fraud_rate: float,
    top_merchants: Dict,
    top_hours: Dict,
    api_key: str = None,  # kept for backward compatibility, no longer used
) -> str:
    """
    Generate an executive summary for batch fraud analysis.
    """

    prompt = f"""You are a fraud analytics manager reviewing a batch transaction report.

BATCH ANALYSIS RESULTS:
- Total Transactions Analyzed: {total:,}
- Flagged as Fraudulent: {flagged:,}
- Overall Fraud Rate: {fraud_rate:.2f}%
- Highest Risk Merchants: {top_merchants}
- Peak Fraud Hours: {top_hours}

Write a 3-paragraph executive summary covering:
1. Overall fraud landscape and severity assessment
2. Key patterns and hotspots identified
3. Strategic recommendations for fraud prevention

Be data-driven, concise, and actionable."""

    try:
        message = _CLIENT.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ Could not generate summary: {str(e)}"