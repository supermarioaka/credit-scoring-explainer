import os
from dotenv import load_dotenv
from google import genai


load_dotenv()


def build_ai_explanation_payload(case_summary):
    return {
        "official_decision": case_summary["official_decision"],
        "probability_of_default": case_summary["probability_of_default"],
        "linear_score": case_summary["linear_score"],
        "argumentation_risk_signal": case_summary["argumentation_risk_signal"],
        "approve_argument_strength": case_summary["approve_argument_strength"],
        "reject_argument_strength": case_summary["reject_argument_strength"],
        "dominant_reasoning_side": case_summary["dominant_reasoning_side"],
        "main_adverse_drivers": case_summary["main_adverse_drivers"],
        "mitigating_factors": case_summary["mitigating_factors"],
        "why": case_summary["why"],
        "why_not": case_summary["why_not"],
    }


def generate_plain_language_explanation(case_summary):
    payload = build_ai_explanation_payload(case_summary)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "AI explanation is unavailable because GEMINI_API_KEY was not found. "
            "The mathematical WHY / WHY-NOT explanation remains valid."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are explaining a mathematical and auditable credit scoring decision.

Rules:
- Do not change the official decision.
- Do not invent new features or reasons.
- Explain only the provided mathematical output.
- Keep the explanation clear for a banking analyst.
- Mention that the AI is only explaining the mathematical output, not deciding.

Structured case summary:
{payload}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as error:
        return (
            "AI explanation is currently unavailable because the Gemini API request failed. "
            f"Error: {error}"
        )
