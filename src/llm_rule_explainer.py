import os
from google import genai


def explain_rule_set_with_llm(rule_set):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "llm_status": "No API key detected",
            "llm_explanation": None,
        }

    try:
        client = genai.Client(api_key=api_key)

        rules_text = ""

        for feature, rule in rule_set.items():
            rules_text += f"""
Feature: {feature}
Threshold: {rule["threshold"]}
Risk direction: {rule["risk_direction"]}
Rule quality: {rule["rule_quality"]}
Base strength: {rule["base_strength"]}
"""

        prompt = f"""
You are assisting in an explainable and auditable credit-scoring research system.

The following candidate argumentation rules were automatically generated from dataset statistics.

{rules_text}

Explain:
1. Which rules appear financially meaningful.
2. Which rules appear statistically weak.
3. Governance concerns.
4. Which rules require human review.
5. The difference between statistical association and governance approval.

Keep the answer professional and concise.
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        return {
            "llm_status": "LLM explanation generated",
            "llm_explanation": response.text,
        }

    except Exception as e:
        return {
            "llm_status": f"LLM unavailable: {str(e)}",
            "llm_explanation": (
                "The LLM explanation could not be generated because the external API "
                "was unavailable or quota-limited. The system still produced candidate "
                "rules using dataset metrics, rule quality labels, and template-based "
                "governance notes. These outputs remain auditable and can be reviewed "
                "without relying on the LLM."
            ),
        }
