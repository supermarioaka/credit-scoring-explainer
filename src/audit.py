from datetime import datetime


def create_audit_record(
    applicant_data: dict,
    probability_of_default: float,
    business_decision: str,
    arguments: list[dict],
    strength_summary: dict,
    linear_score: float | None = None,
    feature_contributions: dict | None = None,
    model_diagnostics: dict | None = None,
    argument_graph: dict | None = None,
    accepted_arguments: list[dict] | None = None,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predictive_layer": {
            "model_type": "Logistic Regression",
            "input_features": list(applicant_data.keys()),
            "linear_score": linear_score,
            "probability_of_default": probability_of_default,
            "business_policy": {
                "approve": "PD < 0.10",
                "review": "0.10 <= PD <= 0.30",
                "reject": "PD > 0.30",
            },
            "business_decision": business_decision,
            "feature_contributions": feature_contributions,
            "model_diagnostics": model_diagnostics,
        },
        "argumentation_layer": {
            "explanation_note": (
                "The argumentation layer uses selected financially interpretable "
                "features and excludes age from the explanation."
            ),
            "strength_formula": "strength = base_strength × activation_strength",
            "base_strength_formula": "base_strength = |beta_j| / max(|beta|)",
            "activation_strength_formula": (
                "activation_strength = sigmoid(distance_from_threshold / scale)"
            ),
            "argument_based_decision": strength_summary["argument_decision"],
            "approve_total": strength_summary["approve_total"],
            "reject_total": strength_summary["reject_total"],
            "arguments": arguments,
            "argument_graph": argument_graph,
            "accepted_arguments": accepted_arguments,
        },
        "audit_statement": (
            "This record stores the applicant inputs, predictive output, business "
            "policy decision, argument-based reasoning, argument strengths, model "
            "diagnostics, argument graph, and supporting explanations required to "
            "reproduce the decision path."
        ),
    }
