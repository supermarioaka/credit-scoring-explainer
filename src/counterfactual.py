from config.argument_rules import ARGUMENT_RULES
from src.rule_engine import evaluate_applicant_rules
from src.argumentation import build_arguments
from src.reasoning_engine import summarize_argument_strengths
from src.decision_policy import apply_business_policy


def get_counterfactual_target(
    current_value: float, threshold: float, risk_direction: str
) -> float:
    epsilon = 0.01

    if risk_direction == "above":
        return threshold - epsilon

    if risk_direction == "below":
        return threshold + epsilon

    raise ValueError(f"Unknown risk direction: {risk_direction}")


def is_risk_active(value: float, threshold: float, risk_direction: str) -> bool:
    if risk_direction == "above":
        return value > threshold

    if risk_direction == "below":
        return value < threshold

    raise ValueError(f"Unknown risk direction: {risk_direction}")


def calculate_argument_decision(applicant_data: dict) -> dict:
    rule_evaluations = evaluate_applicant_rules(applicant_data)
    arguments = build_arguments(rule_evaluations)
    strength_summary = summarize_argument_strengths(arguments)

    return {
        "arguments": arguments,
        "approve_total": strength_summary["approve_total"],
        "reject_total": strength_summary["reject_total"],
        "argument_decision": strength_summary["argument_decision"],
    }


def calculate_counterfactual_impact(
    original_explanation_data: dict,
    changed_explanation_data: dict,
    model_applicant_data: dict,
    model,
    predict_function,
) -> dict:
    original_argument_result = calculate_argument_decision(original_explanation_data)
    new_argument_result = calculate_argument_decision(changed_explanation_data)

    changed_model_data = model_applicant_data.copy()

    for feature, value in changed_explanation_data.items():
        changed_model_data[feature] = value

    original_probability = predict_function(model, model_applicant_data)
    new_probability = predict_function(model, changed_model_data)

    original_business_decision = apply_business_policy(original_probability)
    new_business_decision = apply_business_policy(new_probability)

    return {
        "original_approve_total": original_argument_result["approve_total"],
        "original_reject_total": original_argument_result["reject_total"],
        "new_approve_total": new_argument_result["approve_total"],
        "new_reject_total": new_argument_result["reject_total"],
        "approval_change": (
            new_argument_result["approve_total"]
            - original_argument_result["approve_total"]
        ),
        "rejection_change": (
            new_argument_result["reject_total"]
            - original_argument_result["reject_total"]
        ),
        "original_probability": original_probability,
        "new_probability": new_probability,
        "probability_change": new_probability - original_probability,
        "original_business_decision": original_business_decision,
        "new_business_decision": new_business_decision,
        "original_argument_decision": original_argument_result["argument_decision"],
        "new_argument_decision": new_argument_result["argument_decision"],
    }


def generate_counterfactuals(
    explanation_applicant_data: dict,
    model_applicant_data: dict,
    model,
    predict_function,
) -> list[dict]:
    suggestions = []

    for feature, rule in ARGUMENT_RULES.items():
        current_value = explanation_applicant_data[feature]
        threshold = rule["threshold"]
        risk_direction = rule["risk_direction"]

        risk_active = is_risk_active(
            value=current_value,
            threshold=threshold,
            risk_direction=risk_direction,
        )

        if not risk_active:
            continue

        target_value = get_counterfactual_target(
            current_value=current_value,
            threshold=threshold,
            risk_direction=risk_direction,
        )

        changed_explanation_data = explanation_applicant_data.copy()
        changed_explanation_data[feature] = target_value

        impact = calculate_counterfactual_impact(
            original_explanation_data=explanation_applicant_data,
            changed_explanation_data=changed_explanation_data,
            model_applicant_data=model_applicant_data,
            model=model,
            predict_function=predict_function,
        )

        suggestions.append(
            {
                "title": f"Improve {feature}",
                "feature": feature,
                "current_value": current_value,
                "target_value": target_value,
                "threshold": threshold,
                "risk_direction": risk_direction,
                "change": f"Change {feature} from {current_value} to {target_value}.",
                "meaning": (
                    f"This crosses the threshold for '{rule['risk_name']}' "
                    "and weakens the corresponding rejection-supporting argument."
                ),
                **impact,
            }
        )

    if suggestions:
        combined_explanation_data = explanation_applicant_data.copy()

        for feature, rule in ARGUMENT_RULES.items():
            current_value = explanation_applicant_data[feature]
            threshold = rule["threshold"]
            risk_direction = rule["risk_direction"]

            if is_risk_active(current_value, threshold, risk_direction):
                combined_explanation_data[feature] = get_counterfactual_target(
                    current_value=current_value,
                    threshold=threshold,
                    risk_direction=risk_direction,
                )

        combined_impact = calculate_counterfactual_impact(
            original_explanation_data=explanation_applicant_data,
            changed_explanation_data=combined_explanation_data,
            model_applicant_data=model_applicant_data,
            model=model,
            predict_function=predict_function,
        )

        suggestions.append(
            {
                "title": "Combined Improvement Scenario",
                "feature": "Multiple active risk features",
                "current_value": "Current applicant profile",
                "target_value": "All active risk thresholds improved",
                "threshold": "Multiple thresholds",
                "risk_direction": "Multiple directions",
                "change": "Apply all feasible rule-based improvements together.",
                "meaning": (
                    "This scenario estimates the combined effect of correcting all "
                    "currently active risk-threshold violations."
                ),
                **combined_impact,
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "No Major Rule-Based Improvement Needed",
                "feature": None,
                "current_value": "-",
                "target_value": "-",
                "threshold": None,
                "risk_direction": None,
                "change": "-",
                "meaning": "No rejection-supporting rule is currently active.",
            }
        )

    return suggestions
