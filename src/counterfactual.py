from src.argumentation import generate_arguments
from src.decision_policy import classify_decision
from config.argument_rules import ARGUMENT_RULES


def calculate_counterfactual_impact(
    original_data, changed_data, model, predict_function
):
    _, _, original_approve, original_reject = generate_arguments(original_data)
    _, _, new_approve, new_reject = generate_arguments(changed_data)

    original_probability = predict_function(model, original_data)
    new_probability = predict_function(model, changed_data)

    original_decision = classify_decision(original_probability)
    new_decision = classify_decision(new_probability)

    return {
        "new_approve_total": new_approve,
        "new_reject_total": new_reject,
        "approval_change": new_approve - original_approve,
        "rejection_change": new_reject - original_reject,
        "original_probability": original_probability,
        "new_probability": new_probability,
        "probability_change": new_probability - original_probability,
        "original_decision": original_decision,
        "new_decision": new_decision,
    }


def generate_counterfactuals(applicant_data, model, predict_function):
    suggestions = []

    for feature_name, rule in ARGUMENT_RULES.items():
        current_value = applicant_data[feature_name]
        threshold = rule["threshold"]
        risk_direction = rule["risk_direction"]

        if risk_direction == "above":
            risk_is_active = current_value > threshold
            target_value = threshold - 0.01
        else:
            risk_is_active = current_value < threshold
            target_value = threshold + 1

        if not risk_is_active:
            continue

        changed_data = applicant_data.copy()
        changed_data[feature_name] = target_value

        impact = calculate_counterfactual_impact(
            applicant_data,
            changed_data,
            model,
            predict_function,
        )

        suggestions.append(
            {
                "title": f"Improve {feature_name}",
                "current": f"{current_value}",
                "target": f"{target_value}",
                "change": f"Change {feature_name} from {current_value} to {target_value}",
                "meaning": (
                    f"This change crosses the threshold for '{rule['risk_name']}' "
                    f"and weakens the corresponding rejection-supporting argument."
                ),
                **impact,
            }
        )
    if suggestions:
        combined_data = applicant_data.copy()

        for feature_name, rule in ARGUMENT_RULES.items():
            current_value = applicant_data[feature_name]
            threshold = rule["threshold"]
            risk_direction = rule["risk_direction"]

            if risk_direction == "above" and current_value > threshold:
                combined_data[feature_name] = threshold - 0.01

            elif risk_direction == "below" and current_value < threshold:
                combined_data[feature_name] = threshold + 1

        combined_impact = calculate_counterfactual_impact(
            applicant_data,
            combined_data,
            model,
            predict_function,
        )

        suggestions.append(
            {
                "title": "Combined Improvement Scenario",
                "current": "Current applicant profile",
                "target": "All active risk thresholds improved",
                "change": "Apply all feasible rule-based improvements together.",
                "meaning": (
                    "This scenario estimates the combined effect of correcting all currently "
                    "active risk-threshold violations. It is useful for assessing whether the "
                    "case could move toward a lower-risk profile."
                ),
                **combined_impact,
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "title": "No Major Rule-Based Improvement Needed",
                "current": "-",
                "target": "-",
                "change": "-",
                "meaning": "No major rejection threshold is currently activated.",
                "new_approve_total": None,
                "new_reject_total": None,
                "approval_change": None,
                "rejection_change": None,
                "original_probability": None,
                "new_probability": None,
                "probability_change": None,
                "original_decision": None,
                "new_decision": None,
            }
        )

    return suggestions
