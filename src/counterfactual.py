from src.argumentation import generate_arguments
from src.decision_policy import classify_decision
from src.rule_engine import get_active_rule_set


def calculate_counterfactual_impact(
    original_data, changed_data, model, predict_function, rule_set=None
):
    _, _, original_approve, original_reject = generate_arguments(
        original_data, rule_set=rule_set
    )
    _, _, new_approve, new_reject = generate_arguments(changed_data, rule_set=rule_set)

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


def get_counterfactual_target(current_value, threshold, risk_direction):
    """
    Compute the nearest threshold-crossing counterfactual value.

    For above-risk rules:
        risk active when value > threshold
        improvement moves value just below threshold

    For below-risk rules:
        risk active when value < threshold
        improvement moves value just above threshold
    """

    epsilon = 0.01

    if risk_direction == "above":
        return threshold - epsilon

    if risk_direction == "below":
        return threshold + epsilon

    return current_value


def generate_counterfactuals(applicant_data, model, predict_function, rule_set=None):
    suggestions = []
    if rule_set is None:
        active_rules = get_active_rule_set()
    else:
        active_rules = rule_set
    for feature_name, rule in active_rules.items():
        current_value = applicant_data[feature_name]
        threshold = rule["threshold"]
        risk_direction = rule["risk_direction"]

        if risk_direction == "above":
            risk_is_active = current_value > threshold
        elif risk_direction == "below":
            risk_is_active = current_value < threshold
        else:
            risk_is_active = False

        if not risk_is_active:
            continue

        target_value = get_counterfactual_target(
            current_value,
            threshold,
            risk_direction,
        )

        changed_data = applicant_data.copy()
        changed_data[feature_name] = target_value

        impact = calculate_counterfactual_impact(
            applicant_data, changed_data, model, predict_function, rule_set=rule_set
        )

        absolute_change = target_value - current_value

        suggestions.append(
            {
                "title": f"Improve {feature_name}",
                "feature": feature_name,
                "current": f"{current_value}",
                "target": f"{target_value}",
                "threshold": threshold,
                "risk_direction": risk_direction,
                "absolute_change": absolute_change,
                "change": (
                    f"Change {feature_name} from {current_value} to {target_value}"
                ),
                "mathematical_condition": (
                    f"Risk is active because {feature_name} violates "
                    f"the rule threshold under direction '{risk_direction}'."
                ),
                "meaning": (
                    f"This change crosses the threshold for '{rule['risk_name']}' "
                    f"and weakens the corresponding rejection-supporting argument."
                ),
                **impact,
            }
        )

    if suggestions:
        combined_data = applicant_data.copy()

        for feature_name, rule in active_rules.items():
            current_value = applicant_data[feature_name]
            threshold = rule["threshold"]
            risk_direction = rule["risk_direction"]

            if risk_direction == "above" and current_value > threshold:
                combined_data[feature_name] = get_counterfactual_target(
                    current_value,
                    threshold,
                    risk_direction,
                )

            elif risk_direction == "below" and current_value < threshold:
                combined_data[feature_name] = get_counterfactual_target(
                    current_value,
                    threshold,
                    risk_direction,
                )

        combined_impact = calculate_counterfactual_impact(
            applicant_data,
            combined_data,
            model,
            predict_function,
        )

        suggestions.append(
            {
                "title": "Combined Improvement Scenario",
                "feature": "Multiple active risk features",
                "current": "Current applicant profile",
                "target": "All active risk thresholds improved",
                "threshold": "Multiple thresholds",
                "risk_direction": "Multiple directions",
                "absolute_change": None,
                "change": "Apply all feasible rule-based improvements together.",
                "mathematical_condition": (
                    "All currently active risk rules are moved to the nearest "
                    "non-risk side of their thresholds."
                ),
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
                "feature": None,
                "current": "-",
                "target": "-",
                "threshold": None,
                "risk_direction": None,
                "absolute_change": None,
                "change": "-",
                "mathematical_condition": (
                    "No rejection-supporting rule is currently active."
                ),
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
