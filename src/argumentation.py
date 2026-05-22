import math
from src.rule_engine import get_active_rule_set


def activation_strength(value, threshold, scale, direction):
    """
    Computes sigmoid activation strength and returns
    both the activation value and the intermediate mathematical quantities.
    """

    if direction == "above":
        distance = max(0, value - threshold)
    elif direction == "below":
        distance = max(0, threshold - value)
    else:
        distance = 0

    normalized_distance = distance / scale

    k = 5
    activation = 1 / (1 + math.exp(-k * normalized_distance))

    return {
        "distance_from_threshold": distance,
        "normalized_distance": normalized_distance,
        "sensitivity": k,
        "activation": activation,
    }


def generate_arguments(applicant_data, rule_set=None):
    approve_arguments = []
    reject_arguments = []

    approve_total = 0
    reject_total = 0

    if rule_set is None:
        active_rules = get_active_rule_set()
    else:
        active_rules = rule_set

    for feature_name, rule in active_rules.items():
        value = applicant_data[feature_name]

        threshold = rule["threshold"]
        risk_direction = rule["risk_direction"]
        scale = rule["scale"]
        base_strength = rule["base_strength"]

        if risk_direction == "above":
            risk_is_active = value > threshold
            activation_direction = "above" if risk_is_active else "below"
        else:
            risk_is_active = value < threshold
            activation_direction = "below" if risk_is_active else "above"

        activation_details = activation_strength(
            value,
            threshold,
            scale,
            activation_direction,
        )

        activation = activation_details["activation"]

        strength = base_strength * activation

        argument = {
            "distance_from_threshold": activation_details["distance_from_threshold"],
            "normalized_distance": activation_details["normalized_distance"],
            "activation_formula": "1 / (1 + exp(-k * normalized_distance))",
            "strength_formula": "base_strength * activation_strength",
            "feature": feature_name,
            "value": value,
            "threshold": threshold,
            "name": rule["risk_name"] if risk_is_active else rule["approval_name"],
            "text": rule["risk_text"] if risk_is_active else rule["approval_text"],
            "base_strength": base_strength,
            "activation_strength": activation,
            "strength": strength,
            "financial_meaning": rule["financial_meaning"],
            "governance_justification": rule["governance_justification"],
        }

        if risk_is_active:
            reject_total += strength
            reject_arguments.append(argument)
        else:
            approve_total += strength
            approve_arguments.append(argument)

    return approve_arguments, reject_arguments, approve_total, reject_total


def generate_why_explanation(decision, approve_arguments, reject_arguments):
    approve_total = sum(arg["strength"] for arg in approve_arguments)
    reject_total = sum(arg["strength"] for arg in reject_arguments)

    if reject_total > approve_total:
        dominant_side = "rejection-supporting"
    elif approve_total > reject_total:
        dominant_side = "approval-supporting"
    else:
        dominant_side = "balanced"

    if decision == "Approve":
        why = (
            "The application is approved because the predicted probability of default "
            "falls below the bank's approval threshold. The quantified argumentation "
            f"layer provides an additional audit signal: the dominant evidence is {dominant_side}."
        )
        why_not = (
            "The application is not rejected because the official policy decision is based "
            "on the probability-of-default threshold, which does not indicate high enough risk "
            "for rejection."
        )

    elif decision == "Review":
        why = (
            "The application is assigned to Review because the predicted probability of default "
            "falls inside the intermediate policy zone. The quantified argumentation layer "
            f"shows that the dominant evidence is {dominant_side}, so the case should be "
            "examined carefully before a final manual decision."
        )
        why_not = (
            "The system does not make a direct Approve or Reject decision because the probability "
            "of default is not low enough for automatic approval and not high enough for automatic "
            "rejection under the bank policy."
        )

    else:
        why = (
            "The application is rejected because the predicted probability of default exceeds "
            "the bank's rejection threshold. The quantified argumentation layer supports the audit "
            f"process by showing that the dominant evidence is {dominant_side}."
        )
        why_not = (
            "The application is not approved because the probability-of-default estimate exceeds "
            "the acceptable risk level defined by the bank policy."
        )

    return why, why_not


def make_argument_decision(approve_total, reject_total):
    if reject_total > approve_total:
        return "Reject"
    elif approve_total > reject_total:
        return "Approve"
    else:
        return "Review"


def compute_argumentation_risk_signal(approve_total, reject_total):
    difference = reject_total - approve_total

    if difference >= 0.75:
        return "High adverse argument signal"
    elif difference >= 0.25:
        return "Moderate adverse argument signal"
    elif difference > 0:
        return "Low adverse argument signal"
    else:
        return "No adverse argument dominance"
