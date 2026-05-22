from config.argument_rules import ARGUMENT_RULES


def activation_strength(value, threshold, scale, direction):
    """
    Computes how strongly a feature activates an argument.

    direction = "above" means risk is activated when value > threshold.
    direction = "below" means risk is activated when value < threshold.
    """

    if direction == "above":
        distance = max(0, value - threshold)
    elif direction == "below":
        distance = max(0, threshold - value)
    else:
        distance = 0

    normalized_distance = distance / scale
    activation = 0.5 + normalized_distance

    return min(1.0, activation)


def generate_arguments(applicant_data):
    approve_arguments = []
    reject_arguments = []

    approve_total = 0
    reject_total = 0

    for feature_name, rule in ARGUMENT_RULES.items():
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

        activation = activation_strength(
            value,
            threshold,
            scale,
            activation_direction,
        )

        strength = base_strength * activation

        argument = {
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
