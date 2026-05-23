from config.argument_rules import ARGUMENT_RULES


def determine_argument_side(
    value: float,
    threshold: float,
    risk_direction: str,
) -> str:
    """
    Determines whether the feature supports rejection or approval.

    For risk_direction = "above":
        value > threshold  -> Reject
        value <= threshold -> Approve

    For risk_direction = "below":
        value < threshold  -> Reject
        value >= threshold -> Approve
    """

    if risk_direction == "above":
        return "Reject" if value > threshold else "Approve"

    if risk_direction == "below":
        return "Reject" if value < threshold else "Approve"

    raise ValueError(f"Unknown risk direction: {risk_direction}")


def compute_signed_distance_from_threshold(
    value: float,
    threshold: float,
    risk_direction: str,
) -> float:
    """
    Computes signed distance from the threshold.

    Positive distance means the feature is on the risk side.
    Negative distance means the feature is on the approval side.
    """

    if risk_direction == "above":
        return value - threshold

    if risk_direction == "below":
        return threshold - value

    raise ValueError(f"Unknown risk direction: {risk_direction}")


def compute_absolute_distance_from_threshold(
    value: float,
    threshold: float,
) -> float:
    """
    Computes absolute threshold distance.

    This is used to measure how strongly the feature supports
    its current side, whether that side is Approve or Reject.
    """

    return abs(value - threshold)


def evaluate_rule(feature: str, value: float, rule: dict) -> dict:
    threshold = rule["threshold"]
    risk_direction = rule["risk_direction"]

    side = determine_argument_side(
        value=value,
        threshold=threshold,
        risk_direction=risk_direction,
    )

    signed_distance = compute_signed_distance_from_threshold(
        value=value,
        threshold=threshold,
        risk_direction=risk_direction,
    )

    absolute_distance = compute_absolute_distance_from_threshold(
        value=value,
        threshold=threshold,
    )

    risk_active = side == "Reject"

    return {
        "feature": feature,
        "value": value,
        "threshold": threshold,
        "risk_direction": risk_direction,
        "side": side,
        "risk_active": risk_active,
        "signed_distance_from_threshold": signed_distance,
        "distance_from_threshold": absolute_distance,
        "rule": rule,
    }


def evaluate_applicant_rules(applicant_data: dict) -> list[dict]:
    evaluations = []

    for feature, rule in ARGUMENT_RULES.items():
        if feature not in applicant_data:
            raise ValueError(f"Missing applicant value for feature: {feature}")

        evaluations.append(
            evaluate_rule(
                feature=feature,
                value=applicant_data[feature],
                rule=rule,
            )
        )

    return evaluations
