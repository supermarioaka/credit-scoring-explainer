from config.argument_rules import ARGUMENT_RULES


def evaluate_rule(feature: str, value: float, rule: dict) -> dict:
    threshold = rule["threshold"]
    risk_direction = rule["risk_direction"]

    if risk_direction == "above":
        risk_active = value > threshold
        distance = max(0, value - threshold)
    elif risk_direction == "below":
        risk_active = value < threshold
        distance = max(0, threshold - value)
    else:
        raise ValueError(f"Unknown risk direction: {risk_direction}")

    return {
        "feature": feature,
        "value": value,
        "threshold": threshold,
        "risk_active": risk_active,
        "distance_from_threshold": distance,
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
