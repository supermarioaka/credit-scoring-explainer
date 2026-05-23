import math


def compute_activation_strength(
    distance_from_threshold: float,
    scale: float,
    is_discrete: bool = False,
) -> float:
    """
    Computes activation strength using the thesis idea:
    arguments become stronger as the applicant moves further away
    from the relevant threshold.

    For discrete events, such as NumberOfTimes90DaysLate, activation is 1
    when the risk rule is active.
    """

    if is_discrete:
        return 1.0

    if scale <= 0:
        return 0.5

    normalized_distance = distance_from_threshold / scale

    return 1 / (1 + math.exp(-normalized_distance))


def build_argument(rule_evaluation: dict) -> dict:
    rule = rule_evaluation["rule"]
    risk_active = rule_evaluation["risk_active"]

    base_strength = rule["base_strength"]

    activation_strength = compute_activation_strength(
        distance_from_threshold=rule_evaluation["distance_from_threshold"],
        scale=rule.get("scale", 1.0),
        is_discrete=rule.get("is_discrete", False),
    )

    if not risk_active:
        activation_strength = 1 - activation_strength

    strength = base_strength * activation_strength

    if risk_active:
        side = "Reject"
        name = rule["risk_name"]
        text = rule["risk_text"]
    else:
        side = "Approve"
        name = rule["approval_name"]
        text = rule["approval_text"]

    return {
        "feature": rule_evaluation["feature"],
        "side": side,
        "name": name,
        "text": text,
        "value": rule_evaluation["value"],
        "threshold": rule_evaluation["threshold"],
        "distance_from_threshold": rule_evaluation["distance_from_threshold"],
        "base_strength": base_strength,
        "activation_strength": activation_strength,
        "strength": strength,
        "strength_formula": "strength = base_strength × activation_strength",
        "financial_meaning": rule["financial_meaning"],
        "governance_justification": rule["governance_justification"],
    }


def build_arguments(rule_evaluations: list[dict]) -> list[dict]:
    return [build_argument(evaluation) for evaluation in rule_evaluations]
