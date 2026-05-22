from config.argument_rules import ARGUMENT_RULES


def get_active_rule_set(mode="approved", suggested_rules=None):
    """
    Return the rule set used by the argumentation engine.

    approved:
        Uses governance-approved rules.

    suggested:
        Uses statistically suggested candidate rules.

    custom:
        Allows externally supplied rules.
    """

    if mode == "approved":
        return ARGUMENT_RULES

    if mode == "suggested":
        if suggested_rules is None:
            raise ValueError("suggested_rules must be provided when mode='suggested'.")
        return suggested_rules

    if mode == "custom":
        if suggested_rules is None:
            raise ValueError("custom rules must be provided when mode='custom'.")
        return suggested_rules

    raise ValueError("Unknown rule mode. Use 'approved', 'suggested', or 'custom'.")


def validate_rule_set(rule_set):
    """
    Basic validation for a rule set before it is used by the argumentation engine.
    """

    required_fields = [
        "threshold",
        "risk_direction",
        "scale",
        "base_strength",
        "risk_name",
        "approval_name",
        "risk_text",
        "approval_text",
        "financial_meaning",
        "governance_justification",
    ]

    errors = []

    for feature_name, rule in rule_set.items():
        for field in required_fields:
            if field not in rule:
                errors.append(
                    f"Rule for feature '{feature_name}' is missing field '{field}'."
                )

        if "risk_direction" in rule and rule["risk_direction"] not in [
            "above",
            "below",
        ]:
            errors.append(
                f"Rule for feature '{feature_name}' has invalid risk_direction."
            )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }
