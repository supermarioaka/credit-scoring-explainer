def suggest_rules_from_metrics(feature_metrics):
    """
    Suggest preliminary argumentation rules from dataset-level metrics.

    These rules are statistical suggestions.
    They are not final governance-approved rules.
    """

    suggested_rules = {}

    for feature, metrics in feature_metrics.items():
        correlation = metrics["correlation_with_target"]

        if correlation > 0:
            risk_direction = "above"
            supports = "Reject"
            threshold = metrics["q75"]
        else:
            risk_direction = "below"
            supports = "Approve"
            threshold = metrics["q25"]

        suggested_rules[feature] = {
            "threshold": threshold,
            "risk_direction": risk_direction,
            "supports": supports,
            "correlation_with_target": correlation,
            "base_strength": abs(correlation),
            "statistical_reason": (
                "This rule was suggested from the feature's correlation "
                "with the target variable and its empirical distribution."
            ),
        }

    return suggested_rules
