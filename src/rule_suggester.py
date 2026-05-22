def classify_strength(absolute_correlation):
    if absolute_correlation is None:
        return "Unknown"
    elif absolute_correlation >= 0.20:
        return "Strong"
    elif absolute_correlation >= 0.10:
        return "Moderate"
    elif absolute_correlation >= 0.03:
        return "Weak"
    else:
        return "Very weak"


def suggest_rules_from_metrics(feature_metrics):
    """
    Suggest preliminary argumentation rules from dataset-level metrics.

    These are statistical candidates, not final governance-approved rules.
    """

    suggested_rules = {}

    for feature, metrics in feature_metrics.items():
        correlation = metrics["correlation_with_target"]
        absolute_correlation = metrics["absolute_correlation"]

        if correlation is None:
            continue

        if correlation > 0:
            risk_direction = "above"
            supports = "Reject"
            threshold = metrics["q75"]
        elif correlation < 0:
            risk_direction = "below"
            supports = "Reject"
            threshold = metrics["q25"]
        else:
            risk_direction = "unclear"
            supports = "Review"
            threshold = metrics["median"]

        suggested_rules[feature] = {
            "feature": feature,
            "threshold": threshold,
            "risk_direction": risk_direction,
            "suggested_support": supports,
            "correlation_with_target": correlation,
            "absolute_correlation": absolute_correlation,
            "strength_category": classify_strength(absolute_correlation),
            "distribution_reference": {
                "q25": metrics["q25"],
                "median": metrics["median"],
                "q75": metrics["q75"],
                "iqr": metrics["iqr"],
            },
            "statistical_reason": (
                "This candidate rule is suggested from the empirical relationship "
                "between the feature and the target variable, combined with the "
                "feature's observed distribution."
            ),
            "governance_status": "Candidate rule - requires human review",
        }

    return suggested_rules
