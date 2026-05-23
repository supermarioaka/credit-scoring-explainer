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


def build_candidate_rule(feature, metrics):
    correlation = metrics["correlation_with_target"]
    absolute_correlation = metrics["absolute_correlation"]

    if correlation is None:
        return None

    if correlation > 0:
        risk_direction = "above"
        threshold = metrics["q75"]
        risk_name = f"High {feature}"
        approval_name = f"Acceptable {feature}"
        risk_text = (
            f"{feature} is above the suggested threshold, which is empirically "
            "associated with higher default risk in this dataset."
        )
        approval_text = f"{feature} is not above the suggested risk threshold."

    elif correlation < 0:
        risk_direction = "below"
        threshold = metrics["q25"]
        risk_name = f"Low {feature}"
        approval_name = f"Acceptable {feature}"
        risk_text = (
            f"{feature} is below the suggested threshold, which is empirically "
            "associated with higher default risk in this dataset."
        )
        approval_text = f"{feature} is not below the suggested risk threshold."

    else:
        return None

    return {
        "threshold": threshold,
        "risk_direction": risk_direction,
        "scale": metrics["empirical_range"] if metrics["empirical_range"] else 1,
        "base_strength": absolute_correlation,
        "risk_name": risk_name,
        "approval_name": approval_name,
        "risk_text": risk_text,
        "approval_text": approval_text,
        "financial_meaning": (
            "This is a statistically suggested feature. A human reviewer should "
            "confirm its financial meaning before governance approval."
        ),
        "governance_justification": (
            "This rule is automatically suggested from empirical dataset diagnostics. "
            "It is not yet governance-approved."
        ),
        "threshold_origin": "Empirical quantile-based candidate threshold",
        "mathematical_basis": (
            "Correlation with target and feature distribution quantiles."
        ),
        "governance_status": "Candidate rule - requires human review",
        "interpretability_level": "Pending review",
        "diagnostics": {
            "correlation_with_target": correlation,
            "absolute_correlation": absolute_correlation,
            "strength_category": classify_strength(absolute_correlation),
            "q25": metrics["q25"],
            "median": metrics["median"],
            "q75": metrics["q75"],
            "iqr": metrics["iqr"],
        },
    }


def suggest_rules_from_metrics(
    feature_metrics,
    excluded_features=None,
    minimum_absolute_correlation=0.03,
):
    """
    Suggest preliminary argumentation rules from dataset-level metrics.

    Filters out:
    - index-like columns
    - excluded/sensitive features
    - very weak empirical relationships
    """

    if excluded_features is None:
        excluded_features = []

    suggested_rules = {}

    for feature, metrics in feature_metrics.items():
        if feature in excluded_features:
            continue

        if feature.lower().startswith("unnamed"):
            continue

        if metrics["absolute_correlation"] is None:
            continue

        if metrics["absolute_correlation"] < minimum_absolute_correlation:
            continue

        candidate_rule = build_candidate_rule(feature, metrics)

        if candidate_rule is not None:
            candidate_rule["rule_quality"] = classify_strength(
                metrics["absolute_correlation"]
            )
            suggested_rules[feature] = candidate_rule
    return suggested_rules
