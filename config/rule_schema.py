RULE_SCHEMA_DESCRIPTION = {
    "feature_name": "Name of the dataset feature.",
    "financial_meaning": ("Economic or financial interpretation of the feature."),
    "threshold": (
        "Value at which the argument changes from approval-supporting "
        "to rejection-supporting (or vice versa)."
    ),
    "risk_direction": (
        "'above' means risk activates above threshold. "
        "'below' means risk activates below threshold."
    ),
    "base_strength": (
        "Normalized argument importance derived from model coefficients."
    ),
    "activation_scale": ("Scaling factor used to compute activation intensity."),
    "risk_name": ("Human-readable rejection-supporting argument label."),
    "approval_name": ("Human-readable approval-supporting argument label."),
    "risk_text": ("Formal explanation of why the feature supports elevated risk."),
    "approval_text": ("Formal explanation of why the feature supports approval."),
    "governance_justification": (
        "Reason why this threshold and rule are institutionally acceptable "
        "or financially meaningful."
    ),
}
