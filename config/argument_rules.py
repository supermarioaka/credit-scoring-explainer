ARGUMENT_RULES = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "threshold": 0.50,
        "risk_direction": "above",
        "scale": 1.094439,
        "base_strength": 1.000,
        "financial_meaning": (
            "Measures the proportion of available unsecured revolving credit "
            "that the applicant is currently using."
        ),
        "risk_name": "High Credit Utilization",
        "approval_name": "Acceptable Credit Utilization",
        "risk_text": (
            "Credit utilization is above 50%, indicating elevated reliance on "
            "available revolving credit."
        ),
        "approval_text": (
            "Credit utilization is below the risk threshold, indicating more "
            "available credit capacity."
        ),
        "governance_justification": (
            "The 50% threshold is financially interpretable and consistent with "
            "the idea that high utilization may indicate liquidity pressure. "
            "The base strength is derived from the normalized model coefficient magnitude."
        ),
    },
    "NumberOfTimes90DaysLate": {
        "threshold": 0,
        "risk_direction": "above",
        "scale": 1,
        "base_strength": 0.525,
        "is_discrete": True,
        "risk_name": "Serious Late Payments",
        "approval_name": "No Serious Late Payments",
        "risk_text": "The applicant has at least one serious 90+ days late payment.",
        "approval_text": "The applicant has no serious late payments.",
        "financial_meaning": (
            "Captures whether the applicant has experienced serious delinquency, "
            "defined as payments 90 or more days past due."
        ),
        "governance_justification": (
            "Any 90+ days past-due event is treated as a strong adverse signal because "
            "it reflects severe repayment difficulty. The base strength is derived from "
            "the normalized model coefficient magnitude."
        ),
    },
    "DebtRatio": {
        "threshold": 0.45,
        "risk_direction": "above",
        "scale": 0.495,
        "base_strength": 0.028,
        "risk_name": "High Debt Ratio",
        "approval_name": "Acceptable Debt Ratio",
        "risk_text": "Debt ratio exceeds the acceptable threshold.",
        "approval_text": "Debt ratio remains below the risk threshold.",
        "financial_meaning": (
            "Measures the applicant's debt burden relative to income or financial capacity."
        ),
        "governance_justification": (
            "The 45% threshold is financially interpretable as a high debt-burden level. "
            "The rule is included for audit transparency, even though its model-derived "
            "base strength is relatively low."
        ),
    },
    "MonthlyIncome": {
        "threshold": 2000,
        "risk_direction": "below",
        "scale": 23000,
        "base_strength": 0.047,
        "risk_name": "Low Monthly Income",
        "approval_name": "Sufficient Monthly Income",
        "risk_text": "Monthly income is below the minimum acceptable threshold.",
        "approval_text": "Monthly income is above the minimum threshold.",
        "financial_meaning": (
            "Represents the applicant's monthly income and repayment capacity."
        ),
        "governance_justification": (
            "The 2000 threshold identifies applicants with materially limited income. "
            "The rule is included as a low-income risk indicator, while its low base "
            "strength reflects its weaker model coefficient magnitude."
        ),
    },
}
