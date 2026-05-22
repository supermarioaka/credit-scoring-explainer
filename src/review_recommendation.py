def generate_review_recommendation(
    policy_decision, argumentation_risk_signal, approve_total, reject_total
):
    dominance_gap = reject_total - approve_total

    if policy_decision != "Review":
        return "No manual review recommendation required because the policy decision is automatic."

    if argumentation_risk_signal == "High adverse argument signal":
        return (
            "Escalate for adverse-risk manual review. "
            "The applicant is in the Review zone, but quantified rejection-supporting evidence dominates strongly."
        )

    if argumentation_risk_signal == "Moderate adverse argument signal":
        return (
            "Request additional verification before final decision. "
            "The applicant is in the Review zone and adverse evidence is present but not extreme."
        )

    if abs(dominance_gap) < 0.10:
        return (
            "Borderline review case. "
            "Approval- and rejection-supporting arguments are nearly balanced."
        )

    return (
        "Review case with limited adverse dominance. "
        "Manual analyst assessment is recommended before final approval or rejection."
    )
