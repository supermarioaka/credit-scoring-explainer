def generate_audit_trail(
    applicant_data, probability, decision, approve_total, reject_total
):
    audit_steps = []

    audit_steps.append("Step 1: Applicant data was received by the system.")

    audit_steps.append(
        f"Step 2: The predictive model estimated a default probability of {probability:.2%}."
    )

    audit_steps.append(
        f"Step 3: The business decision policy classified the application as: {decision}."
    )

    audit_steps.append(
        f"Step 4: The argumentation layer computed total approval strength = {approve_total:.3f}."
    )

    audit_steps.append(
        f"Step 5: The argumentation layer computed total rejection strength = {reject_total:.3f}."
    )

    if reject_total > approve_total:
        audit_steps.append(
            "Step 6: Rejection arguments dominate the argumentation comparison."
        )
    elif approve_total > reject_total:
        audit_steps.append(
            "Step 6: Approval arguments dominate the argumentation comparison."
        )
    else:
        audit_steps.append("Step 6: Approval and rejection arguments are balanced.")

    return audit_steps
