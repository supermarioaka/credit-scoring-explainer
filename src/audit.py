def generate_audit_trail(
    applicant_data, probability, decision, approve_total, reject_total
):
    audit_steps = []

    audit_steps.append(
        "Step 1: Applicant data was received and converted into the model input format."
    )

    audit_steps.append(
        f"Step 2: The predictive model estimated a probability of default of {probability:.2%}."
    )

    audit_steps.append(
        f"Step 3: The business decision policy classified the application as: {decision}."
    )

    audit_steps.append(
        "Step 4: The argumentation layer evaluated each governance-approved rule "
        "against the applicant's feature values."
    )

    audit_steps.append(
        "Step 5: For each activated or non-activated rule, argument strength was computed "
        "as: base strength × sigmoid activation strength."
    )

    audit_steps.append(
        f"Step 6: Total approval-supporting argument strength = {approve_total:.3f}."
    )

    audit_steps.append(
        f"Step 7: Total rejection-supporting argument strength = {reject_total:.3f}."
    )

    if reject_total > approve_total:
        audit_steps.append(
            "Step 8: Rejection-supporting arguments dominate the argumentation comparison."
        )
    elif approve_total > reject_total:
        audit_steps.append(
            "Step 8: Approval-supporting arguments dominate the argumentation comparison."
        )
    else:
        audit_steps.append("Step 8: Approval and rejection arguments are balanced.")

    audit_steps.append(
        "Step 9: The audit trail records the path from applicant data to prediction, "
        "policy classification, argument construction, and quantified reasoning."
    )

    return audit_steps
