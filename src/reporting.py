def generate_case_summary(result):
    approve_arguments = result["approve_arguments"]
    reject_arguments = result["reject_arguments"]

    adverse_drivers = [
        {
            "name": arg["name"],
            "strength": round(arg["strength"], 3),
        }
        for arg in reject_arguments
    ]

    mitigating_factors = [
        {
            "name": arg["name"],
            "strength": round(arg["strength"], 3),
        }
        for arg in approve_arguments
    ]

    approve_total = result["approve_total"]
    reject_total = result["reject_total"]

    if reject_total > approve_total:
        dominant_reasoning_side = "Reject"
    elif approve_total > reject_total:
        dominant_reasoning_side = "Approve"
    else:
        dominant_reasoning_side = "Balanced"

    summary = {
        "official_decision": result["policy_decision"],
        "probability_of_default": round(result["probability_of_default"], 4),
        "linear_score": round(result["linear_score"], 4),
        "argumentation_risk_signal": result["argumentation_risk_signal"],
        "approve_argument_strength": round(approve_total, 4),
        "reject_argument_strength": round(reject_total, 4),
        "dominant_reasoning_side": dominant_reasoning_side,
        "main_adverse_drivers": adverse_drivers,
        "mitigating_factors": mitigating_factors,
        "why": result["why_explanation"],
        "why_not": result["why_not_explanation"],
    }

    return summary
