def generate_case_summary(result):
    adverse_drivers = [arg["name"] for arg in result["reject_arguments"]]

    mitigating_factors = [arg["name"] for arg in result["approve_arguments"]]

    summary = {
        "official_decision": result["policy_decision"],
        "probability_of_default": round(result["probability_of_default"], 4),
        "argumentation_risk_signal": result["argumentation_risk_signal"],
        "approve_argument_strength": round(result["approve_total"], 4),
        "reject_argument_strength": round(result["reject_total"], 4),
        "main_adverse_drivers": adverse_drivers,
        "mitigating_factors": mitigating_factors,
        "why": result["why_explanation"],
        "why_not": result["why_not_explanation"],
    }

    return summary
