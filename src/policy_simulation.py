def classify_with_custom_policy(
    probability,
    approve_threshold=0.10,
    reject_threshold=0.30,
):
    if probability < approve_threshold:
        return "Approve"
    elif probability <= reject_threshold:
        return "Review"
    else:
        return "Reject"


def simulate_policy_scenarios(probability):
    scenarios = [
        {
            "name": "Current Policy",
            "approve_threshold": 0.10,
            "reject_threshold": 0.30,
        },
        {
            "name": "Stricter Policy",
            "approve_threshold": 0.08,
            "reject_threshold": 0.25,
        },
        {
            "name": "More Conservative Approval",
            "approve_threshold": 0.07,
            "reject_threshold": 0.30,
        },
        {
            "name": "Earlier Rejection",
            "approve_threshold": 0.10,
            "reject_threshold": 0.25,
        },
    ]

    results = []

    for scenario in scenarios:
        decision = classify_with_custom_policy(
            probability,
            scenario["approve_threshold"],
            scenario["reject_threshold"],
        )

        results.append(
            {
                "policy_name": scenario["name"],
                "approve_threshold": scenario["approve_threshold"],
                "reject_threshold": scenario["reject_threshold"],
                "decision": decision,
            }
        )

    return results
