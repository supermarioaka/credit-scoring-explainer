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


def build_policy_scenario(
    name,
    probability,
    approve_threshold,
    reject_threshold,
):
    decision = classify_with_custom_policy(
        probability,
        approve_threshold,
        reject_threshold,
    )

    review_band_width = reject_threshold - approve_threshold

    return {
        "policy_name": name,
        "approve_threshold": approve_threshold,
        "reject_threshold": reject_threshold,
        "review_band_width": review_band_width,
        "decision": decision,
        "policy_interpretation": (
            f"Approve below {approve_threshold:.2f}, "
            f"Review between {approve_threshold:.2f} and "
            f"{reject_threshold:.2f}, "
            f"Reject above {reject_threshold:.2f}."
        ),
    }


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
        results.append(
            build_policy_scenario(
                scenario["name"],
                probability,
                scenario["approve_threshold"],
                scenario["reject_threshold"],
            )
        )

    return results
