APPROVE_THRESHOLD = 0.10
REJECT_THRESHOLD = 0.30


def classify_decision(probability):
    if probability < APPROVE_THRESHOLD:
        return "Approve"
    elif probability <= REJECT_THRESHOLD:
        return "Review"
    else:
        return "Reject"


def explain_policy_decision(probability):
    decision = classify_decision(probability)

    if decision == "Approve":
        policy_reason = (
            "The predicted probability of default is below the approval threshold."
        )
    elif decision == "Review":
        policy_reason = (
            "The predicted probability of default lies inside the intermediate "
            "manual review zone."
        )
    else:
        policy_reason = (
            "The predicted probability of default exceeds the rejection threshold."
        )

    return {
        "decision": decision,
        "probability": probability,
        "approve_threshold": APPROVE_THRESHOLD,
        "reject_threshold": REJECT_THRESHOLD,
        "policy_reason": policy_reason,
        "policy_rule": (
            "Approve if PD < 0.10; Review if 0.10 <= PD <= 0.30; Reject if PD > 0.30."
        ),
    }
