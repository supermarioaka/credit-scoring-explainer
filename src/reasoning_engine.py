def summarize_argument_strengths(arguments: list[dict]) -> dict:
    approve_total = sum(
        arg["strength"] for arg in arguments if arg["side"] == "Approve"
    )
    reject_total = sum(arg["strength"] for arg in arguments if arg["side"] == "Reject")

    if reject_total > approve_total:
        argument_decision = "Reject"
    elif approve_total > reject_total:
        argument_decision = "Approve"
    else:
        argument_decision = "Review"

    return {
        "approve_total": approve_total,
        "reject_total": reject_total,
        "argument_decision": argument_decision,
    }


def generate_why_explanation(
    arguments: list[dict], argument_decision: str
) -> list[str]:
    return [arg["text"] for arg in arguments if arg["side"] == argument_decision]


def generate_why_not_explanation(
    arguments: list[dict], argument_decision: str
) -> list[str]:
    opposite_side = "Approve" if argument_decision == "Reject" else "Reject"

    return [arg["text"] for arg in arguments if arg["side"] == opposite_side]
