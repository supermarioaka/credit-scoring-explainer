def build_argument_graph(approve_arguments, reject_arguments):
    nodes = []
    attacks = []
    supports = []

    for argument in approve_arguments:
        nodes.append(
            {
                "name": argument["name"],
                "side": "Approve",
                "strength": argument["strength"],
            }
        )

    for argument in reject_arguments:
        nodes.append(
            {
                "name": argument["name"],
                "side": "Reject",
                "strength": argument["strength"],
            }
        )

    for approve_arg in approve_arguments:
        for reject_arg in reject_arguments:
            attacks.append(
                {
                    "from": approve_arg["name"],
                    "to": reject_arg["name"],
                    "reason": "Approval-supporting and rejection-supporting arguments are in conflict.",
                }
            )

            attacks.append(
                {
                    "from": reject_arg["name"],
                    "to": approve_arg["name"],
                    "reason": "Rejection-supporting and approval-supporting arguments are in conflict.",
                }
            )

    for i, arg_a in enumerate(approve_arguments):
        for arg_b in approve_arguments[i + 1 :]:
            supports.append(
                {
                    "from": arg_a["name"],
                    "to": arg_b["name"],
                    "reason": "Both arguments support the approval side.",
                }
            )

    for i, arg_a in enumerate(reject_arguments):
        for arg_b in reject_arguments[i + 1 :]:
            supports.append(
                {
                    "from": arg_a["name"],
                    "to": arg_b["name"],
                    "reason": "Both arguments support the rejection side.",
                }
            )

    return {
        "nodes": nodes,
        "attacks": attacks,
        "supports": supports,
    }
