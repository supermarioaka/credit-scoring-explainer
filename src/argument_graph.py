def build_argument_graph(approve_arguments, reject_arguments):
    nodes = []
    attacks = []
    supports = []

    all_arguments = []

    for index, argument in enumerate(approve_arguments):
        argument_id = f"A{index + 1}"

        node = {
            "id": argument_id,
            "name": argument["name"],
            "side": "Approve",
            "strength": argument["strength"],
            "feature": argument["feature"],
        }

        nodes.append(node)
        all_arguments.append(node)

    for index, argument in enumerate(reject_arguments):
        argument_id = f"R{index + 1}"

        node = {
            "id": argument_id,
            "name": argument["name"],
            "side": "Reject",
            "strength": argument["strength"],
            "feature": argument["feature"],
        }

        nodes.append(node)
        all_arguments.append(node)

    for approve_node in [node for node in nodes if node["side"] == "Approve"]:
        for reject_node in [node for node in nodes if node["side"] == "Reject"]:
            attack_strength = min(
                approve_node["strength"],
                reject_node["strength"],
            )

            attacks.append(
                {
                    "from": approve_node["id"],
                    "to": reject_node["id"],
                    "from_name": approve_node["name"],
                    "to_name": reject_node["name"],
                    "strength": attack_strength,
                    "reason": (
                        "Approval-supporting and rejection-supporting arguments "
                        "are in formal conflict."
                    ),
                }
            )

            attacks.append(
                {
                    "from": reject_node["id"],
                    "to": approve_node["id"],
                    "from_name": reject_node["name"],
                    "to_name": approve_node["name"],
                    "strength": attack_strength,
                    "reason": (
                        "Rejection-supporting and approval-supporting arguments "
                        "are in formal conflict."
                    ),
                }
            )

    approve_nodes = [node for node in nodes if node["side"] == "Approve"]
    reject_nodes = [node for node in nodes if node["side"] == "Reject"]

    for i, node_a in enumerate(approve_nodes):
        for node_b in approve_nodes[i + 1 :]:
            support_strength = min(node_a["strength"], node_b["strength"])

            supports.append(
                {
                    "from": node_a["id"],
                    "to": node_b["id"],
                    "from_name": node_a["name"],
                    "to_name": node_b["name"],
                    "strength": support_strength,
                    "reason": "Both arguments support the approval side.",
                }
            )

    for i, node_a in enumerate(reject_nodes):
        for node_b in reject_nodes[i + 1 :]:
            support_strength = min(node_a["strength"], node_b["strength"])

            supports.append(
                {
                    "from": node_a["id"],
                    "to": node_b["id"],
                    "from_name": node_a["name"],
                    "to_name": node_b["name"],
                    "strength": support_strength,
                    "reason": "Both arguments support the rejection side.",
                }
            )

    return {
        "nodes": nodes,
        "attacks": attacks,
        "supports": supports,
    }
