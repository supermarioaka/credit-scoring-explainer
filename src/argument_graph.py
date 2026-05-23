def create_argument_id(index: int, side: str) -> str:
    prefix = "A" if side == "Approve" else "R"
    return f"{prefix}{index + 1}"


def build_argument_nodes(arguments: list[dict]) -> list[dict]:
    nodes = []

    for index, argument in enumerate(arguments):
        argument_id = create_argument_id(
            index=index,
            side=argument["side"],
        )

        nodes.append(
            {
                "id": argument_id,
                "feature": argument["feature"],
                "side": argument["side"],
                "name": argument["name"],
                "strength": argument["strength"],
                "text": argument["text"],
            }
        )

    return nodes


def build_attack_relations(nodes: list[dict]) -> list[dict]:
    """
    In the thesis structure, approval-supporting arguments and
    rejection-supporting arguments are in conflict.

    Therefore:
        Approve arguments attack Reject arguments
        Reject arguments attack Approve arguments
    """

    attacks = []

    for attacker in nodes:
        for target in nodes:
            if attacker["id"] == target["id"]:
                continue

            if attacker["side"] != target["side"]:
                attacks.append(
                    {
                        "attacker": attacker["id"],
                        "target": target["id"],
                        "attacker_side": attacker["side"],
                        "target_side": target["side"],
                    }
                )

    return attacks


def build_argument_graph(arguments: list[dict]) -> dict:
    nodes = build_argument_nodes(arguments)
    attacks = build_attack_relations(nodes)

    return {
        "nodes": nodes,
        "attacks": attacks,
        "graph_type": "Abstract Argumentation Framework",
        "interpretation": (
            "Arguments supporting approval and arguments supporting rejection "
            "attack each other because they support incompatible conclusions."
        ),
    }


def get_accepted_arguments(
    arguments: list[dict],
    argument_decision: str,
) -> list[dict]:
    """
    Quantitative acceptance rule:

    The accepted arguments are the arguments that support the winning side
    after strength aggregation.
    """

    if argument_decision == "Review":
        return []

    return [argument for argument in arguments if argument["side"] == argument_decision]


def summarize_argument_graph(argument_graph: dict) -> dict:
    nodes = argument_graph["nodes"]
    attacks = argument_graph["attacks"]

    approve_nodes = [node for node in nodes if node["side"] == "Approve"]

    reject_nodes = [node for node in nodes if node["side"] == "Reject"]

    return {
        "number_of_arguments": len(nodes),
        "number_of_approval_arguments": len(approve_nodes),
        "number_of_rejection_arguments": len(reject_nodes),
        "number_of_attacks": len(attacks),
    }
