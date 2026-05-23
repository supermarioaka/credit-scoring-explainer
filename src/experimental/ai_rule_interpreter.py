def interpret_rule_with_template(feature, rule):
    """
    Temporary non-API version.
    Later we will replace this with an API-key AI call.
    """

    return {
        **rule,
        "ai_financial_interpretation": (
            f"The feature '{feature}' is a candidate explanatory variable. "
            f"In this dataset, it shows a {rule['rule_quality'].lower()} empirical "
            f"association with the target outcome. The proposed argumentation rule "
            f"activates an adverse-risk argument when the applicant's value is "
            f"{rule['risk_direction']} {rule['threshold']}."
        ),
        "ai_governance_note": (
            "This rule is not automatically approved. It is a candidate rule generated "
            "from dataset diagnostics and must be reviewed for financial meaning, "
            "fairness, stability, and regulatory acceptability before being used in "
            "a production decision process."
        ),
    }


def interpret_rule_set(suggested_rules):
    interpreted_rules = {}

    for feature, rule in suggested_rules.items():
        interpreted_rules[feature] = interpret_rule_with_template(feature, rule)

    return interpreted_rules
