from src.decision_policy import apply_business_policy
from src.rule_engine import evaluate_applicant_rules
from src.argumentation import build_arguments
from src.reasoning_engine import (
    summarize_argument_strengths,
    generate_why_explanation,
    generate_why_not_explanation,
)
from src.argument_graph import (
    build_argument_graph,
    get_accepted_arguments,
    summarize_argument_graph,
)
from src.audit import create_audit_record
from src.modeling import (
    predict_default_probability,
    compute_linear_score,
    compute_feature_contributions,
)
from src.counterfactual import generate_counterfactuals
from src.model_diagnostics import create_model_diagnostics_report


def reconcile_decisions(
    business_decision: str,
    argument_decision: str,
) -> dict:
    """
    Explains the relationship between the predictive business decision
    and the argument-based decision.
    """

    if business_decision == argument_decision:
        status = "Aligned"
        explanation = (
            "The predictive business policy and the argumentation layer reach "
            "the same decision."
        )
    else:
        status = "Different"
        explanation = (
            "The predictive business policy and the argumentation layer reach "
            "different conclusions. This is expected because the predictive "
            "layer estimates probability of default, while the argumentation "
            "layer evaluates structured financial reasons using approved rules."
        )

    return {
        "status": status,
        "business_decision": business_decision,
        "argument_decision": argument_decision,
        "explanation": explanation,
    }


def create_credit_explanation_report(
    explanation_applicant_data: dict,
    model_applicant_data: dict,
    model,
) -> dict:
    """
    Creates one complete thesis-oriented backend report.

    This is the central output of the backend:
        predictive layer
        business policy
        argumentation layer
        WHY / WHY-NOT
        argument graph
        counterfactuals
        audit trail
    """

    probability_of_default = predict_default_probability(
        model=model,
        applicant_data=model_applicant_data,
    )

    linear_score = compute_linear_score(
        model=model,
        applicant_data=model_applicant_data,
    )

    feature_contributions = compute_feature_contributions(
        model=model,
        applicant_data=model_applicant_data,
    )

    business_decision = apply_business_policy(probability_of_default)

    rule_evaluations = evaluate_applicant_rules(explanation_applicant_data)
    arguments = build_arguments(rule_evaluations)

    strength_summary = summarize_argument_strengths(arguments)
    argument_decision = strength_summary["argument_decision"]

    why_explanation = generate_why_explanation(arguments, argument_decision)
    why_not_explanation = generate_why_not_explanation(arguments, argument_decision)

    argument_graph = build_argument_graph(arguments)
    argument_graph_summary = summarize_argument_graph(argument_graph)

    accepted_arguments = get_accepted_arguments(
        arguments=arguments,
        argument_decision=argument_decision,
    )

    counterfactuals = generate_counterfactuals(
        explanation_applicant_data=explanation_applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
        predict_function=predict_default_probability,
    )

    model_diagnostics = create_model_diagnostics_report(model)

    decision_reconciliation = reconcile_decisions(
        business_decision=business_decision,
        argument_decision=argument_decision,
    )

    audit_record = create_audit_record(
        applicant_data=model_applicant_data,
        probability_of_default=probability_of_default,
        business_decision=business_decision,
        arguments=arguments,
        strength_summary=strength_summary,
        linear_score=linear_score,
        feature_contributions=feature_contributions,
        model_diagnostics=model_diagnostics,
        argument_graph=argument_graph,
        accepted_arguments=accepted_arguments,
    )

    return {
        "predictive_layer": {
            "probability_of_default": probability_of_default,
            "linear_score": linear_score,
            "business_decision": business_decision,
            "feature_contributions": feature_contributions,
            "model_diagnostics": model_diagnostics,
        },
        "argumentation_layer": {
            "arguments": arguments,
            "strength_summary": strength_summary,
            "why_explanation": why_explanation,
            "why_not_explanation": why_not_explanation,
            "argument_graph": argument_graph,
            "argument_graph_summary": argument_graph_summary,
            "accepted_arguments": accepted_arguments,
        },
        "decision_reconciliation": decision_reconciliation,
        "counterfactuals": counterfactuals,
        "audit_record": audit_record,
    }
