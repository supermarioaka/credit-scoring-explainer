from src.feature_metrics import compute_feature_metrics
from src.rule_suggester import suggest_rules_from_metrics
from src.rule_engine import validate_rule_set
from src.ai_rule_interpreter import interpret_rule_set
from src.llm_rule_explainer import explain_rule_set_with_llm
from src.argumentation import generate_arguments, make_argument_decision


def analyze_dataset_for_argument_rules(df, target_column):
    """
    General backend pipeline.

    Takes any dataset and target column, then:
    1. computes feature metrics
    2. suggests candidate argumentation rules
    3. validates that the rules have the correct structure
    """

    feature_metrics = compute_feature_metrics(df, target_column)

    suggested_rules = suggest_rules_from_metrics(
        feature_metrics,
        excluded_features=["age"],
        minimum_absolute_correlation=0.01,
    )
    validation = validate_rule_set(suggested_rules)
    interpreted_rules = interpret_rule_set(suggested_rules)
    llm_analysis = explain_rule_set_with_llm(interpreted_rules)
    return {
        "feature_metrics": feature_metrics,
        "suggested_rules": suggested_rules,
        "interpreted_rules": interpreted_rules,
        "validation": validation,
        "llm_analysis": llm_analysis,
    }


def evaluate_applicant_with_generated_rules(applicant_data, rule_set):
    """
    Evaluate one applicant using a dynamically generated rule set.
    """

    missing_features = []

    for feature in rule_set:
        if feature not in applicant_data:
            missing_features.append(feature)

    if missing_features:
        raise ValueError(
            f"Applicant data is missing required features: {missing_features}"
        )

    approve_arguments, reject_arguments, approve_total, reject_total = (
        generate_arguments(
            applicant_data=applicant_data,
            rule_set=rule_set,
        )
    )

    argument_decision = make_argument_decision(
        approve_total,
        reject_total,
    )

    return {
        "applicant_data": applicant_data,
        "approve_arguments": approve_arguments,
        "reject_arguments": reject_arguments,
        "approve_total": approve_total,
        "reject_total": reject_total,
        "argument_decision": argument_decision,
    }
