from src.modeling import (
    predict_default_probability,
    compute_linear_score,
    compute_feature_contributions,
)

from src.argument_graph import build_argument_graph
from src.decision_policy import classify_decision
from src.review_recommendation import generate_review_recommendation

from src.argumentation import (
    generate_arguments,
    generate_why_explanation,
    compute_argumentation_risk_signal,
)

from src.policy_simulation import simulate_policy_scenarios


def clean_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def evaluate_application(model, applicant_data):

    # -----------------------------
    # Predictive ML layer
    # -----------------------------

    probability = predict_default_probability(model, applicant_data)

    linear_score = compute_linear_score(model, applicant_data)

    feature_contributions = compute_feature_contributions(
        model,
        applicant_data,
    )

    # -----------------------------
    # Business decision policy
    # -----------------------------

    policy_decision = classify_decision(probability)

    policy_scenarios = simulate_policy_scenarios(probability)

    # -----------------------------
    # Argumentation layer
    # -----------------------------

    approve_arguments, reject_arguments, approve_total, reject_total = (
        generate_arguments(applicant_data)
    )

    argument_graph = build_argument_graph(
        approve_arguments,
        reject_arguments,
    )

    argumentation_risk_signal = compute_argumentation_risk_signal(
        approve_total,
        reject_total,
    )

    review_recommendation = generate_review_recommendation(
        policy_decision,
        argumentation_risk_signal,
        approve_total,
        reject_total,
    )

    why, why_not = generate_why_explanation(
        policy_decision,
        approve_arguments,
        reject_arguments,
    )

    # -----------------------------
    # Final structured result
    # -----------------------------

    feature_contributions = {
        key: clean_number(value) for key, value in feature_contributions.items()
    }

    result = {
        "probability_of_default": clean_number(probability),
        "linear_score": clean_number(linear_score),
        # Official bank decision
        "policy_decision": policy_decision,
        # Policy sensitivity analysis
        "policy_scenarios": policy_scenarios,
        # Argumentation layer
        "argumentation_risk_signal": argumentation_risk_signal,
        "review_recommendation": review_recommendation,
        # Arguments
        "approve_arguments": approve_arguments,
        "reject_arguments": reject_arguments,
        # Argument graph
        "argument_graph": argument_graph,
        # Quantified strengths
        "approve_total": clean_number(approve_total),
        "reject_total": clean_number(reject_total),
        # Model interpretability
        "feature_contributions": feature_contributions,
        # Explanations
        "why_explanation": why,
        "why_not_explanation": why_not,
    }

    return result
