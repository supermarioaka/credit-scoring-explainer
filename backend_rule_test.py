from src.rule_engine import evaluate_applicant_rules
from src.argumentation import build_arguments
from src.reasoning_engine import (
    summarize_argument_strengths,
    generate_why_explanation,
    generate_why_not_explanation,
)
from src.decision_policy import apply_business_policy
from src.audit import create_audit_record
from src.modeling import (
    load_model,
    predict_default_probability,
    compute_linear_score,
    compute_feature_contributions,
)
from src.counterfactual import generate_counterfactuals


def main():
    explanation_applicant_data = {
        "RevolvingUtilizationOfUnsecuredLines": 0.58,
        "DebtRatio": 0.42,
        "MonthlyIncome": 2333,
        "NumberOfTimes90DaysLate": 1,
    }

    model_applicant_data = explanation_applicant_data.copy()
    model_applicant_data["age"] = 35

    model = load_model()

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

    audit_record = create_audit_record(
        applicant_data=model_applicant_data,
        probability_of_default=probability_of_default,
        business_decision=business_decision,
        arguments=arguments,
        strength_summary=strength_summary,
        linear_score=linear_score,
        feature_contributions=feature_contributions,
    )

    counterfactuals = generate_counterfactuals(
        explanation_applicant_data=explanation_applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
        predict_function=predict_default_probability,
    )

    print("\nPREDICTIVE LAYER")
    print("Probability of default:", round(probability_of_default, 4))
    print("Linear score:", round(linear_score, 4))
    print("Business decision:", business_decision)

    print("\nARGUMENTATION LAYER")
    print("Approval total:", round(strength_summary["approve_total"], 4))
    print("Rejection total:", round(strength_summary["reject_total"], 4))
    print("Argument-based decision:", argument_decision)

    print("\nWHY EXPLANATION")
    for item in why_explanation:
        print("-", item)

    print("\nWHY-NOT EXPLANATION")
    for item in why_not_explanation:
        print("-", item)

    print("\nCOUNTERFACTUALS")
    for suggestion in counterfactuals:
        print("-", suggestion["title"])
        print(" ", suggestion["change"])
        print(" ", suggestion["meaning"])

    print("\nAUDIT RECORD KEYS")
    print(audit_record.keys())


if __name__ == "__main__":
    main()
