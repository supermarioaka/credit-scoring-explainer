from src.modeling import load_model
from src.reporting import create_credit_explanation_report


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

    report = create_credit_explanation_report(
        explanation_applicant_data=explanation_applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
    )

    predictive = report["predictive_layer"]
    argumentation = report["argumentation_layer"]
    reconciliation = report["decision_reconciliation"]

    strength_summary = argumentation["strength_summary"]

    print("\nPREDICTIVE LAYER")
    print(
        "Probability of default:",
        round(predictive["probability_of_default"], 4),
    )
    print("Linear score:", round(predictive["linear_score"], 4))
    print("Business decision:", predictive["business_decision"])

    print("\nMODEL VALIDATION")
    validation_metrics = predictive["model_diagnostics"]["validation_metrics"]

    if validation_metrics is not None:
        print("Accuracy:", round(validation_metrics["accuracy"], 4))
        print("ROC-AUC:", round(validation_metrics["roc_auc"], 4))
        print("Confusion matrix:", validation_metrics["confusion_matrix"])
    else:
        print("No validation metrics found in model object.")

    print("\nCOEFFICIENT NORMALIZATION")
    coefficient_strengths = predictive["model_diagnostics"]["coefficient_strengths"]

    print("Intercept:", round(coefficient_strengths["intercept"], 4))
    print(
        "Max absolute coefficient:",
        round(coefficient_strengths["max_absolute_coefficient"], 4),
    )

    for feature, details in coefficient_strengths["normalized_strengths"].items():
        print(
            "-",
            feature,
            "| coefficient:",
            round(details["coefficient"], 4),
            "| normalized strength:",
            round(details["normalized_strength"], 4),
        )

    print("\nARGUMENTATION LAYER")
    print("Approval total:", round(strength_summary["approve_total"], 4))
    print("Rejection total:", round(strength_summary["reject_total"], 4))
    print(
        "Argument-based decision:",
        strength_summary["argument_decision"],
    )

    print("\nWHY EXPLANATION")
    for item in argumentation["why_explanation"]:
        print("-", item)

    print("\nWHY-NOT EXPLANATION")
    for item in argumentation["why_not_explanation"]:
        print("-", item)

    print("\nARGUMENT GRAPH")
    graph_summary = argumentation["argument_graph_summary"]
    print("Number of arguments:", graph_summary["number_of_arguments"])
    print("Approval arguments:", graph_summary["number_of_approval_arguments"])
    print("Rejection arguments:", graph_summary["number_of_rejection_arguments"])
    print("Number of attacks:", graph_summary["number_of_attacks"])

    print("\nACCEPTED ARGUMENTS")
    accepted_arguments = argumentation["accepted_arguments"]

    if accepted_arguments:
        for argument in accepted_arguments:
            print(
                "-",
                argument["name"],
                "| strength:",
                round(argument["strength"], 4),
            )
    else:
        print("No accepted arguments because the argument-based decision is Review.")

    print("\nDECISION RECONCILIATION")
    print("Status:", reconciliation["status"])
    print(reconciliation["explanation"])

    print("\nCOUNTERFACTUALS")
    for suggestion in report["counterfactuals"]:
        print("-", suggestion["title"])
        print(" ", suggestion["change"])
        print(" ", suggestion["meaning"])

    print("\nAUDIT RECORD KEYS")
    print(report["audit_record"].keys())


if __name__ == "__main__":
    main()
