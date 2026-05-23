import pandas as pd

from src.generic_rule_pipeline import (
    analyze_dataset_for_argument_rules,
    evaluate_applicant_with_generated_rules,
)

df = pd.read_csv("data/cs-training.csv")

result = analyze_dataset_for_argument_rules(
    df=df,
    target_column="SeriousDlqin2yrs",
)

print("Validation:")
print(result["validation"])

print("\nSuggested rules:")
for feature, rule in result["suggested_rules"].items():
    print("\nFeature:", feature)
    print("Threshold:", rule["threshold"])
    print("Risk direction:", rule["risk_direction"])
    print("Base strength:", rule["base_strength"])
    print("Rule quality:", rule["rule_quality"])
    print("Governance status:", rule["governance_status"])

print("\nInterpreted rules:")

for feature, rule in result["interpreted_rules"].items():
    print("\nFeature:", feature)
    print("AI financial interpretation:", rule["ai_financial_interpretation"])
    print("AI governance note:", rule["ai_governance_note"])

print("\nLLM ANALYSIS:")
print("LLM status:", result["llm_analysis"]["llm_status"])
print("LLM explanation:")
print(result["llm_analysis"]["llm_explanation"])
print("\nDYNAMIC APPLICANT ARGUMENTATION TEST:")

applicant_data = {
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "MonthlyIncome": 2500,
    "NumberOfOpenCreditLinesAndLoans": 4,
    "NumberOfTimes90DaysLate": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}

applicant_result = evaluate_applicant_with_generated_rules(
    applicant_data=applicant_data,
    rule_set=result["suggested_rules"],
)

approve_arguments = applicant_result["approve_arguments"]
reject_arguments = applicant_result["reject_arguments"]
approve_total = applicant_result["approve_total"]
reject_total = applicant_result["reject_total"]
argument_decision = applicant_result["argument_decision"]

print("Approval total:", approve_total)
print("Rejection total:", reject_total)
print("Argument-based decision:", argument_decision)

print("\nApproval arguments:")
for argument in approve_arguments:
    print(
        "-",
        argument["name"],
        "| strength:",
        round(argument["strength"], 3),
        "| base:",
        round(argument["base_strength"], 3),
        "| activation:",
        round(argument["activation_strength"], 3),
        "| distance:",
        round(argument["distance_from_threshold"], 3),
    )
    print("  formula:", argument["strength_formula"])

print("\nRejection arguments:")
for argument in reject_arguments:
    print(
        "-",
        argument["name"],
        "| strength:",
        round(argument["strength"], 3),
        "| base:",
        round(argument["base_strength"], 3),
        "| activation:",
        round(argument["activation_strength"], 3),
        "| distance:",
        round(argument["distance_from_threshold"], 3),
    )
    print("  formula:", argument["strength_formula"])
