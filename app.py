import streamlit as st
import pandas as pd

from config.argument_rules import ARGUMENT_RULES
from src.decision_policy import apply_business_policy
from src.rule_engine import evaluate_applicant_rules
from src.argumentation import build_arguments
from src.reasoning_engine import (
    summarize_argument_strengths,
    generate_why_explanation,
    generate_why_not_explanation,
)
from src.audit import create_audit_record
from src.modeling import (
    load_model,
    predict_default_probability,
    compute_linear_score,
    compute_feature_contributions,
)
from src.counterfactual import generate_counterfactuals


st.set_page_config(
    page_title="Credit Scoring Explainer",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


st.title("🏦 Credit Scoring Explainer")
st.write(
    "An auditable credit-scoring explanation system based on a logistic regression "
    "predictive layer, business decision thresholds, and argument strength aggregation."
)


# ------------------------------------------------------------
# 1. Applicant input
# ------------------------------------------------------------

st.header("1. Applicant Profile")

st.write(
    "The predictive model uses age, while the explanation layer uses only "
    "financially interpretable thesis rules."
)

applicant_data = {}

for feature, rule in ARGUMENT_RULES.items():
    applicant_data[feature] = st.number_input(
        label=feature,
        value=float(rule["threshold"]),
    )

age = st.number_input(
    label="age",
    value=35.0,
)

model_applicant_data = applicant_data.copy()
model_applicant_data["age"] = age


# ------------------------------------------------------------
# 2. Predictive layer
# ------------------------------------------------------------

st.header("2. Predictive Layer")

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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Probability of Default", f"{probability_of_default:.2%}")

with col2:
    st.metric("Linear Score", f"{linear_score:.3f}")

with col3:
    st.metric("Business Policy Decision", business_decision)

with st.expander("Model feature contributions"):
    contribution_df = pd.DataFrame(
        {
            "Feature": list(feature_contributions.keys()),
            "Contribution to linear score": list(feature_contributions.values()),
        }
    )
    st.dataframe(contribution_df, width="stretch")


# ------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------

if st.button("Evaluate Applicant"):
    rule_evaluations = evaluate_applicant_rules(applicant_data)
    arguments = build_arguments(rule_evaluations)

    strength_summary = summarize_argument_strengths(arguments)

    argument_decision = strength_summary["argument_decision"]
    approve_total = strength_summary["approve_total"]
    reject_total = strength_summary["reject_total"]

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
        explanation_applicant_data=applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
        predict_function=predict_default_probability,
    )
    st.header("3. Decision Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Business Policy Decision", business_decision)

    with col2:
        st.metric("Argument-Based Decision", argument_decision)

    with col3:
        st.metric("PD", f"{probability_of_default:.2%}")

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Approval Strength", f"{approve_total:.3f}")

    with col5:
        st.metric("Rejection Strength", f"{reject_total:.3f}")

    if argument_decision == "Reject":
        st.error("Rejection-supporting arguments dominate.")
    elif argument_decision == "Approve":
        st.success("Approval-supporting arguments dominate.")
    else:
        st.warning("Approval and rejection arguments are balanced.")

    st.header("4. WHY Explanation")

    if why_explanation:
        for item in why_explanation:
            st.write(f"- {item}")
    else:
        st.info("No dominant explanation available.")

    st.header("5. WHY-NOT Explanation")

    if why_not_explanation:
        for item in why_not_explanation:
            st.write(f"- {item}")
    else:
        st.info("No opposing arguments available.")

    st.header("6. Argument Details")

    for arg in arguments:
        if arg["side"] == "Reject":
            st.error(f"{arg['name']} | Strength: {arg['strength']:.3f}")
        else:
            st.success(f"{arg['name']} | Strength: {arg['strength']:.3f}")

        st.write(arg["text"])

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Applicant Value", arg["value"])

        with col_b:
            st.metric("Threshold", arg["threshold"])

        with col_c:
            st.metric("Side", arg["side"])
        col_d, col_e, col_f = st.columns(3)

        with col_d:
            st.metric("Base Strength", f"{arg['base_strength']:.3f}")

        with col_e:
            st.metric("Activation Strength", f"{arg['activation_strength']:.3f}")

        with col_f:
            st.metric("Final Strength", f"{arg['strength']:.3f}")

            st.caption(arg["strength_formula"])

        with st.expander("Financial meaning and audit justification"):
            st.write("**Financial meaning:**")
            st.write(arg["financial_meaning"])

            st.write("**Governance justification:**")
            st.write(arg["governance_justification"])

        st.divider()

    st.header("7. Audit Trail")

    audit_df = pd.DataFrame(audit_record["argumentation_layer"]["arguments"])
    st.dataframe(audit_df, width="stretch")

    with st.expander("Full audit record"):
        st.json(audit_record)

    st.header("8. Counterfactual Analysis")

    for suggestion in counterfactuals:
        st.subheader(suggestion["title"])

        st.write(suggestion["change"])

        st.write("**Meaning**")
        st.write(suggestion["meaning"])

        if suggestion.get("new_probability") is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Original PD",
                    f"{suggestion['original_probability']:.2%}",
                )

            with col2:
                st.metric(
                    "New PD",
                    f"{suggestion['new_probability']:.2%}",
                )

            with col3:
                st.metric(
                    "PD Change",
                    f"{suggestion['probability_change']:.2%}",
                )

            col4, col5 = st.columns(2)

            with col4:
                st.metric(
                    "Original Business Decision",
                    suggestion["original_business_decision"],
                )

            with col5:
                st.metric(
                    "New Business Decision",
                    suggestion["new_business_decision"],
                )

        st.divider()
