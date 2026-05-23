import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.argument_rules import ARGUMENT_RULES
from src.modeling import load_model
from src.reporting import create_credit_explanation_report


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
    "A thesis-oriented credit-scoring explanation system combining logistic "
    "regression, business decision thresholds, argumentation-based reasoning, "
    "WHY / WHY-NOT explanations, counterfactuals, and auditability."
)


# ------------------------------------------------------------
# 1. Applicant input
# ------------------------------------------------------------

st.header("1. Applicant Profile")

st.write(
    "The predictive model uses the thesis model features, including age. "
    "The explanation layer uses only financially interpretable argument rules "
    "and deliberately excludes age."
)

explanation_applicant_data = {}

for feature, rule in ARGUMENT_RULES.items():
    explanation_applicant_data[feature] = st.number_input(
        label=feature,
        value=float(rule["threshold"]),
    )

age = st.number_input(
    label="age",
    value=35.0,
)

model_applicant_data = explanation_applicant_data.copy()
model_applicant_data["age"] = age


# ------------------------------------------------------------
# 2. Generate report
# ------------------------------------------------------------

if st.button("Evaluate Applicant"):
    report = create_credit_explanation_report(
        explanation_applicant_data=explanation_applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
    )

    predictive = report["predictive_layer"]
    argumentation = report["argumentation_layer"]
    reconciliation = report["decision_reconciliation"]
    counterfactuals = report["counterfactuals"]
    audit_record = report["audit_record"]

    strength_summary = argumentation["strength_summary"]

    probability_of_default = predictive["probability_of_default"]
    linear_score = predictive["linear_score"]
    business_decision = predictive["business_decision"]
    argument_decision = strength_summary["argument_decision"]

    approve_total = strength_summary["approve_total"]
    reject_total = strength_summary["reject_total"]

    # ------------------------------------------------------------
    # 3. Predictive layer
    # ------------------------------------------------------------

    st.header("2. Predictive Layer")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Probability of Default", f"{probability_of_default:.2%}")

    with col2:
        st.metric("Linear Score", f"{linear_score:.3f}")

    with col3:
        st.metric("Business Policy Decision", business_decision)

    st.info(
        "Business policy: PD < 10% → Approve, "
        "10% ≤ PD ≤ 30% → Review, PD > 30% → Reject."
    )

    with st.expander("Model feature contributions"):
        feature_contributions = predictive["feature_contributions"]

        contribution_df = pd.DataFrame(
            {
                "Feature": list(feature_contributions.keys()),
                "Contribution to linear score": list(feature_contributions.values()),
            }
        )

        st.dataframe(contribution_df, width="stretch")

        contribution_chart = go.Figure(
            data=[
                go.Bar(
                    x=list(feature_contributions.keys()),
                    y=list(feature_contributions.values()),
                )
            ]
        )

        contribution_chart.update_layout(
            title="Feature Contributions to Logistic Regression Linear Score",
            yaxis_title="Contribution",
        )

        st.plotly_chart(contribution_chart, width="stretch")

    # ------------------------------------------------------------
    # 4. Model validation and coefficient normalization
    # ------------------------------------------------------------

    st.header("3. Model Diagnostics")

    model_diagnostics = predictive["model_diagnostics"]
    validation_metrics = model_diagnostics["validation_metrics"]

    if validation_metrics is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{validation_metrics['accuracy']:.4f}")

        with col2:
            st.metric("ROC-AUC", f"{validation_metrics['roc_auc']:.4f}")

        with st.expander("Validation details"):
            st.write("**Confusion matrix**")
            st.write(validation_metrics["confusion_matrix"])

            st.write("**Classification report**")
            st.json(validation_metrics["classification_report"])
    else:
        st.warning("No validation metrics found in the saved model object.")

    with st.expander("Coefficient normalization"):
        coefficient_strengths = model_diagnostics["coefficient_strengths"]

        st.write("**Formula**")
        st.code("base_strength = |β_j| / max(|β|)")

        st.metric("Intercept", f"{coefficient_strengths['intercept']:.4f}")

        coefficient_rows = []

        for feature, details in coefficient_strengths["normalized_strengths"].items():
            coefficient_rows.append(
                {
                    "Feature": feature,
                    "Coefficient": details["coefficient"],
                    "Absolute coefficient": details["absolute_coefficient"],
                    "Normalized strength": details["normalized_strength"],
                }
            )

        coefficient_df = pd.DataFrame(coefficient_rows)
        st.dataframe(coefficient_df, width="stretch")

        comparison_df = pd.DataFrame(model_diagnostics["base_strength_comparison"])
        st.write("**Model-derived strengths vs configured rule strengths**")
        st.dataframe(comparison_df, width="stretch")

    # ------------------------------------------------------------
    # 5. Decision summary
    # ------------------------------------------------------------

    st.header("4. Decision Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Business Policy Decision", business_decision)

    with col2:
        st.metric("Argument-Based Decision", argument_decision)

    with col3:
        st.metric("Reconciliation Status", reconciliation["status"])

    st.write(reconciliation["explanation"])

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Approval Strength", f"{approve_total:.3f}")

    with col5:
        st.metric("Rejection Strength", f"{reject_total:.3f}")

    strength_chart = go.Figure(
        data=[
            go.Bar(
                x=["Approval Strength", "Rejection Strength"],
                y=[approve_total, reject_total],
            )
        ]
    )

    strength_chart.update_layout(
        title="Approval vs Rejection Argument Strength",
        yaxis_title="Total Strength",
    )

    st.plotly_chart(strength_chart, width="stretch")

    if argument_decision == "Reject":
        st.error("Rejection-supporting arguments dominate.")
    elif argument_decision == "Approve":
        st.success("Approval-supporting arguments dominate.")
    else:
        st.warning("Approval and rejection arguments are balanced.")

    # ------------------------------------------------------------
    # 6. WHY / WHY-NOT
    # ------------------------------------------------------------

    st.header("5. WHY Explanation")

    for item in argumentation["why_explanation"]:
        st.write(f"- {item}")

    st.header("6. WHY-NOT Explanation")

    for item in argumentation["why_not_explanation"]:
        st.write(f"- {item}")

    # ------------------------------------------------------------
    # 7. Argument details
    # ------------------------------------------------------------

    st.header("7. Argument Details")

    for arg in argumentation["arguments"]:
        if arg["side"] == "Reject":
            st.error(f"{arg['name']} | Strength: {arg['strength']:.3f}")
        else:
            st.success(f"{arg['name']} | Strength: {arg['strength']:.3f}")

        st.write(arg["text"])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Applicant Value", arg["value"])

        with col2:
            st.metric("Threshold", arg["threshold"])

        with col3:
            st.metric("Side", arg["side"])

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Base Strength", f"{arg['base_strength']:.3f}")

        with col5:
            st.metric("Activation Strength", f"{arg['activation_strength']:.3f}")

        with col6:
            st.metric("Final Strength", f"{arg['strength']:.3f}")

        st.caption(arg["strength_formula"])

        with st.expander("Financial meaning and governance justification"):
            st.write("**Financial meaning:**")
            st.write(arg["financial_meaning"])

            st.write("**Governance justification:**")
            st.write(arg["governance_justification"])

        st.divider()

    # ------------------------------------------------------------
    # 8. Argument graph
    # ------------------------------------------------------------

    st.header("8. Argument Graph")

    graph_summary = argumentation["argument_graph_summary"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Arguments", graph_summary["number_of_arguments"])

    with col2:
        st.metric("Approval Arguments", graph_summary["number_of_approval_arguments"])

    with col3:
        st.metric("Rejection Arguments", graph_summary["number_of_rejection_arguments"])

    with col4:
        st.metric("Attacks", graph_summary["number_of_attacks"])

    st.write(
        "Approval-supporting and rejection-supporting arguments attack each other "
        "because they support incompatible conclusions."
    )

    with st.expander("Accepted arguments"):
        accepted_arguments = argumentation["accepted_arguments"]

        if accepted_arguments:
            for arg in accepted_arguments:
                st.write(f"- {arg['name']} | strength: {arg['strength']:.3f}")
        else:
            st.info(
                "No accepted arguments because the argument-based decision is Review."
            )

    with st.expander("Raw argument graph"):
        st.json(argumentation["argument_graph"])

    # ------------------------------------------------------------
    # 9. Counterfactuals
    # ------------------------------------------------------------

    st.header("9. Counterfactual Analysis")

    for suggestion in counterfactuals:
        st.subheader(suggestion["title"])

        st.write(suggestion["change"])

        st.write("**Meaning**")
        st.write(suggestion["meaning"])

        if suggestion.get("new_probability") is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Original PD", f"{suggestion['original_probability']:.2%}")

            with col2:
                st.metric("New PD", f"{suggestion['new_probability']:.2%}")

            with col3:
                st.metric("PD Change", f"{suggestion['probability_change']:.2%}")

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

    # ------------------------------------------------------------
    # 10. Audit trail
    # ------------------------------------------------------------

    st.header("10. Audit Trail")

    audit_df = pd.DataFrame(audit_record["argumentation_layer"]["arguments"])

    st.dataframe(audit_df, width="stretch")

    with st.expander("Full audit record"):
        st.json(audit_record)
