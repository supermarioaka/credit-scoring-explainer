import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ai_explainer import generate_plain_language_explanation
from src.modeling import load_model, predict_default_probability
from src.reasoning_engine import evaluate_application
from src.reporting import generate_case_summary
from src.audit import generate_audit_trail
from src.counterfactual import generate_counterfactuals
from src.feature_metrics import compute_feature_metrics
from src.rule_suggester import suggest_rules_from_metrics


st.set_page_config(page_title="Credit Scoring Explainer", page_icon="🏦", layout="wide")

model = load_model()

st.title("🏦 Credit Scoring Explainer")
st.write(
    "A thesis-oriented app for explainable and auditable credit scoring decisions."
)

# ------------------------------------------------------------
# Dataset Analysis Layer
# ------------------------------------------------------------

st.header("Dataset Analysis")

uploaded_file = st.file_uploader(
    "Optional: upload a dataset for diagnostic rule analysis",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/cs-training.csv")

target_column = st.selectbox(
    "Select target column",
    df.columns,
    index=list(df.columns).index("SeriousDlqin2yrs")
    if "SeriousDlqin2yrs" in df.columns
    else 0,
)

feature_metrics = compute_feature_metrics(df, target_column)
suggested_rules = suggest_rules_from_metrics(feature_metrics)

with st.expander("Show Feature Metrics"):
    st.write(feature_metrics)

with st.expander("Show Suggested Argumentation Rules"):
    st.write(suggested_rules)


# ------------------------------------------------------------
# Applicant Input Layer
# ------------------------------------------------------------

st.sidebar.header("Applicant Information")

credit_utilization = st.sidebar.number_input(
    "Credit Utilization", min_value=0.0, max_value=2.0, value=0.58
)

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)

debt_ratio = st.sidebar.number_input(
    "Debt Ratio", min_value=0.0, max_value=5.0, value=0.42
)

monthly_income = st.sidebar.number_input("Monthly Income", min_value=0, value=2333)

late_payments = st.sidebar.number_input(
    "Number of 90+ Days Late Payments", min_value=0, value=1
)


if st.sidebar.button("Analyze Applicant"):
    applicant_data = {
        "RevolvingUtilizationOfUnsecuredLines": credit_utilization,
        "age": age,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfTimes90DaysLate": late_payments,
    }

    result = evaluate_application(model, applicant_data)
    summary = generate_case_summary(result)
    ai_explanation = generate_plain_language_explanation(summary)
    probability = result["probability_of_default"]
    decision = result["policy_decision"]
    argumentation_risk_signal = result["argumentation_risk_signal"]

    approve_arguments = result["approve_arguments"]
    reject_arguments = result["reject_arguments"]

    approve_total = result["approve_total"]
    reject_total = result["reject_total"]

    why = result["why_explanation"]
    why_not = result["why_not_explanation"]

    audit_steps = generate_audit_trail(
        applicant_data, probability, decision, approve_total, reject_total
    )

    counterfactual_suggestions = generate_counterfactuals(
        applicant_data, model, predict_default_probability
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Arguments", "Counterfactuals", "Audit Trail"]
    )

    # ------------------------------------------------------------
    # Tab 1: Overview
    # ------------------------------------------------------------

    with tab1:
        st.subheader("Decision Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Default Probability", f"{probability:.2%}")

        with col2:
            st.metric("Decision", decision)
            st.metric("Argumentation Risk Signal", argumentation_risk_signal)

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Default Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 10], "color": "lightgreen"},
                        {"range": [10, 30], "color": "khaki"},
                        {"range": [30, 100], "color": "lightcoral"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": probability * 100,
                    },
                },
            )
        )

        st.plotly_chart(gauge, width="stretch")

        st.subheader("WHY Explanation")
        st.info(why)

        st.subheader("WHY-NOT Explanation")
        st.warning(why_not)

        st.subheader("AI Plain-Language Explanation")
        st.info(ai_explanation)
    # ------------------------------------------------------------
    # Tab 2: Arguments
    # ------------------------------------------------------------

    with tab2:
        st.subheader("Argument Strength Summary")

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

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Total Approval Strength", f"{approve_total:.3f}")

        with col4:
            st.metric("Total Rejection Strength", f"{reject_total:.3f}")

        argument_graph = result["argument_graph"]

        st.subheader("Argument Graph Summary")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Arguments", len(argument_graph["nodes"]))

        with col_b:
            st.metric("Attack Relations", len(argument_graph["attacks"]))

        with col_c:
            st.metric("Support Relations", len(argument_graph["supports"]))

        st.subheader("Argument Relations")

        relations = []

        for attack in argument_graph["attacks"]:
            relations.append(
                {
                    "From": attack["from"],
                    "To": attack["to"],
                    "Relation": "Attack",
                    "Reason": attack["reason"],
                }
            )

        for support in argument_graph["supports"]:
            relations.append(
                {
                    "From": support["from"],
                    "To": support["to"],
                    "Relation": "Support",
                    "Reason": support["reason"],
                }
            )

        st.dataframe(relations, width="stretch")

        st.subheader("Argumentation Risk Signal")
        st.info(argumentation_risk_signal)

        if reject_total > approve_total:
            st.error(
                "Rejection-supporting evidence dominates. "
                "This is an audit signal, not a separate final decision."
            )
        elif approve_total > reject_total:
            st.success(
                "Approval-supporting evidence dominates. "
                "This supports the policy decision but does not replace it."
            )
        else:
            st.warning(
                "Approval and rejection arguments are balanced. "
                "This indicates an ambiguous reasoning profile."
            )

        st.subheader("Arguments Supporting Approval")

        for argument in approve_arguments:
            st.success(f"{argument['name']} | Strength: {argument['strength']:.3f}")
            st.write(argument["text"])

            st.markdown("**Financial meaning**")
            st.write(argument["financial_meaning"])

            st.markdown("**Governance justification**")
            st.write(argument["governance_justification"])

            st.divider()

        st.subheader("Arguments Supporting Rejection")

        for argument in reject_arguments:
            st.error(f"{argument['name']} | Strength: {argument['strength']:.3f}")
            st.write(argument["text"])

            st.markdown("**Financial meaning**")
            st.write(argument["financial_meaning"])

            st.markdown("**Governance justification**")
            st.write(argument["governance_justification"])

            st.divider()

    # ------------------------------------------------------------
    # Tab 3: Counterfactuals
    # ------------------------------------------------------------

    with tab3:
        st.subheader("Counterfactual Improvement Suggestions")

        for suggestion in counterfactual_suggestions:
            st.markdown(f"### {suggestion['title']}")
            st.write(f"**Current value:** {suggestion['current']}")
            st.write(f"**Borderline target value:** {suggestion['target']}")
            st.write(f"**Required change:** {suggestion['change']}")
            st.info(suggestion["meaning"])

            if suggestion["new_approve_total"] is not None:
                col5, col6 = st.columns(2)

                with col5:
                    st.metric(
                        "New Approval Strength",
                        f"{suggestion['new_approve_total']:.3f}",
                        f"{suggestion['approval_change']:+.3f}",
                    )

                with col6:
                    st.metric(
                        "New Rejection Strength",
                        f"{suggestion['new_reject_total']:.3f}",
                        f"{suggestion['rejection_change']:+.3f}",
                    )

                col7, col8 = st.columns(2)

                with col7:
                    st.metric(
                        "New Predicted Default Probability",
                        f"{suggestion['new_probability']:.2%}",
                        f"{suggestion['probability_change']:+.2%}",
                    )

                with col8:
                    st.metric(
                        "New Decision",
                        suggestion["new_decision"],
                    )

                st.write(
                    f"Original decision: **{suggestion['original_decision']}** → "
                    f"New decision after change: **{suggestion['new_decision']}**"
                )

                if suggestion["new_reject_total"] > suggestion["new_approve_total"]:
                    st.warning(
                        "Even after this change, rejection arguments remain stronger."
                    )
                elif suggestion["new_approve_total"] > suggestion["new_reject_total"]:
                    st.success("After this change, approval arguments become stronger.")
                else:
                    st.info(
                        "After this change, approval and rejection arguments become balanced."
                    )

    # ------------------------------------------------------------
    # Tab 4: Audit Trail
    # ------------------------------------------------------------

    with tab4:
        st.subheader("Audit Trail")

        for step in audit_steps:
            st.write(step)

        st.subheader("Thesis Interpretation")
        st.write(
            "The predictive layer estimates the probability of default. "
            "The business decision layer converts this probability into Approve, Review, or Reject. "
            "The argumentation layer translates applicant characteristics into structured arguments. "
            "The counterfactual layer tests borderline changes and shows how each change affects both argument strength totals and the model's predicted probability. "
            "The audit trail records the reasoning path from input data to final decision."
        )

else:
    st.info("Enter applicant information in the sidebar and press Analyze Applicant.")
