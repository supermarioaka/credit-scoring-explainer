import streamlit as st
import plotly.graph_objects as go

from src.modeling import load_model, predict_default_probability
from src.decision_policy import classify_decision
from src.argumentation import generate_arguments, generate_why_explanation
from src.audit import generate_audit_trail
from src.counterfactual import generate_counterfactuals


st.set_page_config(page_title="Credit Scoring Explainer", page_icon="🏦", layout="wide")

model = load_model()

st.title("🏦 Credit Scoring Explainer")
st.write(
    "A thesis-oriented app for explainable and auditable credit scoring decisions."
)

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

    probability = predict_default_probability(model, applicant_data)
    decision = classify_decision(probability)

    approve_arguments, reject_arguments, approve_total, reject_total = (
        generate_arguments(applicant_data)
    )
    why, why_not = generate_why_explanation(
        decision, approve_arguments, reject_arguments
    )
    audit_steps = generate_audit_trail(
        applicant_data, probability, decision, approve_total, reject_total
    )

    counterfactual_suggestions = generate_counterfactuals(
        applicant_data, model, predict_default_probability
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Arguments", "Counterfactuals", "Audit Trail"]
    )

    with tab1:
        st.subheader("Decision Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Default Probability", f"{probability:.2%}")

        with col2:
            st.metric("Decision", decision)

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

        if reject_total > approve_total:
            st.error("Argument-based conclusion: Rejection arguments are stronger.")
        elif approve_total > reject_total:
            st.success("Argument-based conclusion: Approval arguments are stronger.")
        else:
            st.warning(
                "Argument-based conclusion: Approval and rejection arguments are balanced."
            )

        st.subheader("Arguments Supporting Approval")
        for argument in approve_arguments:
            st.success(f"{argument['name']} | Strength: {argument['strength']:.3f}")
            st.write(argument["text"])

        st.subheader("Arguments Supporting Rejection")
        for argument in reject_arguments:
            st.error(f"{argument['name']} | Strength: {argument['strength']:.3f}")
            st.write(argument["text"])

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
