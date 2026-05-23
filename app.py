import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.generic_rule_pipeline import (
    analyze_dataset_for_argument_rules,
    evaluate_applicant_with_generated_rules,
)


st.set_page_config(
    page_title="Credit Scoring Explainer",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Credit Scoring Explainer")
st.write(
    "A generic explainable credit-scoring system that turns dataset patterns "
    "into candidate argumentation rules and auditable applicant explanations."
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def count_rule_quality(suggested_rules):
    counts = {}

    for rule in suggested_rules.values():
        quality = rule.get("rule_quality", "Unknown")
        counts[quality] = counts.get(quality, 0) + 1

    return counts


def make_readable_feature_name(feature):
    return (
        feature.replace("_", " ")
        .replace("-", " ")
        .replace("NumberOf", "Number of ")
        .replace("Times", " Times")
        .replace("Days", " Days")
    )


# ------------------------------------------------------------
# 1. Dataset Upload and Analysis
# ------------------------------------------------------------

st.header("1. Upload & Analyze Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No dataset uploaded. Using the default credit dataset.")
    df = pd.read_csv("data/cs-training.csv")

st.write("Dataset preview:")
st.dataframe(df.head(), width="stretch")

risk_outcome_column = st.selectbox(
    "Select the risk outcome column",
    df.columns,
    index=list(df.columns).index("SeriousDlqin2yrs")
    if "SeriousDlqin2yrs" in df.columns
    else 0,
    help=(
        "This is the column that represents the outcome we want to explain, "
        "for example default, bad loan, fraud, churn, or another risk event."
    ),
)

if st.button("Analyze Dataset"):
    with st.spinner("Analyzing dataset and generating candidate rules..."):
        analysis_result = analyze_dataset_for_argument_rules(
            df=df,
            target_column=risk_outcome_column,
        )

    st.session_state["analysis_result"] = analysis_result
    st.session_state["risk_outcome_column"] = risk_outcome_column


# ------------------------------------------------------------
# 2. Rule Summary
# ------------------------------------------------------------

if "analysis_result" in st.session_state:
    analysis_result = st.session_state["analysis_result"]

    suggested_rules = analysis_result["suggested_rules"]
    interpreted_rules = analysis_result["interpreted_rules"]
    validation = analysis_result["validation"]
    llm_analysis = analysis_result["llm_analysis"]

    st.header("2. Dataset Rule Summary")

    rule_quality_counts = count_rule_quality(suggested_rules)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Candidate Rules", len(suggested_rules))

    with col2:
        st.metric("Validation", "Passed" if validation["is_valid"] else "Failed")

    with col3:
        st.metric("Outcome Column", st.session_state["risk_outcome_column"])

    st.subheader("Rule Quality Distribution")

    if rule_quality_counts:
        quality_df = pd.DataFrame(
            {
                "Rule quality": list(rule_quality_counts.keys()),
                "Count": list(rule_quality_counts.values()),
            }
        )

        st.dataframe(quality_df, width="stretch")

        quality_chart = go.Figure(
            data=[
                go.Bar(
                    x=quality_df["Rule quality"],
                    y=quality_df["Count"],
                )
            ]
        )

        quality_chart.update_layout(
            title="Candidate Rule Quality Distribution",
            yaxis_title="Number of rules",
        )

        st.plotly_chart(quality_chart, width="stretch")
    else:
        st.warning("No candidate rules were generated.")

    st.subheader("Governance Status")

    st.info(
        "The generated rules are candidate explanation rules. "
        "They are not automatically approved decision rules. "
        "They should be reviewed for financial meaning, fairness, stability, "
        "and regulatory acceptability."
    )

    if "unavailable" in llm_analysis["llm_status"].lower():
        st.warning(
            "LLM governance analysis is currently unavailable. "
            "The system is using template-based governance explanations instead."
        )
    elif "error" in llm_analysis["llm_status"].lower():
        st.warning(
            "LLM governance analysis could not be generated. "
            "The system is using fallback governance explanations."
        )
    else:
        st.success("LLM governance analysis generated successfully.")
        st.write(llm_analysis["llm_explanation"])

    # ------------------------------------------------------------
    # 3. Applicant Profile
    # ------------------------------------------------------------

    st.header("3. Applicant Profile")

    st.write(
        "Enter applicant values for the features used by the generated candidate rules."
    )

    applicant_data = {}

    for feature, rule in suggested_rules.items():
        readable_name = make_readable_feature_name(feature)

        with st.container(border=True):
            st.markdown(f"**{readable_name}**")
            st.caption(
                f"Risk rule: value is {rule['risk_direction']} "
                f"{rule['threshold']} | Rule quality: {rule['rule_quality']}"
            )

            applicant_data[feature] = st.number_input(
                label=f"Applicant value for {readable_name}",
                value=float(rule["threshold"]),
                key=f"input_{feature}",
            )

    if st.button("Evaluate Applicant"):
        try:
            applicant_result = evaluate_applicant_with_generated_rules(
                applicant_data=applicant_data,
                rule_set=suggested_rules,
            )

            st.session_state["applicant_result"] = applicant_result

        except ValueError as error:
            st.error(str(error))


# ------------------------------------------------------------
# 4. Applicant Decision and Explanation
# ------------------------------------------------------------

if "applicant_result" in st.session_state:
    applicant_result = st.session_state["applicant_result"]

    approve_arguments = applicant_result["approve_arguments"]
    reject_arguments = applicant_result["reject_arguments"]
    approve_total = applicant_result["approve_total"]
    reject_total = applicant_result["reject_total"]
    argument_decision = applicant_result["argument_decision"]

    st.header("4. Decision & Explanation")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Arguments", "Audit Explanation", "Advanced Details"]
    )

    with tab1:
        st.subheader("Argument-Based Decision Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Approval Strength", f"{approve_total:.3f}")

        with col2:
            st.metric("Rejection Strength", f"{reject_total:.3f}")

        with col3:
            st.metric("Argument-Based Decision", argument_decision)

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

        if reject_total > approve_total:
            st.error("Rejection-supporting arguments dominate.")
        elif approve_total > reject_total:
            st.success("Approval-supporting arguments dominate.")
        else:
            st.warning("Approval and rejection arguments are balanced.")

    with tab2:
        st.subheader("Main Approval Arguments")

        if not approve_arguments:
            st.info("No approval-supporting arguments generated.")

        for argument in approve_arguments:
            st.success(f"{argument['name']} | Strength: {argument['strength']:.3f}")

            st.write(argument["text"])

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Base Strength", f"{argument['base_strength']:.3f}")

            with col_b:
                st.metric(
                    "Activation Strength",
                    f"{argument['activation_strength']:.3f}",
                )

            with col_c:
                st.metric(
                    "Distance",
                    f"{argument['distance_from_threshold']:.3f}",
                )

            st.caption(argument["strength_formula"])

            st.markdown("**Financial meaning**")
            st.write(argument["financial_meaning"])

            st.markdown("**Governance justification**")
            st.write(argument["governance_justification"])

            st.divider()

        st.subheader("Main Rejection Arguments")

        if not reject_arguments:
            st.info("No rejection-supporting arguments generated.")

        for argument in reject_arguments:
            st.error(f"{argument['name']} | Strength: {argument['strength']:.3f}")

            st.write(argument["text"])

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Base Strength", f"{argument['base_strength']:.3f}")

            with col_b:
                st.metric(
                    "Activation Strength",
                    f"{argument['activation_strength']:.3f}",
                )

            with col_c:
                st.metric(
                    "Distance",
                    f"{argument['distance_from_threshold']:.3f}",
                )

            st.caption(argument["strength_formula"])

            st.markdown("**Financial meaning**")
            st.write(argument["financial_meaning"])

            st.markdown("**Governance justification**")
            st.write(argument["governance_justification"])

            st.divider()

    with tab3:
        st.subheader("Audit Explanation")

        st.write(
            "The system analyzes the dataset and proposes candidate argumentation "
            "rules based on statistical diagnostics. Each rule is assigned a quality "
            "label and a governance note."
        )

        st.write(
            "For the applicant, every generated rule creates either an approval-supporting "
            "or rejection-supporting argument. The strength of each argument is computed as:"
        )

        st.code("strength = base_strength * activation_strength")

        st.write(
            "The final argument-based decision is produced by comparing the total "
            "approval-supporting strength against the total rejection-supporting strength."
        )

        st.write(
            "This decision should be understood as an auditable reasoning signal, "
            "not as an automatically approved production banking decision."
        )

    with tab4:
        st.subheader("Advanced / Developer Details")

        with st.expander("Raw suggested rules"):
            st.write(suggested_rules)

        with st.expander("Raw interpreted rules"):
            st.write(interpreted_rules)

        with st.expander("Validation details"):
            st.write(validation)

        with st.expander("LLM status and explanation"):
            st.write(llm_analysis)
