import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config.argument_rules import ARGUMENT_RULES
from src.modeling import load_model, get_model_coefficients
from src.reporting import create_credit_explanation_report
from src.model_diagnostics import create_model_diagnostics_report


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Credit Scoring Explainer",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()
model_diagnostics = create_model_diagnostics_report(model)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def format_feature_name(feature: str) -> str:
    names = {
        "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
        "DebtRatio": "Debt Ratio",
        "MonthlyIncome": "Monthly Income",
        "NumberOfTimes90DaysLate": "90+ Days Late Payments",
        "age": "Age",
    }

    return names.get(feature, feature)


def format_important_number(value: float) -> str:
    """
    Formats numbers so tables are readable.
    Examples:
        0.0000  -> 0
        3.0000  -> 3
        1.0944  -> 1.0944
        23000.0 -> 23,000
    """

    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"

    return f"{value:.4f}".rstrip("0").rstrip(".")


def create_bar_chart(title: str, x_values: list, y_values: list, yaxis_title: str):
    chart = go.Figure(
        data=[
            go.Bar(
                x=x_values,
                y=y_values,
            )
        ]
    )

    chart.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return chart


def create_roc_curve_chart(validation_metrics: dict):
    roc_curve_data = validation_metrics.get("roc_curve")

    if roc_curve_data is None:
        return None

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=roc_curve_data["false_positive_rate"],
            y=roc_curve_data["true_positive_rate"],
            mode="lines",
            name="Logistic regression model",
        )
    )

    chart.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random model baseline",
            line=dict(dash="dash"),
        )
    )

    chart.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return chart


def render_dataset_overview():
    preprocessing_summary = model_diagnostics.get("preprocessing_summary")
    validation_metrics = model_diagnostics.get("validation_metrics")

    st.header("1. Dataset and Methodology Overview")

    st.write(
        "This app is based on a credit-scoring dataset. The goal is to estimate "
        "whether an applicant is likely to have serious repayment difficulty, and "
        "then explain the result in a clear and auditable way."
    )

    st.subheader("Dataset used")

    st.write(
        "The model is trained on `cs-training.csv`. Each row represents one credit "
        "applicant. The target variable is `SeriousDlqin2yrs`, which tells us whether "
        "the applicant had serious delinquency within the next two years."
    )

    if preprocessing_summary is not None:
        train_size = preprocessing_summary["train_size"]
        test_size = preprocessing_summary["test_size"]
        total_size = train_size + test_size

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total observations", f"{total_size:,}")

        with col2:
            st.metric("Training observations", f"{train_size:,}")

        with col3:
            st.metric("Test observations", f"{test_size:,}")

        st.write(
            "The data is split into two parts. The model learns from the training "
            "set and is then tested on a separate test set. This is important because "
            "we want to know how the model behaves on applicants it has not seen before."
        )

    st.subheader("What information does the model use?")

    st.markdown(
        """
    | Feature | Simple meaning | Predictive model | Explanation layer |
    |---|---|---:|---:|
    | Credit Utilization | How much of the applicant's available unsecured credit is being used. | ✅ | ✅ |
    | Age | Applicant's age. | ✅ | ❌ |
    | Debt Ratio | How heavy the applicant's debt burden is. | ✅ | ✅ |
    | Monthly Income | Applicant's reported monthly income. | ✅ | ✅ |
    | 90+ Days Late Payments | Number of serious late-payment events. | ✅ | ✅ |
    """
    )

    st.info(
        "The predictive model and the explanation layer are separate. The model uses "
        "age because it helps the statistical prediction. The explanation layer excludes "
        "age and focuses only on financial reasons, so the explanation is easier to audit."
    )

    st.subheader("Preprocessing: what changed in the data")

    st.write(
        "Preprocessing is not the main focus of the app, but it is important because "
        "it makes the model reliable and reproducible. Below is what actually happened "
        "to this dataset before logistic regression was trained."
    )

    if preprocessing_summary is not None:
        train_size = preprocessing_summary["train_size"]
        test_size = preprocessing_summary["test_size"]

        missing_before = preprocessing_summary["missing_values_before_imputation"]
        missing_after = preprocessing_summary["missing_values_after_imputation"]

        monthly_income_train_missing = missing_before["train"]["MonthlyIncome"]
        monthly_income_test_missing = missing_before["test"]["MonthlyIncome"]

        monthly_income_train_after = missing_after["train"]["MonthlyIncome"]
        monthly_income_test_after = missing_after["test"]["MonthlyIncome"]

        winsorization_bounds = preprocessing_summary["winsorization_bounds"]

        st.markdown("### 1. The dataset was split into training and test data")

        st.write(
            f"The original dataset was split into **{train_size:,} training observations** "
            f"and **{test_size:,} test observations**. The model learns from the training "
            f"set and is evaluated on the test set."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training observations", f"{train_size:,}")

        with col2:
            st.metric("Test observations", f"{test_size:,}")

        st.markdown("### 2. Missing income values were handled")

        st.write(
            "The main missing-value problem was `MonthlyIncome`. Instead of deleting "
            "many applicants, the missing income values were filled using the training-set mean."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Missing MonthlyIncome before",
                f"{monthly_income_train_missing:,} train / {monthly_income_test_missing:,} test",
            )

        with col2:
            st.metric(
                "Missing MonthlyIncome after",
                f"{monthly_income_train_after:,} train / {monthly_income_test_after:,} test",
            )

        st.markdown("### 3. Extreme values were limited with winsorization")

        st.write(
            "Some financial variables had extreme values. Winsorization keeps the applicant "
            "in the dataset, but limits very extreme values to a reasonable lower or upper bound. "
            "This prevents one unusual value from having too much influence on the model."
        )

        important_bounds = [
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfTimes90DaysLate",
        ]

        winsorization_rows = []

        for feature in important_bounds:
            bounds = winsorization_bounds[feature]

            winsorization_rows.append(
                {
                    "Feature": format_feature_name(feature),
                    "Lower bound": format_important_number(bounds["lower"]),
                    "Upper bound": format_important_number(bounds["upper"]),
                }
            )

        st.table(pd.DataFrame(winsorization_rows))
        st.caption(
            "Values below the lower bound or above the upper bound were clipped to these limits."
        )

        st.markdown("### 4. The variables were standardized")

        st.write(
            "The variables in this dataset are measured in very different units. "
            "For example, age is measured in years, monthly income is measured in money, "
            "and credit utilization is a ratio. Logistic regression works better when "
            "these variables are put on a comparable scale."
        )

        st.write(
            "To do this, the project uses `StandardScaler`. For each feature, the scaler "
            "subtracts the training-set mean and divides by the training-set standard deviation."
        )

        st.code(
            "standardized value = (original value - training mean) / training standard deviation"
        )

        scaler_info = preprocessing_summary["scaler"]
        feature_names = preprocessing_summary["features"]
        scaler_means = scaler_info["mean"]
        scaler_scales = scaler_info["scale"]

        scaler_rows = []

        for feature, mean_value, scale_value in zip(
            feature_names,
            scaler_means,
            scaler_scales,
        ):
            scaler_rows.append(
                {
                    "Feature": format_feature_name(feature),
                    "Training mean": format_important_number(mean_value),
                    "Training standard deviation": format_important_number(scale_value),
                }
            )

        st.table(pd.DataFrame(scaler_rows))
        st.caption(
            "These training-set values are stored and reused so every future applicant is "
            "transformed in the same way."
        )

        st.markdown("#### Example: standardizing Monthly Income")

        monthly_income_example = 2333

        monthly_income_mean = scaler_info["mean"][feature_names.index("MonthlyIncome")]

        monthly_income_std = scaler_info["scale"][feature_names.index("MonthlyIncome")]

        monthly_income_scaled = (
            monthly_income_example - monthly_income_mean
        ) / monthly_income_std

        st.write(
            f"Suppose an applicant has **MonthlyIncome = {monthly_income_example:,}**. "
            f"In the training data, the average MonthlyIncome is approximately "
            f"**{format_important_number(monthly_income_mean)}**, and the training standard "
            f"deviation is approximately **{format_important_number(monthly_income_std)}**."
        )

        st.code(
            f"standardized MonthlyIncome = "
            f"({monthly_income_example:,} - {format_important_number(monthly_income_mean)}) "
            f"/ {format_important_number(monthly_income_std)} "
            f"= {monthly_income_scaled:.3f}"
        )

        st.write(
            "This means the model does not read MonthlyIncome only as a raw money amount. "
            "It reads it relative to the training data. A negative standardized value means "
            "the applicant's income is below the training-set average."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Scaler", scaler_info["type"])

        with col2:
            st.metric("Fitted on", scaler_info["fitted_on"])

        st.markdown("### 5. The same preprocessing was applied to the test set")

        st.write(
            "The imputation values, winsorization bounds, and scaler were learned from "
            "the training set only. Then the same transformations were applied to the test set. "
            "This is important because the test set must stay unseen during training."
        )

        st.success(
            "Result: the model was trained on cleaned, bounded, and standardized data, "
            "while the test set remained a fair evaluation sample."
        )

    st.subheader("How good is the model?")

    st.write(
        "After preprocessing, we train a logistic regression model. The model does not "
        "directly give an explanation. First, it estimates a probability of default. "
        "Then we use the argumentation layer to explain the decision."
    )

    if validation_metrics is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy on test set", f"{validation_metrics['accuracy']:.4f}")

        with col2:
            st.metric("ROC-AUC on test set", f"{validation_metrics['roc_auc']:.4f}")

        st.write(
            "Accuracy tells us how often the model classifies applicants correctly. "
            "However, in credit scoring, accuracy alone can be misleading because most "
            "applicants do not default. A model can look accurate just by predicting "
            "'non-default' most of the time."
        )

        st.write(
            "ROC-AUC is more useful here because it measures how well the model separates "
            "higher-risk applicants from lower-risk applicants. A value of 0.5 would be "
            "close to random guessing. A value around 0.82 means the model has a good "
            "ability to rank applicants by risk."
        )

        roc_chart = create_roc_curve_chart(validation_metrics)

        if roc_chart is not None:
            st.plotly_chart(roc_chart, width="stretch")

        st.markdown("### Confusion matrix")

        confusion_matrix = validation_metrics["confusion_matrix"]

        true_non_default = confusion_matrix[0][0]
        false_default = confusion_matrix[0][1]
        false_non_default = confusion_matrix[1][0]
        true_default = confusion_matrix[1][1]

        confusion_df = pd.DataFrame(
            confusion_matrix,
            index=["Actual non-default", "Actual default"],
            columns=["Predicted non-default", "Predicted default"],
        )

        st.dataframe(confusion_df, width="stretch")

        st.write(
            "The confusion matrix shows what the model got right and wrong on the "
            "test set. In simple words:"
        )

        explanation_table = pd.DataFrame(
            [
                {
                    "Part": "Correct non-default predictions",
                    "Value": true_non_default,
                    "Meaning": "Applicants who did not default and were predicted as non-default.",
                },
                {
                    "Part": "False alarms",
                    "Value": false_default,
                    "Meaning": "Applicants who did not default but were predicted as default.",
                },
                {
                    "Part": "Missed defaults",
                    "Value": false_non_default,
                    "Meaning": "Applicants who did default but were predicted as non-default.",
                },
                {
                    "Part": "Correct default predictions",
                    "Value": true_default,
                    "Meaning": "Applicants who did default and were predicted as default.",
                },
            ]
        )

        st.dataframe(explanation_table, width="stretch", hide_index=True)

        st.warning(
            "The model is good at identifying many non-default cases, but it misses "
            "some actual defaults. This is common in imbalanced credit datasets, where "
            "default cases are much fewer than non-default cases. This is one reason "
            "why the thesis does not stop at prediction: it adds an argumentation layer "
            "to make the decision path more transparent and auditable."
        )

    st.subheader("What this means for the thesis")

    st.write(
        "The dataset is useful because it is large, contains real credit-risk signals, "
        "and allows us to train a meaningful logistic regression model. But it also has "
        "typical real-world problems: missing income values, extreme financial values, "
        "and an imbalanced target where defaults are much less frequent than non-defaults."
    )

    st.write(
        "For this reason, the project does not rely only on the model output. The model "
        "produces a probability of default, but the explanation system then builds "
        "financial arguments, calculates their strengths, compares approval and rejection "
        "support, and creates WHY / WHY-NOT explanations and an audit trail."
    )


def render_logistic_regression_overview():
    intercept, coefficients = get_model_coefficients(model)
    coefficient_strengths = model_diagnostics.get("coefficient_strengths")

    st.header("2. Logistic Regression Model")

    st.write(
        "The logistic regression model estimates the applicant's probability of default. "
        "It is used as the predictive layer of the thesis."
    )

    st.subheader("Model formula")

    st.code("z = β0 + β1x1 + β2x2 + ... + βnxn")
    st.code("PD = 1 / (1 + exp(-z))")

    st.write(
        "The model first calculates a linear score `z`. "
        "Then it converts this score into a probability of default."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Intercept", f"{intercept:.4f}")

    with col2:
        st.info(
            "Positive coefficient → increases estimated default risk. "
            "Negative coefficient → decreases estimated default risk."
        )

    st.subheader("Fitted model coefficients")

    coefficient_rows = []

    for feature, coefficient in coefficients.items():
        if coefficient > 0:
            direction = "Increases default risk"
        else:
            direction = "Decreases default risk"

        if feature == "age":
            role = "Predictive model only"
        elif feature in ARGUMENT_RULES:
            role = "Predictive model + explanation layer"
        else:
            role = "Predictive model"

        coefficient_rows.append(
            {
                "Feature": format_feature_name(feature),
                "Coefficient": round(coefficient, 6),
                "Absolute size": round(abs(coefficient), 6),
                "Direction": direction,
                "Role in thesis": role,
            }
        )

    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_df = coefficient_df.sort_values(
        by="Absolute size",
        ascending=False,
    )

    st.dataframe(coefficient_df, width="stretch", hide_index=True)

    coefficient_chart = create_bar_chart(
        title="Logistic Regression Coefficients",
        x_values=coefficient_df["Feature"].tolist(),
        y_values=coefficient_df["Coefficient"].tolist(),
        yaxis_title="Coefficient value",
    )

    st.plotly_chart(coefficient_chart, width="stretch")

    st.subheader("Simple interpretation")

    st.write(
        "**Credit Utilization** and **90+ Days Late Payments** are the main "
        "risk-increasing variables in the fitted model."
    )

    st.write(
        "**Age** is used by the predictive model, but it is not used in the "
        "argumentation explanation layer. This keeps the explanation focused on "
        "financial reasons."
    )

    st.write(
        "**Debt Ratio** and **Monthly Income** are included as financial variables. "
        "Their effect is read through the fitted coefficient and then connected to "
        "the explanation layer through argument strengths."
    )

    st.subheader("Connection with argument strength")

    st.write(
        "The thesis connects logistic regression with argumentation by using the "
        "absolute size of each coefficient as the base strength of an argument."
    )

    st.code("base_strength_j = |β_j| / max(|β|)")

    if coefficient_strengths is not None:
        strength_rows = []

        for feature, details in coefficient_strengths["normalized_strengths"].items():
            rule = ARGUMENT_RULES[feature]

            strength_rows.append(
                {
                    "Explanation feature": format_feature_name(feature),
                    "Coefficient": round(details["coefficient"], 6),
                    "Base strength": round(details["normalized_strength"], 3),
                    "Rule threshold": rule["threshold"],
                    "Risk direction": rule["risk_direction"],
                }
            )

        strength_df = pd.DataFrame(strength_rows)

        st.dataframe(strength_df, width="stretch", hide_index=True)

    st.success(
        "In short: logistic regression predicts the probability of default, "
        "and the argumentation layer explains the financial reasons behind the case."
    )


def render_applicant_form():
    st.header("2. Applicant Profile")

    st.write(
        "Enter an applicant profile below. The model will estimate the probability "
        "of default, then the argumentation layer will generate WHY / WHY-NOT "
        "explanations, counterfactuals, and an audit trail."
    )

    default_values = {
        "RevolvingUtilizationOfUnsecuredLines": 0.58,
        "DebtRatio": 0.42,
        "MonthlyIncome": 2333,
        "NumberOfTimes90DaysLate": 1,
        "age": 35,
    }

    with st.form("applicant_form"):
        st.subheader("Financial inputs")

        col1, col2 = st.columns(2)

        explanation_applicant_data = {}

        features = list(ARGUMENT_RULES.keys())

        for index, feature in enumerate(features):
            rule = ARGUMENT_RULES[feature]

            target_column = col1 if index % 2 == 0 else col2

            with target_column:
                raw_default_value = default_values.get(feature)

                if raw_default_value is None:
                    raw_default_value = rule["threshold"]

                default_value = float(raw_default_value)

                explanation_applicant_data[feature] = st.number_input(
                    label=format_feature_name(feature),
                    value=default_value,
                    help=rule["financial_meaning"],
                )

        st.subheader("Predictive-model-only input")

        age = st.number_input(
            label="Age",
            value=default_values["age"],
            help=(
                "Age is used by the predictive logistic regression model, "
                "but excluded from the argumentation explanation layer."
            ),
        )

        submitted = st.form_submit_button("Evaluate Applicant")

    model_applicant_data = explanation_applicant_data.copy()
    model_applicant_data["age"] = age

    return submitted, explanation_applicant_data, model_applicant_data


def render_decision_overview(report: dict):
    predictive = report["predictive_layer"]
    argumentation = report["argumentation_layer"]
    reconciliation = report["decision_reconciliation"]

    strength_summary = argumentation["strength_summary"]

    probability_of_default = predictive["probability_of_default"]
    business_decision = predictive["business_decision"]
    argument_decision = strength_summary["argument_decision"]

    approve_total = strength_summary["approve_total"]
    reject_total = strength_summary["reject_total"]

    st.subheader("Decision Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Probability of Default", f"{probability_of_default:.2%}")

    with col2:
        st.metric("Business Policy Decision", business_decision)

    with col3:
        st.metric("Argument-Based Decision", argument_decision)

    st.info(
        "Business policy: PD < 10% → Approve, "
        "10% ≤ PD ≤ 30% → Review, PD > 30% → Reject."
    )

    st.subheader("Decision Reconciliation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Reconciliation Status", reconciliation["status"])

    with col2:
        st.metric(
            "Dominant Argument Side",
            argument_decision,
        )

    st.write(reconciliation["explanation"])

    st.subheader("Argument Strength Balance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Approval Strength", f"{approve_total:.3f}")

    with col2:
        st.metric("Rejection Strength", f"{reject_total:.3f}")

    strength_chart = create_bar_chart(
        title="Approval vs Rejection Argument Strength",
        x_values=["Approval Strength", "Rejection Strength"],
        y_values=[approve_total, reject_total],
        yaxis_title="Total Strength",
    )

    st.plotly_chart(strength_chart, width="stretch")

    if argument_decision == "Reject":
        st.error("Rejection-supporting arguments dominate.")
    elif argument_decision == "Approve":
        st.success("Approval-supporting arguments dominate.")
    else:
        st.warning("Approval and rejection arguments are balanced.")


def render_predictive_layer(report: dict):
    predictive = report["predictive_layer"]

    st.subheader("Predictive Layer")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Probability of Default",
            f"{predictive['probability_of_default']:.2%}",
        )

    with col2:
        st.metric("Linear Score", f"{predictive['linear_score']:.3f}")

    with col3:
        st.metric("Business Decision", predictive["business_decision"])

    st.write(
        "The predictive layer estimates default risk using logistic regression. "
        "The output is a probability of default, which is then mapped to the "
        "business policy decision."
    )

    st.subheader("Feature Contributions")

    feature_contributions = predictive["feature_contributions"]

    contribution_df = pd.DataFrame(
        {
            "Feature": [
                format_feature_name(feature) for feature in feature_contributions.keys()
            ],
            "Contribution to linear score": list(feature_contributions.values()),
        }
    )

    st.dataframe(contribution_df, width="stretch")

    contribution_chart = create_bar_chart(
        title="Feature Contributions to Logistic Regression Linear Score",
        x_values=contribution_df["Feature"].tolist(),
        y_values=contribution_df["Contribution to linear score"].tolist(),
        yaxis_title="Contribution",
    )

    st.plotly_chart(contribution_chart, width="stretch")


def render_argumentation_layer(report: dict):
    argumentation = report["argumentation_layer"]
    strength_summary = argumentation["strength_summary"]

    st.subheader("Argumentation Layer")

    st.write(
        "The argumentation layer transforms selected financial signals into "
        "approval-supporting and rejection-supporting arguments. Each argument "
        "has a quantified strength."
    )

    st.code("strength = base_strength × activation_strength")
    st.code("base_strength = |β_j| / max(|β|)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Approval Strength", f"{strength_summary['approve_total']:.3f}")

    with col2:
        st.metric("Rejection Strength", f"{strength_summary['reject_total']:.3f}")

    with col3:
        st.metric("Argument Decision", strength_summary["argument_decision"])

    st.subheader("WHY Explanation")

    for item in argumentation["why_explanation"]:
        st.write(f"- {item}")

    st.subheader("WHY-NOT Explanation")

    for item in argumentation["why_not_explanation"]:
        st.write(f"- {item}")

    st.subheader("Argument Details")

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
            st.write("**Financial meaning**")
            st.write(arg["financial_meaning"])

            st.write("**Governance justification**")
            st.write(arg["governance_justification"])

        st.divider()

    st.subheader("Argument Graph")

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


def render_counterfactuals(report: dict):
    counterfactuals = report["counterfactuals"]

    st.subheader("Counterfactual Analysis")

    st.write(
        "Counterfactuals show what would need to change in the applicant profile "
        "to weaken rejection-supporting arguments and potentially improve the decision."
    )

    for suggestion in counterfactuals:
        st.markdown(f"### {suggestion['title']}")

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


def render_diagnostics_and_audit(report: dict):
    predictive = report["predictive_layer"]
    audit_record = report["audit_record"]
    model_diagnostics = predictive["model_diagnostics"]

    st.subheader("Model Diagnostics")

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

    with st.expander("Coefficient normalization"):
        coefficient_strengths = model_diagnostics["coefficient_strengths"]

        st.write("**Formula**")
        st.code("base_strength = |β_j| / max(|β|)")

        st.metric("Intercept", f"{coefficient_strengths['intercept']:.4f}")

        coefficient_rows = []

        for feature, details in coefficient_strengths["normalized_strengths"].items():
            coefficient_rows.append(
                {
                    "Feature": format_feature_name(feature),
                    "Coefficient": details["coefficient"],
                    "Absolute coefficient": details["absolute_coefficient"],
                    "Normalized strength": details["normalized_strength"],
                }
            )

        coefficient_df = pd.DataFrame(coefficient_rows)
        st.dataframe(coefficient_df, width="stretch")

        comparison_df = pd.DataFrame(model_diagnostics["base_strength_comparison"])
        comparison_df["feature"] = comparison_df["feature"].apply(format_feature_name)

        st.write("**Model-derived strengths vs configured rule strengths**")
        st.dataframe(comparison_df, width="stretch")

    with st.expander("Preprocessing summary"):
        preprocessing_summary = model_diagnostics.get("preprocessing_summary")

        if preprocessing_summary is not None:
            st.write("**Train/test split**")
            st.json(preprocessing_summary["split"])

            st.write("**Missing values before imputation**")
            st.json(preprocessing_summary["missing_values_before_imputation"])

            st.write("**Missing values after imputation**")
            st.json(preprocessing_summary["missing_values_after_imputation"])

            st.write("**Winsorization bounds**")
            st.json(preprocessing_summary["winsorization_bounds"])

            st.write("**Scaler**")
            st.json(
                {
                    "type": preprocessing_summary["scaler"]["type"],
                    "fitted_on": preprocessing_summary["scaler"]["fitted_on"],
                    "applied_to": preprocessing_summary["scaler"]["applied_to"],
                }
            )

    st.subheader("Audit Trail")

    audit_df = pd.DataFrame(audit_record["argumentation_layer"]["arguments"])
    st.dataframe(audit_df, width="stretch")

    with st.expander("Full audit record"):
        st.json(audit_record)


# ------------------------------------------------------------
# Main app layout
# ------------------------------------------------------------

st.title("🏦 Credit Scoring Explainer")

st.write(
    "A thesis-oriented system for credit-scoring explanations. "
    "It combines logistic regression, business decision thresholds, "
    "argumentation-based reasoning, WHY / WHY-NOT explanations, "
    "counterfactual analysis, and auditability."
)

with st.sidebar:
    with st.expander("1. Dataset and Methodology Overview", expanded=False):
        render_dataset_overview()

render_logistic_regression_overview()

st.divider()

submitted, explanation_applicant_data, model_applicant_data = render_applicant_form()

if submitted:
    report = create_credit_explanation_report(
        explanation_applicant_data=explanation_applicant_data,
        model_applicant_data=model_applicant_data,
        model=model,
    )

    st.divider()

    st.header("3. Applicant Evaluation Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Decision Overview",
            "Predictive Layer",
            "Argumentation Layer",
            "Counterfactuals",
            "Diagnostics & Audit",
        ]
    )

    with tab1:
        render_decision_overview(report)

    with tab2:
        render_predictive_layer(report)

    with tab3:
        render_argumentation_layer(report)

    with tab4:
        render_counterfactuals(report)

    with tab5:
        render_diagnostics_and_audit(report)
else:
    st.info(
        "Enter or adjust the applicant values above and press "
        "**Evaluate Applicant** to generate the full explanation report."
    )
