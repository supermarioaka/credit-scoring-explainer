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


def create_coefficient_chart(coefficient_df: pd.DataFrame):
    chart_df = coefficient_df.copy()
    chart_df["Absolute coefficient"] = chart_df["Coefficient"].abs()
    chart_df = chart_df.sort_values(
        by="Absolute coefficient",
        ascending=True,
    )

    chart = go.Figure()

    chart.add_trace(
        go.Bar(
            x=chart_df["Coefficient"],
            y=chart_df["Feature"],
            orientation="h",
            text=chart_df["Coefficient"].round(4),
            textposition="auto",
            hovertemplate=("<b>%{y}</b><br>Coefficient: %{x:.4f}<br><extra></extra>"),
        )
    )

    chart.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
    )

    chart.update_layout(
        title="Fitted Logistic Regression Coefficients",
        xaxis_title="Coefficient value",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
    )

    return chart


def create_pd_policy_gauge(probability_of_default: float):
    """
    Creates a half-circle gauge showing the applicant's probability of default
    inside the business policy zones.
    """

    pd_percent = probability_of_default * 100

    chart = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_percent,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Probability of Default"},
            gauge={
                "axis": {
                    "range": [0, 50],
                    "tickmode": "array",
                    "tickvals": [0, 10, 30, 50],
                    "ticktext": ["0%", "10%", "30%", "50%"],
                },
                "bar": {"color": "#1f77b4", "thickness": 0.22},
                "steps": [
                    {"range": [0, 10], "color": "#d9f2d9"},
                    {"range": [10, 30], "color": "#fff2cc"},
                    {"range": [30, 50], "color": "#f4cccc"},
                ],
                "threshold": {
                    "line": {"color": "#1f77b4", "width": 5},
                    "thickness": 0.85,
                    "value": pd_percent,
                },
            },
        )
    )

    chart.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return chart


def create_argument_strength_donut(approve_total: float, reject_total: float):
    """
    Creates a clean donut chart showing the strength balance between
    approval-supporting and rejection-supporting arguments.
    """

    chart = go.Figure(
        data=[
            go.Pie(
                labels=["Approval strength", "Rejection strength"],
                values=[approve_total, reject_total],
                hole=0.62,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Strength: %{value:.3f}<extra></extra>",
                marker=dict(colors=["#2e7d32", "#b91c1c"]),
            )
        ]
    )

    chart.update_layout(
        title="Argument Strength Balance",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
    )

    return chart


def get_decision_message(business_decision: str, probability_of_default: float) -> str:
    pd_text = f"{probability_of_default:.2%}"

    if business_decision == "Approve":
        return (
            f"The applicant is in the **Approve** zone because the estimated "
            f"probability of default is **{pd_text}**, below the 10% policy threshold."
        )

    if business_decision == "Review":
        return (
            f"The applicant is in the **Review** zone because the estimated "
            f"probability of default is **{pd_text}**, between 10% and 30%."
        )

    return (
        f"The applicant is in the **Reject** zone because the estimated "
        f"probability of default is **{pd_text}**, above the 30% policy threshold."
    )


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
        "This is the predictive layer of the system. The fitted model estimates "
        "the applicant's probability of default."
    )

    st.subheader("Model form")

    st.latex(r"z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n")

    st.latex(r"PD = \frac{1}{1 + e^{-z}}")

    st.write(
        "The model first calculates the linear score, then converts it into a "
        "probability of default."
    )

    st.subheader("Fitted model from our dataset")

    feature_order = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfTimes90DaysLate",
    ]

    feature_symbols = {
        "RevolvingUtilizationOfUnsecuredLines": r"x_{\mathrm{utilization}}",
        "age": r"x_{\mathrm{age}}",
        "DebtRatio": r"x_{\mathrm{debt}}",
        "MonthlyIncome": r"x_{\mathrm{income}}",
        "NumberOfTimes90DaysLate": r"x_{\mathrm{90dayslate}}",
    }

    fitted_terms = []

    for feature in feature_order:
        coefficient = coefficients[feature]
        sign = "+" if coefficient >= 0 else "-"
        fitted_terms.append(f"{sign} {abs(coefficient):.4f}{feature_symbols[feature]}")

    fitted_model = f"z = {intercept:.4f} " + " ".join(fitted_terms)

    st.latex(fitted_model)

    coefficient_rows = []

    for feature in feature_order:
        coefficient = coefficients[feature]

        if coefficient > 0:
            direction = "Increases default risk"
        else:
            direction = "Decreases default risk"

        if feature == "age":
            role = "Prediction only"
        else:
            role = "Prediction + explanation"

        coefficient_rows.append(
            {
                "Feature": format_feature_name(feature),
                "Coefficient": round(coefficient, 6),
                "Direction": direction,
                "Role": role,
            }
        )

    coefficient_df = pd.DataFrame(coefficient_rows)

    st.write(
        "The graph below shows how each fitted coefficient affects the estimated "
        "probability of default."
    )

    st.write(
        "**Positive coefficients** increase estimated default risk, while "
        "**negative coefficients** decrease estimated default risk."
    )

    st.write(
        "Based on the fitted model, **Credit Utilization** and "
        "**90+ Days Late Payments** are the strongest risk-increasing signals. "
        "**Age** is included in the predictive model, but it is excluded from the "
        "argumentation explanation layer so that the explanation focuses only on "
        "financial reasons."
    )

    coefficient_chart = create_coefficient_chart(coefficient_df)

    st.plotly_chart(coefficient_chart, width="stretch")

    st.caption(
        "Bars to the right increase estimated default risk. Bars to the left decrease "
        "estimated default risk. Larger bars mean stronger influence in the fitted model."
    )

    st.subheader("Connection to argument strength")

    st.write(
        "The explanation layer uses only the financial variables. The fitted "
        "coefficient of each financial feature gives the base strength of the "
        "corresponding argument."
    )

    st.latex(r"S_{\mathrm{base},j} = \frac{|\beta_j|}{\max_k |\beta_k|}")

    st.write(
        "This means that the strongest financial coefficient receives base strength "
        "equal to 1.000, and the remaining financial features are scaled relative to it."
    )

    if coefficient_strengths is not None:
        strength_rows = []

        for feature, details in coefficient_strengths["normalized_strengths"].items():
            rule = ARGUMENT_RULES[feature]

            strength_rows.append(
                {
                    "Feature": format_feature_name(feature),
                    "Coefficient": round(details["coefficient"], 6),
                    "Base strength": round(details["normalized_strength"], 3),
                    "Rule threshold": rule["threshold"],
                }
            )

        strength_df = pd.DataFrame(strength_rows)
        strength_df = strength_df.sort_values(
            by="Base strength",
            ascending=False,
        )

        st.dataframe(strength_df, width="stretch", hide_index=True)

    st.write(
        "For each applicant, the app then compares the applicant's value with the "
        "rule threshold. This produces the activation strength of the argument."
    )

    st.latex(r"S_j = S_{\mathrm{base},j} \times S_{\mathrm{activation},j}")

    st.write(
        "The final argument strengths are shown later in the applicant evaluation. "
        "This keeps the model section focused on the fitted coefficients, while the "
        "case-specific reasoning is shown only after an applicant is evaluated."
    )

    st.success(
        "In short: logistic regression provides the base importance of each financial "
        "argument, and the applicant profile activates those arguments during evaluation."
    )


def render_applicant_form():
    st.header("3. Applicant Profile")

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

    probability_of_default = predictive["probability_of_default"]
    business_decision = predictive["business_decision"]

    st.subheader("Predictive Layer")

    st.write(
        "The model estimates the applicant's probability of default and places "
        "the applicant inside the business policy thresholds."
    )

    st.plotly_chart(
        create_pd_policy_gauge(probability_of_default),
        width="stretch",
    )

    st.markdown("#### Business policy thresholds")

    policy_cards = [
        {
            "decision": "Approve",
            "threshold": "PD < 10%",
            "risk": "Low risk",
            "border": "#2e7d32",
            "background": "#e8f5e9",
        },
        {
            "decision": "Review",
            "threshold": "10% ≤ PD ≤ 30%",
            "risk": "Intermediate risk",
            "border": "#b7791f",
            "background": "#fff8e1",
        },
        {
            "decision": "Reject",
            "threshold": "PD > 30%",
            "risk": "High risk",
            "border": "#b91c1c",
            "background": "#fdecea",
        },
    ]

    cols = st.columns(3)

    for col, card in zip(cols, policy_cards):
        is_current = business_decision == card["decision"]

        border_width = "4px" if is_current else "1px"
        opacity = "1" if is_current else "0.55"
        badge = "Current applicant" if is_current else "&nbsp;"

        card_html = (
            f"<div style='"
            f"background-color:{card['background']};"
            f"border:{border_width} solid {card['border']};"
            f"border-radius:14px;"
            f"padding:18px;"
            f"min-height:145px;"
            f"opacity:{opacity};"
            f"box-shadow:0 4px 12px rgba(0,0,0,0.08);"
            f"'>"
            f"<div style='font-size:24px;font-weight:800;color:{card['border']};margin-bottom:8px;'>"
            f"{card['decision']}"
            f"</div>"
            f"<div style='font-size:18px;font-weight:600;margin-bottom:6px;'>"
            f"{card['threshold']}"
            f"</div>"
            f"<div style='font-size:15px;color:#444;'>"
            f"{card['risk']}"
            f"</div>"
            f"<div style='font-size:14px;font-weight:700;color:{card['border']};margin-top:14px;'>"
            f"{badge}"
            f"</div>"
            f"</div>"
        )

        with col:
            st.markdown(card_html, unsafe_allow_html=True)

    st.caption(
        f"The applicant's estimated probability of default is "
        f"{probability_of_default:.2%}, so the current zone is {business_decision}."
    )


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

    arguments = argumentation["arguments"]
    approve_total = strength_summary["approve_total"]
    reject_total = strength_summary["reject_total"]
    argument_decision = strength_summary["argument_decision"]

    st.subheader("Argumentation Layer")

    st.write(
        "The explanation layer converts the applicant's financial profile into "
        "approval-supporting and rejection-supporting arguments. Each argument receives "
        "a strength, and the side with the greater total strength becomes the dominant explanation."
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.plotly_chart(
            create_argument_strength_donut(
                approve_total=approve_total,
                reject_total=reject_total,
            ),
            width="stretch",
        )

    with col2:
        st.markdown("#### Final argument-based result")

        if argument_decision == "Reject":
            result_color = "#b91c1c"
            result_background = "#fdecea"
            result_text = "Rejection-supporting arguments dominate."
        elif argument_decision == "Approve":
            result_color = "#2e7d32"
            result_background = "#e8f5e9"
            result_text = "Approval-supporting arguments dominate."
        else:
            result_color = "#b7791f"
            result_background = "#fff8e1"
            result_text = "Approval and rejection arguments are balanced."

        result_html = (
            f"<div style='"
            f"background-color:{result_background};"
            f"border-left:7px solid {result_color};"
            f"border-radius:14px;"
            f"padding:22px;"
            f"box-shadow:0 4px 12px rgba(0,0,0,0.08);"
            f"'>"
            f"<div style='font-size:34px;font-weight:800;color:{result_color};margin-bottom:8px;'>"
            f"{argument_decision}"
            f"</div>"
            f"<div style='font-size:16px;color:#333;margin-bottom:14px;'>"
            f"{result_text}"
            f"</div>"
            f"<div style='font-size:16px;'>"
            f"<b>Approval strength:</b> {approve_total:.3f}<br>"
            f"<b>Rejection strength:</b> {reject_total:.3f}"
            f"</div>"
            f"</div>"
        )

        st.markdown(result_html, unsafe_allow_html=True)

    st.markdown("#### WHY / WHY-NOT explanation")

    why_col, why_not_col = st.columns(2)

    with why_col:
        st.markdown("##### WHY this result?")

        why_items = argumentation["why_explanation"]

        if argument_decision == "Reject":
            st.error(
                "The result is **Reject** because rejection-supporting arguments "
                "are stronger overall."
            )
        elif argument_decision == "Approve":
            st.success(
                "The result is **Approve** because approval-supporting arguments "
                "are stronger overall."
            )
        else:
            st.warning(
                "The result is **Review** because approval and rejection support are balanced."
            )

        if why_items:
            for item in why_items:
                st.write(f"- {item}")

    with why_not_col:
        st.markdown("##### WHY-NOT the opposite?")

        why_not_items = argumentation["why_not_explanation"]

        if argument_decision == "Reject":
            st.warning(
                f"The applicant is not approved because approval-supporting arguments "
                f"exist, but their total strength is only **{approve_total:.3f}**, "
                f"while rejection strength is **{reject_total:.3f}**."
            )

            if why_not_items:
                st.write("Approval-supporting reasons were considered:")
                for item in why_not_items:
                    st.write(f"- {item}")

        elif argument_decision == "Approve":
            st.warning(
                f"The applicant is not rejected because rejection-supporting arguments "
                f"exist, but their total strength is only **{reject_total:.3f}**, "
                f"while approval strength is **{approve_total:.3f}**."
            )

            if why_not_items:
                st.write("Rejection-supporting reasons were considered:")
                for item in why_not_items:
                    st.write(f"- {item}")

        else:
            st.info(
                "There is no clear opposite decision because the argument strengths are balanced, "
                "so the applicant is sent to Review."
            )

    st.markdown("#### How each argument strength is calculated")

    formula_html = (
        "<div style='background-color:#f8fafc;border:1px solid #d9e2ec;"
        "border-radius:16px;padding:22px;margin-top:10px;margin-bottom:18px;"
        "text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.05);'>"
        "<div style='font-size:15px;color:#555;margin-bottom:8px;'>Argument strength formula</div>"
        "<div style='font-size:30px;font-weight:800;color:#1f2937;'>"
        "Base strength × Activation strength = Final strength"
        "</div>"
        "</div>"
    )

    st.markdown(formula_html, unsafe_allow_html=True)

    with st.expander("What do these numbers mean?"):
        st.write(
            "This section explains the three numbers used to calculate each argument's strength."
        )

        info_cols = st.columns(3)

        with info_cols[0]:
            st.markdown(
                (
                    "<div style='background-color:white;border:1px solid #d9e2ec;"
                    "border-radius:14px;padding:16px;min-height:150px;"
                    "box-shadow:0 3px 10px rgba(0,0,0,0.04);'>"
                    "<div style='font-size:21px;font-weight:800;color:#111;margin-bottom:8px;'>"
                    "Base strength"
                    "</div>"
                    "<div style='font-size:15px;color:#444;'>"
                    "How important this financial feature is in the fitted logistic regression model."
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        with info_cols[1]:
            st.markdown(
                (
                    "<div style='background-color:white;border:1px solid #d9e2ec;"
                    "border-radius:14px;padding:16px;min-height:150px;"
                    "box-shadow:0 3px 10px rgba(0,0,0,0.04);'>"
                    "<div style='font-size:21px;font-weight:800;color:#111;margin-bottom:8px;'>"
                    "Activation strength"
                    "</div>"
                    "<div style='font-size:15px;color:#444;'>"
                    "How strongly the applicant activates the rule, based on the rule threshold."
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        with info_cols[2]:
            st.markdown(
                (
                    "<div style='background-color:white;border:1px solid #d9e2ec;"
                    "border-radius:14px;padding:16px;min-height:150px;"
                    "box-shadow:0 3px 10px rgba(0,0,0,0.04);'>"
                    "<div style='font-size:21px;font-weight:800;color:#111;margin-bottom:8px;'>"
                    "Final strength"
                    "</div>"
                    "<div style='font-size:15px;color:#444;'>"
                    "The final number added to either the approval side or the rejection side."
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        st.info(
            "The app adds all approval final strengths and all rejection final strengths. "
            "The larger total becomes the dominant argument-based explanation."
        )

    for arg in arguments:
        if arg["side"] == "Reject":
            border_color = "#b91c1c"
            background_color = "#fdecea"
            side_text = "Supports rejection"
        else:
            border_color = "#2e7d32"
            background_color = "#e8f5e9"
            side_text = "Supports approval"

        value = arg["value"]
        threshold = arg["threshold"]

        value_text = format_important_number(value)
        threshold_text = format_important_number(threshold)

        if arg["feature"] == "RevolvingUtilizationOfUnsecuredLines":
            value_text = f"{value:.2%}"
            threshold_text = f"{threshold:.0%}"

        strength_html = (
            f"<div style='"
            f"background-color:{background_color};"
            f"border:1px solid {border_color};"
            f"border-left:7px solid {border_color};"
            f"border-radius:16px;"
            f"padding:20px;"
            f"margin-bottom:18px;"
            f"box-shadow:0 3px 10px rgba(0,0,0,0.06);"
            f"'>"
            f"<div style='font-size:14px;font-weight:800;color:{border_color};margin-bottom:6px;'>"
            f"{side_text}"
            f"</div>"
            f"<div style='font-size:24px;font-weight:800;color:#222;margin-bottom:8px;'>"
            f"{arg['name']}"
            f"</div>"
            f"<div style='font-size:15px;color:#444;margin-bottom:16px;'>"
            f"Applicant value: <b>{value_text}</b> &nbsp; | &nbsp; "
            f"Rule threshold: <b>{threshold_text}</b>"
            f"</div>"
            f"<div style='display:flex;gap:12px;'>"
            f"<div style='flex:1;background-color:white;border-radius:12px;padding:14px;text-align:center;'>"
            f"<div style='font-size:13px;color:#666;'>Base strength</div>"
            f"<div style='font-size:30px;font-weight:800;color:#111;'>{arg['base_strength']:.3f}</div>"
            f"</div>"
            f"<div style='flex:1;background-color:white;border-radius:12px;padding:14px;text-align:center;'>"
            f"<div style='font-size:13px;color:#666;'>Activation strength</div>"
            f"<div style='font-size:30px;font-weight:800;color:#111;'>{arg['activation_strength']:.3f}</div>"
            f"</div>"
            f"<div style='flex:1;background-color:white;border-radius:12px;padding:14px;text-align:center;'>"
            f"<div style='font-size:13px;color:#666;'>Final strength</div>"
            f"<div style='font-size:30px;font-weight:800;color:{border_color};'>{arg['strength']:.3f}</div>"
            f"</div>"
            f"</div>"
            f"</div>"
        )

        st.markdown(strength_html, unsafe_allow_html=True)

    st.info(
        "The final argument-based decision is obtained by comparing the total approval strength "
        "with the total rejection strength."
    )


def render_counterfactuals(report: dict):
    counterfactuals = report["counterfactuals"]

    st.subheader("Counterfactual Analysis")

    st.write(
        "This section shows how a targeted change affects both the model prediction "
        "and the argumentation strength balance."
    )

    valid_counterfactuals = [
        suggestion
        for suggestion in counterfactuals
        if suggestion.get("new_approve_total") is not None
    ]

    if not valid_counterfactuals:
        st.success(
            "No major rule-based improvement is needed. "
            "The applicant does not currently trigger any rejection-supporting rule."
        )
        return

    def format_counterfactual_value(feature: str, value):
        if isinstance(value, str):
            return value

        if feature == "RevolvingUtilizationOfUnsecuredLines":
            return f"{value:.0%}"

        if feature == "MonthlyIncome":
            return f"{value:,.0f}"

        if feature == "NumberOfTimes90DaysLate":
            return f"{int(value)}"

        return format_important_number(value)

    def explain_target_choice(suggestion: dict) -> str:
        feature = suggestion["feature"]
        threshold = suggestion["threshold"]
        target_value = suggestion["target_value"]
        risk_direction = suggestion["risk_direction"]

        threshold_text = format_counterfactual_value(feature, threshold)
        target_text = format_counterfactual_value(feature, target_value)

        if feature == "NumberOfTimes90DaysLate":
            return (
                f"The target is {target_text} because this rule is triggered by any "
                f"90+ days late payment. Moving to 0 removes the serious late-payment signal."
            )

        if risk_direction == "above":
            return (
                f"The threshold is {threshold_text}. The target {target_text} is chosen "
                f"because it moves the applicant just below the risk threshold, so the rule "
                f"no longer supports rejection."
            )

        if risk_direction == "below":
            return (
                f"The threshold is {threshold_text}. The target {target_text} is chosen "
                f"because it moves the applicant just above the risk threshold, so the rule "
                f"no longer supports rejection."
            )

        return "The target is chosen because it crosses the relevant rule threshold."

    for suggestion in valid_counterfactuals:
        is_combined = suggestion["title"] == "Combined Improvement Scenario"

        if is_combined:
            border_color = "#1f2937"
            background_color = "#f8fafc"
            card_title = "Combined improvement scenario"
            current_text = "Current profile"
            target_text = "Improved profile"
            target_reason = (
                "This scenario applies all active rule-based improvements together."
            )
        else:
            border_color = "#2563eb"
            background_color = "#eff6ff"
            feature_name = format_feature_name(suggestion["feature"])
            card_title = f"Improve {feature_name}"

            current_text = format_counterfactual_value(
                suggestion["feature"],
                suggestion["current_value"],
            )
            target_text = format_counterfactual_value(
                suggestion["feature"],
                suggestion["target_value"],
            )
            target_reason = explain_target_choice(suggestion)

        original_approve = suggestion["original_approve_total"]
        original_reject = suggestion["original_reject_total"]
        new_approve = suggestion["new_approve_total"]
        new_reject = suggestion["new_reject_total"]

        original_argument_decision = suggestion["original_argument_decision"]
        new_argument_decision = suggestion["new_argument_decision"]

        max_strength = max(
            original_approve,
            original_reject,
            new_approve,
            new_reject,
            0.001,
        )

        original_approve_width = (original_approve / max_strength) * 100
        original_reject_width = (original_reject / max_strength) * 100
        new_approve_width = (new_approve / max_strength) * 100
        new_reject_width = (new_reject / max_strength) * 100

        header_html = (
            f"<div style='"
            f"background-color:{background_color};"
            f"border:1px solid {border_color};"
            f"border-left:8px solid {border_color};"
            f"border-radius:18px;"
            f"padding:24px;"
            f"margin-top:14px;"
            f"margin-bottom:22px;"
            f"box-shadow:0 5px 16px rgba(0,0,0,0.08);"
            f"'>"
            f"<div style='font-size:26px;font-weight:900;color:{border_color};margin-bottom:18px;'>"
            f"{card_title}"
            f"</div>"
            f"<div style='display:flex;align-items:center;gap:18px;margin-bottom:22px;'>"
            f"<div style='background-color:white;border-radius:14px;padding:18px 22px;min-width:160px;text-align:center;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05);'>"
            f"<div style='font-size:14px;color:#666;font-weight:800;margin-bottom:4px;'>CURRENT VALUE</div>"
            f"<div style='font-size:32px;font-weight:900;color:#111;'>{current_text}</div>"
            f"</div>"
            f"<div style='font-size:40px;font-weight:900;color:{border_color};'>→</div>"
            f"<div style='background-color:white;border-radius:14px;padding:18px 22px;min-width:160px;text-align:center;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05);'>"
            f"<div style='font-size:14px;color:#666;font-weight:800;margin-bottom:4px;'>SUGGESTED TARGET</div>"
            f"<div style='font-size:32px;font-weight:900;color:{border_color};'>{target_text}</div>"
            f"</div>"
            f"</div>"
            f"<div style='"
            f"background-color:white;"
            f"border-radius:16px;"
            f"padding:20px;"
            f"border:1px solid #d9e2ec;"
            f"box-shadow:0 2px 8px rgba(0,0,0,0.05);"
            f"'>"
            f"<div style='font-size:15px;font-weight:900;color:{border_color};margin-bottom:8px;'>"
            f"WHY THIS NUMBER?"
            f"</div>"
            f"<div style='font-size:20px;font-weight:700;color:#222;line-height:1.5;'>"
            f"{target_reason}"
            f"</div>"
            f"</div>"
        )

        st.markdown(header_html, unsafe_allow_html=True)

        before_after_html = (
            "<div style='display:flex;gap:18px;margin-bottom:26px;'>"
            # Before card
            "<div style='flex:1;background-color:white;border:1px solid #d9e2ec;"
            "border-radius:16px;padding:22px;box-shadow:0 4px 12px rgba(0,0,0,0.06);'>"
            "<div style='font-size:14px;font-weight:800;color:#6b7280;margin-bottom:4px;'>"
            "BEFORE CHANGE"
            "</div>"
            f"<div style='font-size:30px;font-weight:800;color:#111;margin-bottom:20px;'>"
            f"{original_argument_decision}"
            "</div>"
            "<div style='font-size:18px;font-weight:800;color:#2e7d32;margin-bottom:8px;'>"
            f"Approval strength: <span style='font-size:22px;'>{original_approve:.3f}</span>"
            "</div>"
            "<div style='height:16px;background-color:#e5e7eb;border-radius:999px;margin-bottom:18px;'>"
            f"<div style='height:16px;width:{original_approve_width:.1f}%;"
            "background-color:#2e7d32;border-radius:999px;'></div>"
            "</div>"
            "<div style='font-size:18px;font-weight:800;color:#b91c1c;margin-bottom:8px;'>"
            f"Rejection strength: <span style='font-size:22px;'>{original_reject:.3f}</span>"
            "</div>"
            "<div style='height:16px;background-color:#e5e7eb;border-radius:999px;'>"
            f"<div style='height:16px;width:{original_reject_width:.1f}%;"
            "background-color:#b91c1c;border-radius:999px;'></div>"
            "</div>"
            "</div>"
            # Arrow
            "<div style='display:flex;align-items:center;justify-content:center;"
            "font-size:36px;font-weight:800;color:#6b7280;'>"
            "→"
            "</div>"
            # After card
            "<div style='flex:1;background-color:white;border:1px solid #d9e2ec;"
            "border-radius:16px;padding:22px;box-shadow:0 4px 12px rgba(0,0,0,0.06);'>"
            "<div style='font-size:14px;font-weight:800;color:#6b7280;margin-bottom:4px;'>"
            "AFTER CHANGE"
            "</div>"
            f"<div style='font-size:30px;font-weight:800;color:#111;margin-bottom:20px;'>"
            f"{new_argument_decision}"
            "</div>"
            "<div style='font-size:18px;font-weight:800;color:#2e7d32;margin-bottom:8px;'>"
            f"Approval strength: <span style='font-size:22px;'>{new_approve:.3f}</span>"
            "</div>"
            "<div style='height:16px;background-color:#e5e7eb;border-radius:999px;margin-bottom:18px;'>"
            f"<div style='height:16px;width:{new_approve_width:.1f}%;"
            "background-color:#2e7d32;border-radius:999px;'></div>"
            "</div>"
            "<div style='font-size:18px;font-weight:800;color:#b91c1c;margin-bottom:8px;'>"
            f"Rejection strength: <span style='font-size:22px;'>{new_reject:.3f}</span>"
            "</div>"
            "<div style='height:16px;background-color:#e5e7eb;border-radius:999px;'>"
            f"<div style='height:16px;width:{new_reject_width:.1f}%;"
            "background-color:#b91c1c;border-radius:999px;'></div>"
            "</div>"
            "</div>"
            "</div>"
        )

        st.markdown(before_after_html, unsafe_allow_html=True)
        original_probability = suggestion["original_probability"]
        new_probability = suggestion["new_probability"]
        probability_change = suggestion["probability_change"]

        original_business_decision = suggestion["original_business_decision"]
        new_business_decision = suggestion["new_business_decision"]

        if probability_change < 0:
            change_color = "#2e7d32"
            change_label = f"{probability_change:.2%}"
        elif probability_change > 0:
            change_color = "#b91c1c"
            change_label = f"+{probability_change:.2%}"
        else:
            change_color = "#6b7280"
            change_label = "0.00%"

        predictive_impact_html = (
            "<div style='"
            "background-color:#ffffff;"
            "border:1px solid #d9e2ec;"
            "border-radius:16px;"
            "padding:18px;"
            "margin-top:6px;"
            "margin-bottom:22px;"
            "box-shadow:0 3px 10px rgba(0,0,0,0.05);"
            "'>"
            "<div style='font-size:18px;font-weight:900;color:#1f2937;margin-bottom:14px;'>"
            "Predictive impact"
            "</div>"
            "<div style='display:flex;align-items:center;gap:14px;'>"
            "<div style='flex:1;background-color:#f8fafc;border-radius:14px;padding:16px;text-align:center;'>"
            "<div style='font-size:13px;font-weight:800;color:#6b7280;margin-bottom:4px;'>BEFORE CHANGE</div>"
            f"<div style='font-size:28px;font-weight:900;color:#111;'>{original_probability:.2%}</div>"
            f"<div style='font-size:18px;font-weight:800;color:#374151;margin-top:4px;'>{original_business_decision}</div>"
            "</div>"
            "<div style='font-size:30px;font-weight:900;color:#6b7280;'>→</div>"
            "<div style='flex:1;background-color:#f8fafc;border-radius:14px;padding:16px;text-align:center;'>"
            "<div style='font-size:13px;font-weight:800;color:#6b7280;margin-bottom:4px;'>AFTER CHANGE</div>"
            f"<div style='font-size:28px;font-weight:900;color:#111;'>{new_probability:.2%}</div>"
            f"<div style='font-size:18px;font-weight:800;color:#374151;margin-top:4px;'>{new_business_decision}</div>"
            "</div>"
            "<div style='flex:1;background-color:#f8fafc;border-radius:14px;padding:16px;text-align:center;'>"
            "<div style='font-size:13px;font-weight:800;color:#6b7280;margin-bottom:4px;'>PD CHANGE</div>"
            f"<div style='font-size:30px;font-weight:900;color:{change_color};'>{change_label}</div>"
            "</div>"
            "</div>"
            "</div>"
        )

        st.markdown(predictive_impact_html, unsafe_allow_html=True)

        st.divider()


def render_diagnostics_and_audit(report: dict):
    predictive = report["predictive_layer"]
    argumentation = report["argumentation_layer"]
    counterfactuals = report["counterfactuals"]

    model_diagnostics = predictive["model_diagnostics"]
    validation_metrics = model_diagnostics.get("validation_metrics")
    strength_summary = argumentation["strength_summary"]

    probability_of_default = predictive["probability_of_default"]
    business_decision = predictive["business_decision"]

    approve_total = strength_summary["approve_total"]
    reject_total = strength_summary["reject_total"]
    argument_decision = strength_summary["argument_decision"]

    st.subheader("Diagnostics & Audit Summary")

    st.write(
        "A compact summary of the trained model, the current applicant decision, "
        "and the explanation logic used by the system."
    )

    # ------------------------------------------------------------
    # 1. Model validation summary
    # ------------------------------------------------------------

    st.markdown("#### 1. Model validation")

    if validation_metrics is not None:
        roc_auc = validation_metrics["roc_auc"]
        confusion_matrix = validation_metrics["confusion_matrix"]

        true_non_default = confusion_matrix[0][0]
        false_default = confusion_matrix[0][1]
        missed_defaults = confusion_matrix[1][0]
        true_default = confusion_matrix[1][1]

        validation_html = (
            "<div style='background-color:#ffffff;border:1px solid #d9e2ec;"
            "border-radius:18px;padding:24px;margin-bottom:24px;"
            "box-shadow:0 5px 16px rgba(0,0,0,0.06);'>"
            "<div style='display:flex;gap:18px;align-items:stretch;margin-bottom:20px;'>"
            "<div style='flex:1.1;background-color:#eff6ff;border-radius:18px;"
            "padding:24px;text-align:center;border:1px solid #2563eb;'>"
            "<div style='font-size:18px;font-weight:900;color:#2563eb;margin-bottom:6px;'>"
            "ROC-AUC"
            "</div>"
            f"<div style='font-size:56px;font-weight:900;color:#111;'>{roc_auc:.3f}</div>"
            "<div style='font-size:18px;font-weight:700;color:#333;margin-top:6px;'>"
            "The model is good at ranking applicants from lower to higher risk."
            "</div>"
            "</div>"
            "<div style='flex:1.6;background-color:#f8fafc;border-radius:18px;"
            "padding:24px;border:1px solid #d9e2ec;'>"
            "<div style='font-size:24px;font-weight:900;color:#1f2937;margin-bottom:10px;'>"
            "What this validation tells us"
            "</div>"
            "<div style='font-size:18px;color:#333;line-height:1.5;'>"
            "The model separates risk reasonably well. Its main limitation is that the dataset is imbalanced: "
            "most applicants do not default, so some actual default cases are still missed."
            "</div>"
            "</div>"
            "</div>"
            "<div style='font-size:22px;font-weight:900;color:#1f2937;margin-bottom:14px;'>"
            "Test set outcome summary"
            "</div>"
            "<div style='display:flex;gap:14px;'>"
            "<div style='flex:1;background-color:#e8f5e9;border-radius:16px;padding:18px;text-align:center;'>"
            "<div style='font-size:14px;font-weight:900;color:#2e7d32;'>CORRECT NON-DEFAULTS</div>"
            f"<div style='font-size:34px;font-weight:900;color:#111;'>{true_non_default:,}</div>"
            "</div>"
            "<div style='flex:1;background-color:#e8f5e9;border-radius:16px;padding:18px;text-align:center;'>"
            "<div style='font-size:14px;font-weight:900;color:#2e7d32;'>CORRECT DEFAULTS</div>"
            f"<div style='font-size:34px;font-weight:900;color:#111;'>{true_default:,}</div>"
            "</div>"
            "<div style='flex:1;background-color:#fff8e1;border-radius:16px;padding:18px;text-align:center;'>"
            "<div style='font-size:14px;font-weight:900;color:#b7791f;'>FALSE ALARMS</div>"
            f"<div style='font-size:34px;font-weight:900;color:#111;'>{false_default:,}</div>"
            "</div>"
            "<div style='flex:1;background-color:#fdecea;border-radius:16px;padding:18px;text-align:center;'>"
            "<div style='font-size:14px;font-weight:900;color:#b91c1c;'>MISSED DEFAULTS</div>"
            f"<div style='font-size:34px;font-weight:900;color:#111;'>{missed_defaults:,}</div>"
            "</div>"
            "</div>"
            "</div>"
        )

        st.markdown(validation_html, unsafe_allow_html=True)

    else:
        st.info("No validation metrics found in the saved model object.")
    # ------------------------------------------------------------
    # 2. Logistic regression base model
    # ------------------------------------------------------------

    st.markdown("#### 2. Logistic regression base model")

    intercept, coefficients = get_model_coefficients(model)

    feature_order = [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "age",
        "MonthlyIncome",
        "DebtRatio",
    ]

    feature_cards_html = (
        "<div style='background-color:#ffffff;border:1px solid #d9e2ec;"
        "border-radius:18px;padding:22px;margin-bottom:22px;"
        "box-shadow:0 5px 16px rgba(0,0,0,0.06);'>"
        "<div style='font-size:22px;font-weight:900;color:#1f2937;margin-bottom:8px;'>"
        "Base logistic regression coefficients"
        "</div>"
        "<div style='font-size:17px;color:#444;line-height:1.5;margin-bottom:18px;'>"
        "These are not applicant values. They are fixed model weights learned during training. "
        "They show how each feature generally affects estimated default risk before any specific applicant is entered."
        "</div>"
        "<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
    )

    for feature in feature_order:
        coefficient = coefficients[feature]

        if coefficient >= 0:
            color = "#b91c1c"
            background = "#fdecea"
            direction = "Risk ↑"
        else:
            color = "#2e7d32"
            background = "#e8f5e9"
            direction = "Risk ↓"

        feature_cards_html += (
            f"<div style='flex:1;min-width:160px;background-color:{background};"
            f"border:1px solid {color};border-left:6px solid {color};"
            f"border-radius:14px;padding:16px;'>"
            f"<div style='font-size:14px;font-weight:900;color:{color};margin-bottom:6px;'>"
            f"{direction}"
            f"</div>"
            f"<div style='font-size:16px;font-weight:900;color:#111;margin-bottom:8px;'>"
            f"{format_feature_name(feature)}"
            f"</div>"
            f"<div style='font-size:13px;font-weight:900;color:#6b7280;margin-bottom:4px;'>"
            f"MODEL WEIGHT"
            f"</div>"
            f"<div style='font-size:30px;font-weight:900;color:#111;'>"
            f"{coefficient:.4f}"
            f"</div>"
            f"</div>"
        )

    feature_cards_html += (
        "</div>"
        f"<div style='font-size:13px;color:#555;margin-top:14px;'>"
        f"Intercept: {intercept:.4f}"
        "</div>"
        "</div>"
    )

    st.markdown(feature_cards_html, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # 3. Current applicant explanation summary
    # ------------------------------------------------------------

    st.markdown("#### 3. Current applicant summary")

    valid_counterfactuals = [
        suggestion
        for suggestion in counterfactuals
        if suggestion.get("new_approve_total") is not None
    ]

    suggested_changes_html = ""

    if valid_counterfactuals:
        for suggestion in valid_counterfactuals:
            if suggestion["title"] == "Combined Improvement Scenario":
                change_title = "Combined improvement"
                change_text = "Apply all active rule-based improvements together."
            else:
                change_title = f"Improve {format_feature_name(suggestion['feature'])}"
                change_text = suggestion["change"]

            suggested_changes_html += (
                "<div style='background-color:white;border:1px solid #d9e2ec;"
                "border-left:6px solid #2563eb;border-radius:14px;padding:14px;"
                "margin-bottom:10px;'>"
                f"<div style='font-size:16px;font-weight:900;color:#2563eb;margin-bottom:4px;'>"
                f"{change_title}"
                f"</div>"
                f"<div style='font-size:17px;font-weight:800;color:#111;'>"
                f"{change_text}"
                f"</div>"
                "</div>"
            )
    else:
        suggested_changes_html = (
            "<div style='background-color:white;border:1px solid #d9e2ec;"
            "border-left:6px solid #2e7d32;border-radius:14px;padding:14px;'>"
            "<div style='font-size:17px;font-weight:800;color:#111;'>"
            "No major rule-based improvement is currently needed."
            "</div>"
            "</div>"
        )

    if business_decision == "Approve":
        decision_color = "#2e7d32"
        decision_background = "#e8f5e9"
    elif business_decision == "Review":
        decision_color = "#b7791f"
        decision_background = "#fff8e1"
    else:
        decision_color = "#b91c1c"
        decision_background = "#fdecea"

    summary_html = (
        "<div style='background-color:#ffffff;border:1px solid #d9e2ec;"
        "border-radius:18px;padding:22px;margin-bottom:22px;"
        "box-shadow:0 5px 16px rgba(0,0,0,0.06);'>"
        "<div style='display:flex;gap:16px;margin-bottom:18px;'>"
        f"<div style='flex:1;background-color:{decision_background};border-radius:16px;"
        f"padding:18px;text-align:center;border:1px solid {decision_color};'>"
        "<div style='font-size:13px;font-weight:900;color:#6b7280;'>PREDICTIVE RESULT</div>"
        f"<div style='font-size:36px;font-weight:900;color:{decision_color};'>{business_decision}</div>"
        f"<div style='font-size:18px;font-weight:800;color:#111;'>{probability_of_default:.2%} PD</div>"
        "</div>"
        "<div style='flex:1;background-color:#e8f5e9;border-radius:16px;padding:18px;text-align:center;'>"
        "<div style='font-size:13px;font-weight:900;color:#2e7d32;'>APPROVAL STRENGTH</div>"
        f"<div style='font-size:38px;font-weight:900;color:#111;'>{approve_total:.3f}</div>"
        "</div>"
        "<div style='flex:1;background-color:#fdecea;border-radius:16px;padding:18px;text-align:center;'>"
        "<div style='font-size:13px;font-weight:900;color:#b91c1c;'>REJECTION STRENGTH</div>"
        f"<div style='font-size:38px;font-weight:900;color:#111;'>{reject_total:.3f}</div>"
        "</div>"
        "<div style='flex:1;background-color:#f8fafc;border-radius:16px;padding:18px;text-align:center;'>"
        "<div style='font-size:13px;font-weight:900;color:#6b7280;'>ARGUMENT RESULT</div>"
        f"<div style='font-size:36px;font-weight:900;color:#111;'>{argument_decision}</div>"
        "</div>"
        "</div>"
        "<div style='background-color:#f8fafc;border-radius:16px;padding:18px;"
        "border:1px solid #d9e2ec;'>"
        "<div style='font-size:16px;font-weight:900;color:#1f2937;margin-bottom:10px;'>"
        "Suggested improvements"
        "</div>"
        f"{suggested_changes_html}"
        "</div>"
        "</div>"
    )

    st.markdown(summary_html, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Optional raw audit record
    # ------------------------------------------------------------

    with st.expander("Show full audit record"):
        st.json(report["audit_record"])


# ------------------------------------------------------------
# Main app layout
# ------------------------------------------------------------

SECTIONS = [
    "Preprocessing",
    "Logistic Regression",
    "Applicant Profile",
]


def go_to_section(section_name: str):
    st.session_state["selected_section"] = section_name
    st.rerun()


def render_home_page():
    validation_metrics = model_diagnostics.get("validation_metrics")
    preprocessing_summary = model_diagnostics.get("preprocessing_summary")

    if validation_metrics is not None:
        roc_auc_text = f"{validation_metrics['roc_auc']:.3f}"
    else:
        roc_auc_text = "-"

    if preprocessing_summary is not None:
        train_size = f"{preprocessing_summary['train_size']:,}"
        test_size = f"{preprocessing_summary['test_size']:,}"
    else:
        train_size = "-"
        test_size = "-"

    # ------------------------------------------------------------
    # Hero section
    # ------------------------------------------------------------

    hero_html = (
        "<div style='"
        "background:linear-gradient(135deg,#0f172a,#1e3a8a);"
        "border-radius:24px;"
        "padding:38px;"
        "margin-bottom:28px;"
        "box-shadow:0 8px 24px rgba(0,0,0,0.18);"
        "'>"
        "<div style='font-size:16px;font-weight:800;color:#bfdbfe;margin-bottom:10px;'>"
        "Explainable Credit Scoring · Argumentation · Auditability"
        "</div>"
        "<div style='font-size:46px;font-weight:900;color:white;line-height:1.1;margin-bottom:14px;'>"
        "Turning credit-risk predictions into transparent, auditable decisions"
        "</div>"
        "<div style='font-size:20px;color:#dbeafe;line-height:1.5;max-width:980px;'>"
        "This thesis application goes beyond a simple prediction. It estimates credit risk, "
        "translates financial signals into structured arguments, explains WHY and WHY-NOT, "
        "tests possible improvements through counterfactuals, and records the full decision path "
        "for auditability."
        "</div>"
        "</div>"
    )

    st.markdown(hero_html, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Key thesis highlights
    # ------------------------------------------------------------

    highlight_cols = st.columns(4)

    highlights = [
        {
            "title": "Model",
            "value": "Logistic Regression",
            "text": "Transparent predictive layer",
            "color": "#2563eb",
            "background": "#eff6ff",
        },
        {
            "title": "ROC-AUC",
            "value": roc_auc_text,
            "text": "Risk ranking quality",
            "color": "#7c3aed",
            "background": "#f3e8ff",
        },
        {
            "title": "Training / Test",
            "value": f"{train_size} / {test_size}",
            "text": "Reproducible split",
            "color": "#b7791f",
            "background": "#fff8e1",
        },
        {
            "title": "Explanation",
            "value": "WHY / WHY-NOT",
            "text": "Argument-based reasoning",
            "color": "#2e7d32",
            "background": "#e8f5e9",
        },
    ]

    for col, item in zip(highlight_cols, highlights):
        card_html = (
            f"<div style='"
            f"background-color:{item['background']};"
            f"border:1px solid {item['color']};"
            f"border-left:7px solid {item['color']};"
            f"border-radius:18px;"
            f"padding:18px;"
            f"min-height:150px;"
            f"box-shadow:0 4px 12px rgba(0,0,0,0.06);"
            f"'>"
            f"<div style='font-size:13px;font-weight:900;color:{item['color']};margin-bottom:8px;'>"
            f"{item['title']}"
            f"</div>"
            f"<div style='font-size:26px;font-weight:900;color:#111;margin-bottom:8px;'>"
            f"{item['value']}"
            f"</div>"
            f"<div style='font-size:14px;color:#444;'>"
            f"{item['text']}"
            f"</div>"
            f"</div>"
        )

        with col:
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Thesis workflow
    # ------------------------------------------------------------

    st.markdown("### Thesis workflow")

    workflow_html = (
        "<div style='"
        "background-color:#ffffff;"
        "border:1px solid #d9e2ec;"
        "border-radius:20px;"
        "padding:24px;"
        "margin-bottom:28px;"
        "box-shadow:0 5px 16px rgba(0,0,0,0.06);"
        "'>"
        "<div style='display:flex;align-items:center;gap:12px;text-align:center;'>"
        "<div style='flex:1;background-color:#eff6ff;border-radius:16px;padding:16px;'>"
        "<div style='font-size:24px;font-weight:900;color:#2563eb;'>1</div>"
        "<div style='font-size:17px;font-weight:900;color:#111;'>Preprocess</div>"
        "<div style='font-size:13px;color:#555;'>Clean and standardize data</div>"
        "</div>"
        "<div style='font-size:28px;font-weight:900;color:#94a3b8;'>→</div>"
        "<div style='flex:1;background-color:#f3e8ff;border-radius:16px;padding:16px;'>"
        "<div style='font-size:24px;font-weight:900;color:#7c3aed;'>2</div>"
        "<div style='font-size:17px;font-weight:900;color:#111;'>Predict</div>"
        "<div style='font-size:13px;color:#555;'>Estimate default risk</div>"
        "</div>"
        "<div style='font-size:28px;font-weight:900;color:#94a3b8;'>→</div>"
        "<div style='flex:1;background-color:#fff8e1;border-radius:16px;padding:16px;'>"
        "<div style='font-size:24px;font-weight:900;color:#b7791f;'>3</div>"
        "<div style='font-size:17px;font-weight:900;color:#111;'>Argue</div>"
        "<div style='font-size:13px;color:#555;'>Build financial arguments</div>"
        "</div>"
        "<div style='font-size:28px;font-weight:900;color:#94a3b8;'>→</div>"
        "<div style='flex:1;background-color:#e8f5e9;border-radius:16px;padding:16px;'>"
        "<div style='font-size:24px;font-weight:900;color:#2e7d32;'>4</div>"
        "<div style='font-size:17px;font-weight:900;color:#111;'>Explain</div>"
        "<div style='font-size:13px;color:#555;'>WHY / WHY-NOT and audit</div>"
        "</div>"
        "</div>"
        "</div>"
    )

    st.markdown(workflow_html, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Navigation cards
    # ------------------------------------------------------------

    st.markdown("### Explore the system")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            (
                "<div style='background-color:#f8fafc;border:1px solid #d9e2ec;"
                "border-radius:18px;padding:22px;min-height:185px;"
                "box-shadow:0 4px 12px rgba(0,0,0,0.06);'>"
                "<div style='font-size:26px;font-weight:900;color:#2563eb;margin-bottom:8px;'>"
                "1. Preprocessing"
                "</div>"
                "<div style='font-size:16px;color:#444;line-height:1.5;'>"
                "See how the dataset was split, cleaned, winsorized, and standardized "
                "before model training."
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if st.button("Open Preprocessing", width="stretch"):
            go_to_section("Preprocessing")

    with col2:
        st.markdown(
            (
                "<div style='background-color:#f8fafc;border:1px solid #d9e2ec;"
                "border-radius:18px;padding:22px;min-height:185px;"
                "box-shadow:0 4px 12px rgba(0,0,0,0.06);'>"
                "<div style='font-size:26px;font-weight:900;color:#7c3aed;margin-bottom:8px;'>"
                "2. Logistic Regression"
                "</div>"
                "<div style='font-size:16px;color:#444;line-height:1.5;'>"
                "Inspect the fitted model, coefficients, validation, and how model weights "
                "connect to argument strength."
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if st.button("Open Logistic Regression", width="stretch"):
            go_to_section("Logistic Regression")

    with col3:
        st.markdown(
            (
                "<div style='background-color:#f8fafc;border:1px solid #d9e2ec;"
                "border-radius:18px;padding:22px;min-height:185px;"
                "box-shadow:0 4px 12px rgba(0,0,0,0.06);'>"
                "<div style='font-size:26px;font-weight:900;color:#2e7d32;margin-bottom:8px;'>"
                "3. Applicant Evaluation"
                "</div>"
                "<div style='font-size:16px;color:#444;line-height:1.5;'>"
                "Enter an applicant profile and generate prediction, argumentation, "
                "counterfactuals, and audit summary."
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if st.button("Open Applicant Profile", width="stretch"):
            go_to_section("Applicant Profile")


def render_applicant_profile_page():
    submitted, explanation_applicant_data, model_applicant_data = (
        render_applicant_form()
    )

    if submitted:
        report = create_credit_explanation_report(
            explanation_applicant_data=explanation_applicant_data,
            model_applicant_data=model_applicant_data,
            model=model,
        )

        st.divider()

        st.header("Applicant Evaluation Results")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Predictive Layer",
                "Argumentation Layer",
                "Counterfactuals",
                "Diagnostics & Audit",
            ]
        )

        with tab1:
            render_decision_overview(report)

        with tab2:
            render_argumentation_layer(report)

        with tab3:
            render_counterfactuals(report)

        with tab4:
            render_diagnostics_and_audit(report)

    else:
        st.info(
            "Enter or adjust the applicant values above and press "
            "**Evaluate Applicant** to generate the full explanation report."
        )


if "selected_section" not in st.session_state:
    st.session_state["selected_section"] = None


if st.session_state["selected_section"] is None:
    render_home_page()

else:
    with st.sidebar:
        st.title("Navigation")

        selected_section = st.radio(
            "Choose section",
            SECTIONS,
            index=SECTIONS.index(st.session_state["selected_section"]),
        )

        st.session_state["selected_section"] = selected_section

        st.divider()

        if st.button("Back to home", width="stretch"):
            st.session_state["selected_section"] = None
            st.rerun()

    if st.session_state["selected_section"] == "Preprocessing":
        render_dataset_overview()

    elif st.session_state["selected_section"] == "Logistic Regression":
        render_logistic_regression_overview()

    elif st.session_state["selected_section"] == "Applicant Profile":
        render_applicant_profile_page()
