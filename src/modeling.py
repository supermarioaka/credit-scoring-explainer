import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/credit_model.pkl"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTimes90DaysLate",
]


def load_model():
    return joblib.load(MODEL_PATH)


def create_input_dataframe(applicant_data):
    return pd.DataFrame([applicant_data], columns=FEATURES)


def get_final_estimator(model):
    """
    If the model is a Pipeline, return the final estimator.
    If it is already a model, return it directly.
    """

    if hasattr(model, "steps"):
        return model.steps[-1][1]

    return model


def get_preprocessor(model):
    """
    If the model is a Pipeline, return everything before the final estimator.
    """

    if hasattr(model, "steps"):
        return model[:-1]

    return None


def predict_default_probability(model, applicant_data):
    input_df = create_input_dataframe(applicant_data)
    probability = model.predict_proba(input_df)[0][1]
    return probability


def compute_linear_score(model, applicant_data):
    input_df = create_input_dataframe(applicant_data)
    z = model.decision_function(input_df)[0]
    return z


def get_model_coefficients(model):
    estimator = get_final_estimator(model)

    coefficients = estimator.coef_[0]
    intercept = estimator.intercept_[0]

    return intercept, dict(zip(FEATURES, coefficients))


def compute_feature_contributions(model, applicant_data):
    input_df = create_input_dataframe(applicant_data)

    estimator = get_final_estimator(model)
    preprocessor = get_preprocessor(model)

    if preprocessor is not None:
        transformed_values = preprocessor.transform(input_df)[0]
    else:
        transformed_values = input_df.iloc[0].values

    coefficients = estimator.coef_[0]

    contributions = {}

    for feature, value, coefficient in zip(FEATURES, transformed_values, coefficients):
        contributions[feature] = value * coefficient

    return contributions


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
