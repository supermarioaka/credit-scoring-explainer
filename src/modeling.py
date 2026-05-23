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


def load_model(model_path: str = MODEL_PATH):
    return joblib.load(model_path)


def create_input_dataframe(applicant_data: dict) -> pd.DataFrame:
    return pd.DataFrame([applicant_data], columns=FEATURES)


def is_model_bundle(model_object) -> bool:
    return isinstance(model_object, dict) and "model" in model_object


def get_final_estimator(model_object):
    if is_model_bundle(model_object):
        return model_object["model"]

    if hasattr(model_object, "steps"):
        return model_object.steps[-1][1]

    return model_object


def get_preprocessor(model_object):
    if hasattr(model_object, "steps"):
        return model_object[:-1]

    return None


def apply_bundle_preprocessing(
    model_bundle: dict,
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    X = input_df.copy()

    imputation_values = model_bundle["imputation_values"]
    winsorization_bounds = model_bundle["winsorization_bounds"]
    scaler = model_bundle["scaler"]

    for feature, value in imputation_values.items():
        X[feature] = X[feature].fillna(value)

    for feature, bounds in winsorization_bounds.items():
        X[feature] = X[feature].clip(
            lower=bounds["lower"],
            upper=bounds["upper"],
        )

    scaled_values = scaler.transform(X)

    return pd.DataFrame(
        scaled_values,
        columns=X.columns,
        index=X.index,
    )


def prepare_input_for_model(model_object, applicant_data: dict) -> pd.DataFrame:
    input_df = create_input_dataframe(applicant_data)

    if is_model_bundle(model_object):
        return apply_bundle_preprocessing(model_object, input_df)

    preprocessor = get_preprocessor(model_object)

    if preprocessor is not None:
        transformed_values = preprocessor.transform(input_df)

        return pd.DataFrame(
            transformed_values,
            columns=FEATURES,
        )

    return input_df


def predict_default_probability(model, applicant_data: dict) -> float:
    X = prepare_input_for_model(model, applicant_data)
    estimator = get_final_estimator(model)

    probability = estimator.predict_proba(X)[0][1]

    return float(probability)


def compute_linear_score(model, applicant_data: dict) -> float:
    X = prepare_input_for_model(model, applicant_data)
    estimator = get_final_estimator(model)

    z = estimator.decision_function(X)[0]

    return float(z)


def get_model_coefficients(model) -> tuple[float, dict]:
    estimator = get_final_estimator(model)

    coefficients = estimator.coef_[0]
    intercept = estimator.intercept_[0]

    coefficient_dict = {
        feature: float(coefficient)
        for feature, coefficient in zip(FEATURES, coefficients)
    }

    return float(intercept), coefficient_dict


def compute_feature_contributions(model, applicant_data: dict) -> dict:
    X = prepare_input_for_model(model, applicant_data)
    estimator = get_final_estimator(model)

    transformed_values = X.iloc[0].values
    coefficients = estimator.coef_[0]

    contributions = {}

    for feature, value, coefficient in zip(
        FEATURES,
        transformed_values,
        coefficients,
    ):
        contributions[feature] = float(value * coefficient)

    return contributions


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))
