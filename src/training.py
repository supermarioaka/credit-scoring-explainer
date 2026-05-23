import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.preprocessing import prepare_training_data
from src.validation import evaluate_classification_model, print_validation_report


DATA_PATH = "data/cs-training.csv"
MODEL_PATH = "models/credit_model.pkl"


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def build_model_bundle(
    model,
    prepared_data: dict,
    validation_metrics: dict,
) -> dict:
    return {
        "model_type": "Logistic Regression",
        "model": model,
        "features": prepared_data["features"],
        "target": prepared_data["target"],
        "imputation_values": prepared_data["imputation_values"],
        "winsorization_bounds": prepared_data["winsorization_bounds"],
        "scaler": prepared_data["scaler"],
        "validation_metrics": validation_metrics,
        "preprocessing_description": {
            "imputation": "Missing numeric values are filled using training-set means.",
            "winsorization": "Features are clipped using training-set 1st and 99th percentiles.",
            "scaling": "Features are standardized using a StandardScaler fitted on the training set.",
            "split": "80/20 train-test split with stratification on the target.",
        },
    }


def train_and_save_model(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
) -> dict:
    df = pd.read_csv(data_path)

    prepared_data = prepare_training_data(df)

    model = train_logistic_regression(
        X_train=prepared_data["X_train"],
        y_train=prepared_data["y_train"],
    )

    validation_metrics = evaluate_classification_model(
        model=model,
        X_test=prepared_data["X_test"],
        y_test=prepared_data["y_test"],
    )

    model_bundle = build_model_bundle(
        model=model,
        prepared_data=prepared_data,
        validation_metrics=validation_metrics,
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model_bundle, model_path)

    print("Model trained and saved successfully.")
    print(f"Model path: {model_path}")
    print_validation_report(validation_metrics)

    return model_bundle
