from config.argument_rules import ARGUMENT_RULES
from src.modeling import get_model_coefficients


def get_validation_metrics(model_object) -> dict | None:
    """
    Returns validation metrics if the saved model is a model bundle.
    """

    if isinstance(model_object, dict):
        return model_object.get("validation_metrics")

    return None


def get_preprocessing_description(model_object) -> dict | None:
    """
    Returns preprocessing details if the saved model is a model bundle.
    """

    if isinstance(model_object, dict):
        return model_object.get("preprocessing_description")

    return None


def get_preprocessing_summary(model_object) -> dict | None:
    """
    Returns the full preprocessing summary if the saved model is a model bundle.
    """

    if isinstance(model_object, dict):
        return model_object.get("preprocessing_summary")

    return None


def compute_normalized_coefficient_strengths(
    model_object,
    explanation_features: list[str] | None = None,
) -> dict:
    """
    Computes normalized coefficient magnitudes:

        normalized_strength_j = |beta_j| / max(|beta|)

    This is the thesis logic used to justify base argument strengths.
    """

    if explanation_features is None:
        explanation_features = list(ARGUMENT_RULES.keys())

    intercept, coefficients = get_model_coefficients(model_object)

    selected_coefficients = {
        feature: coefficients[feature]
        for feature in explanation_features
        if feature in coefficients
    }

    max_abs_coefficient = max(
        abs(coefficient) for coefficient in selected_coefficients.values()
    )

    normalized_strengths = {}

    for feature, coefficient in selected_coefficients.items():
        normalized_strengths[feature] = {
            "coefficient": float(coefficient),
            "absolute_coefficient": float(abs(coefficient)),
            "normalized_strength": float(abs(coefficient) / max_abs_coefficient),
        }

    return {
        "intercept": float(intercept),
        "max_absolute_coefficient": float(max_abs_coefficient),
        "normalized_strengths": normalized_strengths,
    }


def compare_model_strengths_with_rule_strengths(model_object) -> list[dict]:
    """
    Compares the model-derived normalized coefficient strengths with
    the configured base strengths in config/argument_rules.py.
    """

    coefficient_report = compute_normalized_coefficient_strengths(model_object)

    comparisons = []

    for feature, rule in ARGUMENT_RULES.items():
        model_strength = coefficient_report["normalized_strengths"][feature][
            "normalized_strength"
        ]

        configured_strength = rule["base_strength"]

        comparisons.append(
            {
                "feature": feature,
                "model_normalized_strength": model_strength,
                "configured_base_strength": configured_strength,
                "difference": configured_strength - model_strength,
            }
        )

    return comparisons


def create_model_diagnostics_report(model_object) -> dict:
    """
    Creates a compact model diagnostics report for auditability.
    """

    return {
        "coefficient_strengths": compute_normalized_coefficient_strengths(model_object),
        "base_strength_comparison": compare_model_strengths_with_rule_strengths(
            model_object
        ),
        "validation_metrics": get_validation_metrics(model_object),
        "preprocessing_description": get_preprocessing_description(model_object),
        "preprocessing_summary": get_preprocessing_summary(model_object),
    }
