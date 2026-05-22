import numpy as np


def compute_feature_metrics(df, target_column):
    """
    Compute statistical and risk-related metrics
    for every numerical feature in the dataset.
    """

    metrics = {}

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    for column in numerical_columns:
        series = df[column].dropna()

        metrics[column] = {
            "missing_values": int(df[column].isna().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q01": float(series.quantile(0.01)),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q95": float(series.quantile(0.95)),
            "q99": float(series.quantile(0.99)),
            "correlation_with_target": float(df[column].corr(df[target_column])),
        }

    return metrics
