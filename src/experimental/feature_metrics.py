import numpy as np


def safe_float(value):
    if value is None or np.isnan(value):
        return None
    return float(value)


def compute_feature_metrics(df, target_column):
    """
    Compute dataset-level mathematical diagnostics
    for every numerical feature.

    These metrics support rule suggestion, auditability,
    and dataset-specific interpretation.
    """

    metrics = {}

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    total_rows = len(df)

    for column in numerical_columns:
        series = df[column].dropna()

        q01 = series.quantile(0.01)
        q25 = series.quantile(0.25)
        q50 = series.quantile(0.50)
        q75 = series.quantile(0.75)
        q95 = series.quantile(0.95)
        q99 = series.quantile(0.99)

        iqr = q75 - q25
        lower_outlier_bound = q25 - 1.5 * iqr
        upper_outlier_bound = q75 + 1.5 * iqr

        correlation = df[column].corr(df[target_column])

        if correlation > 0:
            empirical_risk_direction = "higher values associated with higher risk"
        elif correlation < 0:
            empirical_risk_direction = "higher values associated with lower risk"
        else:
            empirical_risk_direction = "no clear linear association"

        missing_values = df[column].isna().sum()

        metrics[column] = {
            "missing_values": int(missing_values),
            "missing_rate": safe_float(missing_values / total_rows),
            "mean": safe_float(series.mean()),
            "median": safe_float(series.median()),
            "std": safe_float(series.std()),
            "min": safe_float(series.min()),
            "max": safe_float(series.max()),
            "q01": safe_float(q01),
            "q25": safe_float(q25),
            "q50": safe_float(q50),
            "q75": safe_float(q75),
            "q95": safe_float(q95),
            "q99": safe_float(q99),
            "iqr": safe_float(iqr),
            "lower_outlier_bound": safe_float(lower_outlier_bound),
            "upper_outlier_bound": safe_float(upper_outlier_bound),
            "empirical_range": safe_float(series.max() - series.min()),
            "correlation_with_target": safe_float(correlation),
            "absolute_correlation": safe_float(abs(correlation)),
            "empirical_risk_direction": empirical_risk_direction,
        }

    return metrics
