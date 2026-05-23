import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTimes90DaysLate",
]


def remove_unnamed_index_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def select_thesis_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df[required_columns].copy()


def split_features_and_target(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def compute_imputation_values(X_train: pd.DataFrame) -> dict:
    return X_train.mean(numeric_only=True).to_dict()


def apply_imputation(
    X: pd.DataFrame,
    imputation_values: dict,
) -> pd.DataFrame:
    X = X.copy()

    for feature, value in imputation_values.items():
        X[feature] = X[feature].fillna(value)

    return X


def compute_winsorization_bounds(
    X_train: pd.DataFrame,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict:
    bounds = {}

    for feature in X_train.columns:
        bounds[feature] = {
            "lower": X_train[feature].quantile(lower_quantile),
            "upper": X_train[feature].quantile(upper_quantile),
        }

    return bounds


def apply_winsorization(
    X: pd.DataFrame,
    bounds: dict,
) -> pd.DataFrame:
    X = X.copy()

    for feature, feature_bounds in bounds.items():
        X[feature] = X[feature].clip(
            lower=feature_bounds["lower"],
            upper=feature_bounds["upper"],
        )

    return X


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)

    return scaler


def apply_scaling(
    X: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    scaled_values = scaler.transform(X)

    return pd.DataFrame(
        scaled_values,
        columns=X.columns,
        index=X.index,
    )


def prepare_training_data(df: pd.DataFrame) -> dict:
    df = remove_unnamed_index_column(df)
    df = select_thesis_columns(df)

    X, y = split_features_and_target(df)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    imputation_values = compute_imputation_values(X_train)
    X_train = apply_imputation(X_train, imputation_values)
    X_test = apply_imputation(X_test, imputation_values)

    winsorization_bounds = compute_winsorization_bounds(X_train)
    X_train = apply_winsorization(X_train, winsorization_bounds)
    X_test = apply_winsorization(X_test, winsorization_bounds)

    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaling(X_train, scaler)
    X_test_scaled = apply_scaling(X_test, scaler)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "imputation_values": imputation_values,
        "winsorization_bounds": winsorization_bounds,
        "scaler": scaler,
        "features": FEATURES,
        "target": TARGET,
    }
