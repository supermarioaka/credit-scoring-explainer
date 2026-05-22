import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


DATA_PATH = "data/cs-training.csv"
MODEL_PATH = "models/credit_model.pkl"


df = pd.read_csv(DATA_PATH)

# Remove first unnamed index column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

target = "SeriousDlqin2yrs"

features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTimes90DaysLate",
]

df = df[features + [target]]

# Simple missing value handling
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("Model trained successfully.")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC: {auc:.4f}")

joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
