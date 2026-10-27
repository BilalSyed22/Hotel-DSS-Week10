
import sqlite3
from pathlib import Path

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data/final/hotel_dss.db"

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query(
        "SELECT * FROM fact_booking",
        conn
    )

# Features available when a reservation is created.
numeric_features = [
    "lead_time",
    "adr",
    "adults",
    "children",
    "previous_cancellations",
    "is_repeated_guest",
    "total_special_requests",
    "total_nights"
]

categorical_features = [
    "hotel_id",
    "segment_code",
    "channel_code",
    "deposit_type_code"
]

features = numeric_features + categorical_features

X = df[features]
y = df["is_canceled"]

# Chronological split using the arrival date.
df = df.sort_values("arrival_date_id").reset_index(drop=True)
X = df[features]
y = df["is_canceled"]

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print("Brier Score:", round(brier_score_loss(y_test, probabilities), 4))

joblib.dump(
    {"model": model, "features": features},
    ROOT / "cancellation_model.pkl"
)

print("Model saved successfully!")
