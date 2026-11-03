
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(
    ROOT / "nightly_future_demand_wk11.csv"
)

df["stay_date"] = pd.to_datetime(df["stay_date"])

features = [
    "hotel_id",
    "capacity_rooms",
    "on_books_30d",
    "month",
    "day_of_week",
    "is_weekend",
]

target = "future_new_30d"

# Chronological split by distinct stay dates.
dates = np.sort(df["stay_date"].unique())

train_end = pd.Timestamp(dates[int(len(dates) * 0.70)])
valid_end = pd.Timestamp(dates[int(len(dates) * 0.85)])

# 30-day embargo:
# Training outcomes must be known before
# the earliest validation forecast date.
train = df[
    df["stay_date"] <
    train_end - pd.Timedelta(days=30)
].copy()

valid = df[
    (df["stay_date"] >= train_end)
    & (df["stay_date"] < valid_end)
].copy()

test = df[
    df["stay_date"] >= valid_end
].copy()

print("=== NIGHTLY DEMAND DATA SPLIT ===")

for name, part in [
    ("Training", train),
    ("Validation", valid),
    ("Test", test),
]:
    print(
        name,
        len(part),
        part["stay_date"].min().date(),
        "to",
        part["stay_date"].max().date()
    )

assert not train.empty and not valid.empty

# Baseline: average by hotel, calculated
# from the training set only.
hotel_means = train.groupby("hotel_id")[target].mean()
overall_mean = train[target].mean()

baseline_valid = (
    valid["hotel_id"]
    .map(hotel_means)
    .fillna(overall_mean)
    .to_numpy()
)

# Candidate model
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.05,
    min_samples_leaf=10,
    random_state=42,
)

model.fit(train[features], train[target])

pred_valid = np.maximum(
    model.predict(valid[features]),
    0
)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
    }

results = pd.DataFrame([
    {
        "Model": "Historical Hotel Average",
        **evaluate(valid[target], baseline_valid),
    },
    {
        "Model": "Gradient Boosting",
        **evaluate(valid[target], pred_valid),
    },
])

print("\n=== NIGHTLY DEMAND VALIDATION ===")
print(results.round(4).to_string(index=False))

# Inspect predictions for possible systematic issues.
comparison = valid[
    [
        "hotel_id",
        "stay_date",
        "capacity_rooms",
        "on_books_30d",
        target,
    ]
].copy()

comparison["predicted_new_bookings"] = pred_valid
comparison["absolute_error"] = np.abs(
    comparison[target] - pred_valid
)

print("\n=== SAMPLE PREDICTIONS ===")
print(
    comparison.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n=== ERROR BY HOTEL ===")
print(
    comparison.groupby("hotel_id")["absolute_error"]
    .agg(["count", "mean", "max"])
    .round(3)
)

print("\nFinal test set remains untouched.")


# ============================================
# WEEK 11 — DEMAND FORECAST ERROR ANALYSIS
# ============================================

print("\n=== DEMAND FORECAST ERROR ANALYSIS ===")

comparison["signed_error"] = (
    comparison["predicted_new_bookings"]
    - comparison["future_new_30d"]
)

print("\nOverall forecast bias:")
print(
    "Mean signed error:",
    round(comparison["signed_error"].mean(), 3)
)

print(
    "Overprediction rate:",
    round(
        (comparison["signed_error"] > 0).mean() * 100,
        2
    ),
    "%"
)

# Analyze error by realized demand level
comparison["demand_group"] = pd.cut(
    comparison["future_new_30d"],
    bins=[-1, 20, 35, np.inf],
    labels=[
        "Low (0-20)",
        "Medium (21-35)",
        "High (36+)"
    ]
)

print("\n=== ERROR BY DEMAND LEVEL ===")

print(
    comparison.groupby(
        "demand_group",
        observed=True
    ).agg(
        count=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        bias=("signed_error", "mean")
    ).round(3)
)

# Empirical error quantiles, for diagnostics.
print("\n=== ABSOLUTE ERROR QUANTILES ===")
print(
    comparison["absolute_error"]
    .quantile([0.50, 0.80, 0.90, 0.95])
    .round(3)
)

print("\nError analysis completed.")


# ============================================
# WEEK 11 — NIGHTLY DEMAND FINAL TEST
# ============================================

print("\n=== NIGHTLY DEMAND FINAL TEST ===")

# Exclude stay dates after the last booking
# arrival date in the synthetic dataset.
test_final = test[
    test["stay_date"] <= pd.Timestamp("2017-08-31")
].copy()

assert not test_final.empty, "Final test set is empty."

print("Final test rows:", len(test_final))
print(
    "Final test period:",
    test_final["stay_date"].min().date(),
    "to",
    test_final["stay_date"].max().date()
)

# Historical baseline: training data only.
baseline_test = (
    test_final["hotel_id"]
    .map(hotel_means)
    .fillna(overall_mean)
    .to_numpy()
)

# Evaluate the already-trained model.
# No retraining or hyperparameter tuning.
pred_test = np.maximum(
    model.predict(test_final[features]),
    0
)

test_results = pd.DataFrame([
    {
        "Model": "Historical Hotel Average",
        **evaluate(
            test_final[target],
            baseline_test
        ),
    },
    {
        "Model": "Gradient Boosting",
        **evaluate(
            test_final[target],
            pred_test
        ),
    },
])

print("\n=== FINAL TEST METRICS ===")
print(
    test_results.round(4).to_string(index=False)
)

# Measure forecast bias on the held-out test.
signed_error = (
    pred_test - test_final[target].to_numpy()
)

print("\n=== FINAL TEST BIAS ===")
print(
    "Mean signed error:",
    round(float(signed_error.mean()), 3)
)
print(
    "Overprediction rate:",
    round(float((signed_error > 0).mean() * 100), 2),
    "%"
)

# Monthly diagnostic: identify distribution shifts.
test_diagnostics = test_final[
    ["hotel_id", "stay_date", target]
].copy()

test_diagnostics["predicted"] = pred_test
test_diagnostics["absolute_error"] = np.abs(
    signed_error
)
test_diagnostics["month"] = (
    test_diagnostics["stay_date"]
    .dt.to_period("M")
)

print("\n=== FINAL TEST ERROR BY MONTH ===")
print(
    test_diagnostics.groupby("month").agg(
        count=("absolute_error", "size"),
        actual_mean=(target, "mean"),
        predicted_mean=("predicted", "mean"),
        MAE=("absolute_error", "mean")
    ).round(3)
)

print("\nFinal test evaluation completed.")


import joblib

model_bundle = {
    "model": model,
    "features": features,
    "forecast_horizon_days": 30,
    "validation_mae": 4.6743,
    "test_mae": 5.2187,
}

model_path = ROOT / "nightly_demand_model_wk11.pkl"

joblib.dump(model_bundle, model_path)

print("\nSaved demand model:", model_path)


loaded_bundle = joblib.load(model_path)

print("\n=== SAVED MODEL CHECK ===")
print("Features:", loaded_bundle["features"])
print(
    "Forecast horizon:",
    loaded_bundle["forecast_horizon_days"],
    "days"
)
print("Model loaded successfully.")

