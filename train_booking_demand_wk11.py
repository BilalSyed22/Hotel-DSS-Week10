
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
    ROOT / "future_booking_forecast_wk11.csv"
)

# Date features
df["arrival_date"] = pd.to_datetime(
    df["arrival_date_id"].astype(str),
    format="%Y%m%d"
)

df["month"] = df["arrival_date"].dt.month
df["day_of_week"] = df["arrival_date"].dt.dayofweek
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Only features available 30 days before arrival
features = [
    "hotel_id",
    "on_books_30d",
    "month",
    "day_of_week",
    "is_weekend",
]

target = "future_new_bookings"

# Chronological split by unique arrival dates
dates = np.sort(df["arrival_date"].unique())

train_end = dates[int(len(dates) * 0.70)]
valid_end = dates[int(len(dates) * 0.85)]


# Use a 30-day embargo to avoid training
# on outcomes unavailable at validation forecast time.

forecast_horizon_days = 30

train = df[
    df["arrival_date"]
    < (
        pd.Timestamp(train_end)
        - pd.Timedelta(days=forecast_horizon_days)
    )
].copy()

valid = df[
    (df["arrival_date"] >= train_end)
    & (df["arrival_date"] < valid_end)
].copy()


valid = df[
    (df["arrival_date"] >= train_end)
    & (df["arrival_date"] < valid_end)
].copy()

test = df[df["arrival_date"] >= valid_end].copy()

print("=== FUTURE DEMAND DATA SPLIT ===")

for name, part in [
    ("Training", train),
    ("Validation", valid),
    ("Test", test),
]:
    print(
        name,
        len(part),
        part["arrival_date"].min().date(),
        "to",
        part["arrival_date"].max().date()
    )

# Baseline: historical average new bookings by hotel
hotel_means = train.groupby("hotel_id")[target].mean()

baseline_valid = (
    valid["hotel_id"]
    .map(hotel_means)
    .fillna(train[target].mean())
)

# Gradient Boosting Model
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

def metrics(y, predicted):
    return {
        "MAE": mean_absolute_error(y, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(y, predicted)
        ),
        "R2": r2_score(y, predicted),
    }

results = pd.DataFrame([
    {
        "Model": "Historical Hotel Average",
        **metrics(valid[target], baseline_valid)
    },
    {
        "Model": "Gradient Boosting",
        **metrics(valid[target], pred_valid)
    },
])

print("\n=== FUTURE DEMAND VALIDATION ===")
print(results.round(4).to_string(index=False))

print(
    "\nFinal test set has not been evaluated."
)
