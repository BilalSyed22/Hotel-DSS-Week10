
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "final" / "hotel_dss.db"

with sqlite3.connect(DB) as con:
    bookings = pd.read_sql_query(
        """
        SELECT
            booking_id,
            hotel_id,
            booking_date_id,
            arrival_date_id,
            total_nights,
            is_canceled,
            status_date_id
        FROM fact_booking
        """,
        con
    )

    daily = pd.read_sql_query(
        """
        SELECT
            hotel_id,
            stay_date_id,
            capacity_rooms
        FROM fact_daily_stay
        """,
        con
    )

# Convert date columns
bookings["booking_date"] = pd.to_datetime(
    bookings["booking_date_id"].astype(str),
    format="%Y%m%d"
)

bookings["arrival_date"] = pd.to_datetime(
    bookings["arrival_date_id"].astype(str),
    format="%Y%m%d"
)

bookings["status_date"] = pd.to_datetime(
    bookings["status_date_id"].astype(str),
    format="%Y%m%d"
)

# Expand each booking over its stay nights
expanded = []

for row in bookings.itertuples(index=False):
    for offset in range(int(row.total_nights)):
        stay_date = row.arrival_date + pd.Timedelta(days=offset)

        expanded.append({
            "booking_id": row.booking_id,
            "hotel_id": row.hotel_id,
            "stay_date": stay_date,
            "booking_date": row.booking_date,
            "status_date": row.status_date,
            "is_canceled": row.is_canceled,
            "arrival_date": row.arrival_date,
        })

nightly = pd.DataFrame(expanded)

# Define forecasting cutoff for each stay night
nightly["forecast_date"] = (
    nightly["stay_date"] - pd.Timedelta(days=30)
)

# Reservations active on the forecast date
active_at_forecast = (
    (nightly["booking_date"] <= nightly["forecast_date"])
    &
    (
        (nightly["is_canceled"] == 0)
        | (nightly["status_date"] > nightly["forecast_date"])
    )
)


# Count bookings created after the 30-day forecast
# cutoff but no later than the guest's arrival date.

new_after_forecast = (
    (nightly["booking_date"] > nightly["forecast_date"])
    & (nightly["booking_date"] <= nightly["arrival_date"])
)


nightly["on_books_30d"] = active_at_forecast.astype(int)
nightly["future_new_30d"] = new_after_forecast.astype(int)

summary = nightly.groupby(
    ["hotel_id", "stay_date"]
).agg(
    on_books_30d=("on_books_30d", "sum"),
    future_new_30d=("future_new_30d", "sum"),
).reset_index()

daily["stay_date"] = pd.to_datetime(
    daily["stay_date_id"].astype(str),
    format="%Y%m%d"
)

summary = daily[
    ["hotel_id", "stay_date", "capacity_rooms"]
].merge(
    summary,
    on=["hotel_id", "stay_date"],
    how="left"
)

summary[["on_books_30d", "future_new_30d"]] = (
    summary[["on_books_30d", "future_new_30d"]]
    .fillna(0)
    .astype(int)
)

summary["month"] = summary["stay_date"].dt.month
summary["day_of_week"] = summary["stay_date"].dt.dayofweek
summary["is_weekend"] = (
    summary["day_of_week"] >= 5
).astype(int)

print("\n=== NIGHTLY FUTURE DEMAND DATASET ===")
print("Rows:", len(summary))
print("Date range:",
      summary["stay_date"].min(),
      "to",
      summary["stay_date"].max())

print("\nFuture demand summary:")
print(summary["future_new_30d"].describe())

print("\nSample records:")
print(summary.head(10).to_string(index=False))


print("\n=== NIGHTLY DEMAND VALIDATION ===")

# Every new booking must have a valid booking timeline.
invalid_new_bookings = (
    (nightly["future_new_30d"] == 1)
    & (
        (nightly["booking_date"] > nightly["arrival_date"])
        | (nightly["booking_date"] <= nightly["forecast_date"])
    )
).sum()

print("Invalid future bookings:", int(invalid_new_bookings))

# Check the same booking-night does not appear twice.
duplicate_booking_nights = nightly.duplicated(
    ["booking_id", "stay_date"]
).sum()

print("Duplicate booking-night rows:",
      int(duplicate_booking_nights))

print("Number of hotel-night records:", len(summary))

print("Maximum future demand:",
      int(summary["future_new_30d"].max()))

print("\nNightly demand validation completed.")


output = ROOT / "nightly_future_demand_wk11.csv"
summary.to_csv(output, index=False)

print("\nSaved:", output)


print("\n=== DEMAND DATA BOUNDARY CHECK ===")

print(
    "Latest booking arrival:",
    bookings["arrival_date"].max().date()
)

print(
    "Latest booking creation:",
    bookings["booking_date"].max().date()
)

print(
    "Latest stay date:",
    summary["stay_date"].max().date()
)

latest_arrival = bookings["arrival_date"].max()

boundary = summary[
    summary["stay_date"] > latest_arrival
]

print(
    "Hotel-night rows after last arrival:",
    len(boundary)
)

print("\nAverage future demand by stay month:")
print(
    summary.assign(
        stay_month=summary["stay_date"].dt.to_period("M")
    )
    .groupby("stay_month")["future_new_30d"]
    .agg(["count", "mean", "min", "max"])
    .tail(6)
    .round(2)
)
