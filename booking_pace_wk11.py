
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "final" / "hotel_dss.db"

print("=== BUILDING BOOKING PACE DATASET ===")

with sqlite3.connect(DB) as con:
    bookings = pd.read_sql_query(
        """
        SELECT
            booking_id,
            hotel_id,
            booking_date_id,
            arrival_date_id,
            status_date_id,
            is_canceled,
            total_nights
        FROM fact_booking
        """,
        con
    )

    snapshots = pd.read_sql_query(
        """
        SELECT
            booking_id,
            snapshot_date_id,
            days_before_arrival,
            is_on_books
        FROM fact_reservation_snapshot
        """,
        con
    )

# Validate join coverage
merged = snapshots.merge(
    bookings,
    on="booking_id",
    how="left",
    validate="many_to_one"
)

assert merged["hotel_id"].notna().all(), (
    "Some snapshots have no matching booking."
)

# Aggregate active reservations by arrival date
pace = (
    merged[merged["is_on_books"] == 1]
    .groupby(
        [
            "hotel_id",
            "arrival_date_id",
            "days_before_arrival"
        ]
    )
    .agg(
        on_books=("booking_id", "nunique")
    )
    .reset_index()
)

# Convert to one row per hotel-arrival date
pace_wide = pace.pivot_table(
    index=["hotel_id", "arrival_date_id"],
    columns="days_before_arrival",
    values="on_books",
    aggfunc="first",
    fill_value=0
)

pace_wide = pace_wide.rename(
    columns=lambda x: f"on_books_{x}d"
).reset_index()

print("\nBooking pace dataset shape:")
print(pace_wide.shape)

print("\nSample records:")
print(pace_wide.head(10).to_string(index=False))

# Export for further modeling
OUTPUT = ROOT / "booking_pace_wk11.csv"
pace_wide.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)


# ============================================
# WEEK 11 — FUTURE NEW BOOKINGS TARGET
# ============================================

print("\n=== FUTURE NEW BOOKINGS TARGET ===")

# Convert date fields
bookings["booking_date"] = pd.to_datetime(
    bookings["booking_date_id"].astype(str),
    format="%Y%m%d"
)

bookings["arrival_date"] = pd.to_datetime(
    bookings["arrival_date_id"].astype(str),
    format="%Y%m%d"
)

# Forecast date: 30 days before arrival
bookings["forecast_date"] = (
    bookings["arrival_date"]
    - pd.Timedelta(days=30)
)

# New bookings created after the forecast date
# and on or before the arrival date.
future_mask = (
    (bookings["booking_date"] > bookings["forecast_date"])
    & (bookings["booking_date"] <= bookings["arrival_date"])
)

future_bookings = (
    bookings.loc[future_mask]
    .groupby(["hotel_id", "arrival_date_id"])
    .agg(
        future_new_bookings=("booking_id", "nunique")
    )
    .reset_index()
)

# Merge with booking pace information
forecast_data = pace_wide.merge(
    future_bookings,
    on=["hotel_id", "arrival_date_id"],
    how="left"
)

forecast_data["future_new_bookings"] = (
    forecast_data["future_new_bookings"]
    .fillna(0)
    .astype(int)
)

print("\nFuture booking target summary:")
print(
    forecast_data["future_new_bookings"]
    .describe()
)

print("\nSample forecast records:")
print(
    forecast_data[
        [
            "hotel_id",
            "arrival_date_id",
            "on_books_30d",
            "future_new_bookings"
        ]
    ].head(10).to_string(index=False)
)

OUTPUT_FORECAST = ROOT / "future_booking_forecast_wk11.csv"

forecast_data.to_csv(
    OUTPUT_FORECAST,
    index=False
)

print("\nSaved:", OUTPUT_FORECAST)
