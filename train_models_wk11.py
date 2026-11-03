
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "final" / "hotel_dss.db"
SEED = 42

# 1. Load data
with sqlite3.connect(DB) as con:
    bookings = pd.read_sql_query(
        "SELECT * FROM fact_booking", con
    )
    daily = pd.read_sql_query(
        "SELECT * FROM fact_daily_stay", con
    )
    dates = pd.read_sql_query(
        "SELECT * FROM dim_date", con
    )

# 2. Split by unique dates, not row positions
def temporal_split(df, date_col):
    unique_dates = np.sort(df[date_col].unique())
    n = len(unique_dates)

    train_end = unique_dates[int(n * 0.70)]
    valid_end = unique_dates[int(n * 0.85)]

    train = df[df[date_col] < train_end].copy()
    valid = df[
        (df[date_col] >= train_end)
        & (df[date_col] < valid_end)
    ].copy()
    test = df[df[date_col] >= valid_end].copy()

    for name, part in [
        ("Train", train),
        ("Validation", valid),
        ("Test", test),
    ]:
        print(
            name, "rows:", len(part),
            "dates:", part[date_col].min(),
            "to", part[date_col].max()
        )

    return train, valid, test


def make_pipeline(numerical, categorical, estimator):
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])


# KPI 1 — Booking Cancellation
print("\n=== KPI 1: Cancellation ===")

num1 = [
    "lead_time", "adr", "adults", "children",
    "previous_cancellations", "is_repeated_guest",
    "total_special_requests", "total_nights",
]
cat1 = [
    "hotel_id", "segment_code",
    "channel_code", "deposit_type_code",
]
feat1 = num1 + cat1

tr1, va1, te1 = temporal_split(
    bookings, "arrival_date_id"
)


# Exclude training outcomes unavailable
# when validation begins.

validation_start_id = va1["arrival_date_id"].min()

original_train_count = len(tr1)

tr1 = tr1[
    tr1["status_date_id"] < validation_start_id
].copy()

print("\n=== LEAKAGE-AWARE TRAINING ===")
print("Original training rows:", original_train_count)
print("Filtered training rows:", len(tr1))
print(
    "Excluded rows:",
    original_train_count - len(tr1)
)


clf = make_pipeline(
    num1, cat1,
    GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=SEED,
    ),
)

clf.fit(tr1[feat1], tr1["is_canceled"])

# Baseline: historical cancellation rate from training only
base1 = np.full(
    len(va1), tr1["is_canceled"].mean()
)

pred1 = clf.predict_proba(va1[feat1])[:, 1]
actual1 = va1["is_canceled"]

print("\nCancellation — VALIDATION results")
print(
    "Baseline Brier:",
    round(brier_score_loss(actual1, base1), 4)
)
print(
    "Model Brier:",
    round(brier_score_loss(actual1, pred1), 4)
)
print(
    "Baseline AUC:", 0.5
)
print(
    "Model AUC:",
    round(roc_auc_score(actual1, pred1), 4)
)


# KPI 2 — Nightly Occupancy
print("\n=== KPI 2: Occupancy ===")

occ = daily.merge(
    dates,
    left_on="stay_date_id",
    right_on="date_id",
    how="left",
)
occ["occupancy_rate"] = (
    occ["occupied_rooms"] / occ["capacity_rooms"]
)


# ============================================
# WEEK 11 — OCCUPANCY TARGET CORRECTION
# ============================================

# The synthetic daily table counts room demand,
# which can exceed physical hotel capacity.
# Actual occupancy cannot exceed capacity.

occ["room_demand"] = occ["occupied_rooms"].copy()

occ["walked_rooms_estimate"] = (
    occ["room_demand"] - occ["capacity_rooms"]
).clip(lower=0)

occ["actual_occupied_rooms"] = np.minimum(
    occ["room_demand"],
    occ["capacity_rooms"]
)

occ["occupancy_rate"] = (
    occ["actual_occupied_rooms"]
    / occ["capacity_rooms"]
)

print("\n=== OCCUPANCY TARGET CORRECTION ===")
print(
    "Nights with excess demand:",
    (occ["room_demand"] > occ["capacity_rooms"]).sum()
)
print(
    "Maximum corrected occupancy:",
    round(occ["occupancy_rate"].max() * 100, 2),
    "%"
)
print(
    "Estimated walked rooms:",
    int(occ["walked_rooms_estimate"].sum())
)


num2 = [
    "month", "week_number", "day_of_month",
    "is_weekend", "year",
]
cat2 = ["hotel_id", "season"]
feat2 = num2 + cat2

tr2, va2, te2 = temporal_split(
    occ, "stay_date_id"
)

reg = make_pipeline(
    num2, cat2,
    GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=SEED,
    ),
)

reg.fit(tr2[feat2], tr2["occupancy_rate"])

# Baseline: training mean occupancy for each hotel
hotel_means = tr2.groupby(
    "hotel_id"
)["occupancy_rate"].mean()

base2 = va2["hotel_id"].map(hotel_means)
base2 = base2.fillna(tr2["occupancy_rate"].mean())

pred2 = reg.predict(va2[feat2])
actual2 = va2["occupancy_rate"]

print("\nOccupancy — VALIDATION results")
print(
    "Baseline MAE (pp):",
    round(mean_absolute_error(actual2, base2) * 100, 2)
)
print(
    "Model MAE (pp):",
    round(mean_absolute_error(actual2, pred2) * 100, 2)
)
print(
    "Baseline RMSE (pp):",
    round(np.sqrt(mean_squared_error(actual2, base2)) * 100, 2)
)
print(
    "Model RMSE (pp):",
    round(np.sqrt(mean_squared_error(actual2, pred2)) * 100, 2)
)
print(
    "Baseline R2:",
    round(r2_score(actual2, base2), 4)
)
print(
    "Model R2:",
    round(r2_score(actual2, pred2), 4)
)


# ============================================
# PART 2 — Model Improvement Experiments
# ============================================

print("\n=== MODEL IMPROVEMENT EXPERIMENTS ===")

# KPI 1: Try a simpler, regularized classifier
clf_revised = make_pipeline(
    num1, cat1,
    GradientBoostingClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=SEED,
    ),
)

clf_revised.fit(tr1[feat1], tr1["is_canceled"])
pred1_revised = clf_revised.predict_proba(
    va1[feat1]
)[:, 1]

cancellation_comparison = pd.DataFrame([
    {
        "Model": "Original",
        "ROC-AUC": roc_auc_score(actual1, pred1),
        "Brier": brier_score_loss(actual1, pred1),
    },
    {
        "Model": "Revised",
        "ROC-AUC": roc_auc_score(
            actual1, pred1_revised
        ),
        "Brier": brier_score_loss(
            actual1, pred1_revised
        ),
    },
])

print("\nKPI 1 — Original vs Revised")
print(cancellation_comparison.round(4).to_string(index=False))


# KPI 2: Try stronger regularization
reg_revised = make_pipeline(
    num2, cat2,
    GradientBoostingRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        min_samples_leaf=10,
        random_state=SEED,
    ),
)

reg_revised.fit(tr2[feat2], tr2["occupancy_rate"])
pred2_revised = reg_revised.predict(va2[feat2])

occupancy_comparison = pd.DataFrame([
    {
        "Model": "Original",
        "MAE_pp": mean_absolute_error(
            actual2, pred2
        ) * 100,
        "RMSE_pp": np.sqrt(
            mean_squared_error(actual2, pred2)
        ) * 100,
        "R2": r2_score(actual2, pred2),
    },
    {
        "Model": "Revised",
        "MAE_pp": mean_absolute_error(
            actual2, pred2_revised
        ) * 100,
        "RMSE_pp": np.sqrt(
            mean_squared_error(actual2, pred2_revised)
        ) * 100,
        "R2": r2_score(actual2, pred2_revised),
    },
])

print("\nKPI 2 — Original vs Revised")
print(occupancy_comparison.round(4).to_string(index=False))


# ============================================
# PART 3 — ERROR ANALYSIS
# ============================================

print("\n=== ERROR ANALYSIS ===")

# KPI 1: Cancellation errors by lead time
cancel_errors = va1[
    ["lead_time", "hotel_id", "is_canceled"]
].copy()

cancel_errors["predicted_probability"] = pred1
cancel_errors["absolute_error"] = abs(
    cancel_errors["is_canceled"]
    - cancel_errors["predicted_probability"]
)

cancel_errors["lead_time_group"] = pd.cut(
    cancel_errors["lead_time"],
    bins=[-1, 7, 30, 90, float("inf")],
    labels=[
        "0-7 days",
        "8-30 days",
        "31-90 days",
        "91+ days",
    ],
)

print("\nKPI 1 — Error by Lead Time")
print(
    cancel_errors.groupby(
        "lead_time_group", observed=True
    )["absolute_error"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nKPI 1 — Error by Hotel")
print(
    cancel_errors.groupby("hotel_id")[
        "absolute_error"
    ].agg(["count", "mean"]).round(4)
)


# KPI 2: Occupancy error by actual occupancy
occupancy_errors = va2[
    ["hotel_id", "occupancy_rate"]
].copy()

occupancy_errors["predicted_occupancy"] = pred2
occupancy_errors["absolute_error_pp"] = (
    abs(
        occupancy_errors["occupancy_rate"]
        - occupancy_errors["predicted_occupancy"]
    ) * 100
)

occupancy_errors["occupancy_group"] = pd.cut(
    occupancy_errors["occupancy_rate"],
    bins=[-float("inf"), 0.5, 0.8, float("inf")],
    labels=[
        "Low (<=50%)",
        "Medium (50-80%)",
        "High (>80%)",
    ],
)

print("\nKPI 2 — Error by Occupancy Level")
print(
    occupancy_errors.groupby(
        "occupancy_group", observed=True
    )["absolute_error_pp"]
    .agg(["count", "mean"])
    .round(2)
)

print("\nKPI 2 — Error by Hotel")
print(
    occupancy_errors.groupby("hotel_id")[
        "absolute_error_pp"
    ].agg(["count", "mean"]).round(2)
)


# ============================================
# PART 4 — DATA LEAKAGE CHECK
# ============================================

print("\n=== DATA LEAKAGE CHECK ===")

# Check date-related booking columns
date_columns = [
    c for c in bookings.columns
    if "date" in c.lower()
    or "status" in c.lower()
    or "cancel" in c.lower()
]

print("\nBooking date/status columns:")
print(date_columns)

print("\nCancellation model predictors:")
print(feat1)

print("\nOccupancy model predictors:")
print(feat2)

# Check for obvious outcome-related predictors
forbidden_cancel = {
    "is_canceled",
    "reservation_status",
    "reservation_status_date",
}

forbidden_occupancy = {
    "occupied_rooms",
    "occupancy_rate",
    "realized_occupancy",
}

print("\nCancellation leakage columns:",
      sorted(set(feat1) & forbidden_cancel))

print("Occupancy leakage columns:",
      sorted(set(feat2) & forbidden_occupancy))

# Check whether date-based splits overlap
for label, train, valid, test, col in [
    ("Cancellation", tr1, va1, te1, "arrival_date_id"),
    ("Occupancy", tr2, va2, te2, "stay_date_id"),
]:
    chronological = (
        train[col].max() < valid[col].min()
        and valid[col].max() < test[col].min()
    )
    print(label, "split dates non-overlapping:",
          chronological)

print("\nLeakage screening completed.")
print("Manual decision-time availability review still required.")


# ============================================
# PART 5 — TEMPORAL AVAILABILITY AUDIT
# ============================================

print("\n=== TEMPORAL AVAILABILITY AUDIT ===")

date_fields = [
    "booking_date_id",
    "arrival_date_id",
    "status_date_id"
]

print("\nExample booking records:")
print(
    bookings[date_fields + ["is_canceled"]]
    .head(10)
    .to_string(index=False)
)

print("\nNull counts:")
print(bookings[date_fields].isna().sum())

# Convert YYYYMMDD integers to timestamps
for col in date_fields:
    bookings[col + "_dt"] = pd.to_datetime(
        bookings[col].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

print("\nInvalid dates:")
print(
    bookings[
        [c + "_dt" for c in date_fields]
    ].isna().sum()
)

# Booking should not occur after arrival
booking_after_arrival = (
    bookings["booking_date_id_dt"]
    > bookings["arrival_date_id_dt"]
).sum()

# Status should not predate booking
status_before_booking = (
    bookings["status_date_id_dt"]
    < bookings["booking_date_id_dt"]
).sum()

print("\nBookings created after arrival:",
      booking_after_arrival)

print("Status dates before booking:",
      status_before_booking)

print("\nBooking dates range:",
      bookings["booking_date_id_dt"].min(),
      "to",
      bookings["booking_date_id_dt"].max())

print("\nStatus dates range:",
      bookings["status_date_id_dt"].min(),
      "to",
      bookings["status_date_id_dt"].max())

print("\nTemporal availability audit completed.")


# ============================================
# PART 6 — LABEL AVAILABILITY CHECK
# ============================================

print("\n=== LABEL AVAILABILITY CHECK ===")

validation_start = pd.to_datetime(
    str(va1["arrival_date_id"].min()),
    format="%Y%m%d"
)

test_start = pd.to_datetime(
    str(te1["arrival_date_id"].min()),
    format="%Y%m%d"
)

print("Validation start:", validation_start.date())
print("Test start:", test_start.date())

# Check whether training outcomes were known
# before the validation period began.
train_status_dates = pd.to_datetime(
    tr1["status_date_id"].astype(str),
    format="%Y%m%d"
)

unavailable_train_labels = (
    train_status_dates >= validation_start
).sum()

print(
    "\nTraining records with outcomes not finalized "
    "before validation:",
    unavailable_train_labels
)

print(
    "Percentage:",
    round(
        unavailable_train_labels / len(tr1) * 100, 2
    ),
    "%"
)

# Check validation outcomes before final test period
valid_status_dates = pd.to_datetime(
    va1["status_date_id"].astype(str),
    format="%Y%m%d"
)

unavailable_valid_labels = (
    valid_status_dates >= test_start
).sum()

print(
    "\nValidation records with outcomes not finalized "
    "before test:",
    unavailable_valid_labels
)

print(
    "Percentage:",
    round(
        unavailable_valid_labels / len(va1) * 100, 2
    ),
    "%"
)

print("\nLabel availability check completed.")


# ============================================
# PART 7 — FEATURE ABLATION TEST
# ============================================

print("\n=== FEATURE ABLATION TEST ===")

# Remove potentially time-sensitive feature
num1_safe = [
    col for col in num1
    if col != "previous_cancellations"
]

feat1_safe = num1_safe + cat1

clf_without_history = make_pipeline(
    num1_safe,
    cat1,
    GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=SEED,
    ),
)

clf_without_history.fit(
    tr1[feat1_safe],
    tr1["is_canceled"]
)

pred_without_history = (
    clf_without_history.predict_proba(
        va1[feat1_safe]
    )[:, 1]
)

print("\nKPI 1 — Feature Ablation Comparison")

results = pd.DataFrame([
    {
        "Model": "Original",
        "ROC-AUC": roc_auc_score(
            actual1, pred1
        ),
        "Brier": brier_score_loss(
            actual1, pred1
        ),
    },
    {
        "Model": "Without previous_cancellations",
        "ROC-AUC": roc_auc_score(
            actual1, pred_without_history
        ),
        "Brier": brier_score_loss(
            actual1, pred_without_history
        ),
    },
])

print(results.round(4).to_string(index=False))


# ============================================
# PART 8 — OCCUPANCY LEAKAGE CHECK
# ============================================

print("\n=== OCCUPANCY LEAKAGE CHECK ===")

# 1. Check training dates
occ_validation_start = va2["stay_date_id"].min()

future_train_nights = (
    tr2["stay_date_id"] >= occ_validation_start
).sum()

print(
    "Training nights on or after validation start:",
    future_train_nights
)

# 2. Check occupancy calculations
invalid_capacity = (
    occ["capacity_rooms"] <= 0
).sum()

invalid_occupancy = (
    (occ["occupancy_rate"] < 0)
    | (occ["occupancy_rate"] > 1)
    | (~np.isfinite(occ["occupancy_rate"]))
).sum()

print("Invalid capacity records:", invalid_capacity)
print("Invalid occupancy records:", invalid_occupancy)

# 3. Check target leakage
target_columns = {
    "occupied_rooms",
    "occupancy_rate",
    "realized_occupancy",
    "capacity_rooms",
}

leaked_features = set(feat2) & target_columns

print(
    "Outcome-related features used:",
    sorted(leaked_features)
)

# 4. Verify calendar predictors
print("\nOccupancy predictors:")
for feature in feat2:
    print("-", feature)

print("\nOccupancy leakage screening completed.")


# ============================================
# PART 9 — INVESTIGATE INVALID OCCUPANCY
# ============================================

print("\n=== INVALID OCCUPANCY INVESTIGATION ===")

invalid_rows = occ[
    (occ["occupancy_rate"] < 0)
    | (occ["occupancy_rate"] > 1)
    | (~np.isfinite(occ["occupancy_rate"]))
].copy()

invalid_rows["occupancy_pct"] = (
    invalid_rows["occupancy_rate"] * 100
)

print("\nNumber of invalid records:", len(invalid_rows))

print("\nInvalid occupancy examples:")
print(
    invalid_rows[
        [
            "stay_date_id",
            "hotel_id",
            "occupied_rooms",
            "capacity_rooms",
            "occupancy_pct",
        ]
    ].head(15).to_string(index=False)
)

print("\nInvalid records by hotel:")
print(invalid_rows.groupby("hotel_id").size())

print("\nInvalid records by dataset split:")

for name, part in [
    ("Training", tr2),
    ("Validation", va2),
    ("Test", te2),
]:
    count = (
        (part["occupancy_rate"] < 0)
        | (part["occupancy_rate"] > 1)
        | (~np.isfinite(part["occupancy_rate"]))
    ).sum()

    print(name, int(count))

print("\nMaximum occupancy:",
      round(occ["occupancy_rate"].max() * 100, 2), "%")

print("Minimum occupancy:",
      round(occ["occupancy_rate"].min() * 100, 2), "%")


# ============================================
# PART 10 — FINAL HELD-OUT TEST EVALUATION
# ============================================

print("\n=== FINAL TEST EVALUATION ===")

# --------------------------------------------
# KPI 1 — Cancellation
# --------------------------------------------

# Use the selected original classifier
# that was trained on leakage-aware data.
cancel_test_pred = clf.predict_proba(
    te1[feat1]
)[:, 1]

cancel_test_actual = te1["is_canceled"]

# Historical cancellation-rate baseline
cancel_test_baseline = np.full(
    len(te1),
    tr1["is_canceled"].mean()
)

cancel_test_results = pd.DataFrame([
    {
        "Model": "Historical Rate Baseline",
        "ROC-AUC": 0.5,
        "Brier": brier_score_loss(
            cancel_test_actual,
            cancel_test_baseline
        ),
    },
    {
        "Model": "Gradient Boosting",
        "ROC-AUC": roc_auc_score(
            cancel_test_actual,
            cancel_test_pred
        ),
        "Brier": brier_score_loss(
            cancel_test_actual,
            cancel_test_pred
        ),
    },
])

print("\nKPI 1 — FINAL TEST")
print(
    cancel_test_results.round(4).to_string(
        index=False
    )
)


# --------------------------------------------
# KPI 2 — Occupancy
# --------------------------------------------

# Use the selected original regressor.
# The corrected occupancy target must
# already be applied before training.
occupancy_test_pred = reg.predict(
    te2[feat2]
)

occupancy_test_actual = te2["occupancy_rate"]

# Historical average occupancy by hotel
occupancy_test_baseline = (
    te2["hotel_id"]
    .map(hotel_means)
    .fillna(tr2["occupancy_rate"].mean())
)

occupancy_test_results = pd.DataFrame([
    {
        "Model": "Historical Hotel Average",
        "MAE_pp": mean_absolute_error(
            occupancy_test_actual,
            occupancy_test_baseline
        ) * 100,
        "RMSE_pp": np.sqrt(
            mean_squared_error(
                occupancy_test_actual,
                occupancy_test_baseline
            )
        ) * 100,
        "R2": r2_score(
            occupancy_test_actual,
            occupancy_test_baseline
        ),
    },
    {
        "Model": "Gradient Boosting",
        "MAE_pp": mean_absolute_error(
            occupancy_test_actual,
            occupancy_test_pred
        ) * 100,
        "RMSE_pp": np.sqrt(
            mean_squared_error(
                occupancy_test_actual,
                occupancy_test_pred
            )
        ) * 100,
        "R2": r2_score(
            occupancy_test_actual,
            occupancy_test_pred
        ),
    },
])

print("\nKPI 2 — FINAL TEST")
print(
    occupancy_test_results.round(4).to_string(
        index=False
    )
)

print("\nFinal held-out test evaluation completed.")



# ============================================
# WEEK 11 — SAVE COMPATIBLE MODEL BUNDLES
# ============================================

import joblib

def create_model_bundle(model, features, task, background):
    preprocessor = model.named_steps["preprocessor"]
    tree_model = model.named_steps["estimator"]

    feature_names = preprocessor.get_feature_names_out()

    return {
        "model": model,
        "features": features,
        "preprocessor": preprocessor,
        "tree_model": tree_model,
        "feature_names_out": feature_names,
        "task": task,
        "background": background,
    }


# Cancellation bundle
cancel_bundle = create_model_bundle(
    model=clf,
    features=feat1,
    task="classification",
    background=tr1[feat1].sample(
        min(200, len(tr1)),
        random_state=SEED
    ),
)

# Occupancy bundle
occupancy_bundle = create_model_bundle(
    model=reg,
    features=feat2,
    task="regression",
    background=tr2[feat2].sample(
        min(200, len(tr2)),
        random_state=SEED
    ),
)

joblib.dump(
    cancel_bundle,
    ROOT / "cancellation_model_wk11.pkl"
)

joblib.dump(
    occupancy_bundle,
    ROOT / "occupancy_model_wk11.pkl"
)

print("\n=== WEEK 11 MODEL BUNDLES SAVED ===")
print("Cancellation:", list(cancel_bundle.keys()))
print("Occupancy:", list(occupancy_bundle.keys()))


# ============================================
# WEEK 11 — BOOKING PACE DATA INSPECTION
# ============================================

print("\n=== BOOKING PACE DATA INSPECTION ===")

with sqlite3.connect(DB) as con:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' ORDER BY name",
        con
    )

print("\nAvailable database tables:")
print(tables.to_string(index=False))

snapshot_tables = [
    name for name in tables["name"]
    if "snapshot" in name.lower()
    or "pace" in name.lower()
]

for table_name in snapshot_tables:
    print(f"\n=== {table_name} ===")

    with sqlite3.connect(DB) as con:
        sample = pd.read_sql_query(
            f'SELECT * FROM "{table_name}" LIMIT 5',
            con
        )

        count = pd.read_sql_query(
            f'SELECT COUNT(*) AS n FROM "{table_name}"',
            con
        )

    print("Rows:", int(count.iloc[0]["n"]))
    print("Columns:", list(sample.columns))
    print(sample.to_string(index=False))


# ============================================
# WEEK 11 — SNAPSHOT QUALITY CHECK
# ============================================

print("\n=== SNAPSHOT QUALITY CHECK ===")

with sqlite3.connect(DB) as con:
    snapshots = pd.read_sql_query(
        """
        SELECT
            s.snapshot_id,
            s.booking_id,
            s.snapshot_date_id,
            s.days_before_arrival,
            s.is_on_books,
            b.hotel_id,
            b.booking_date_id,
            b.arrival_date_id,
            b.status_date_id,
            b.is_canceled
        FROM fact_reservation_snapshot AS s
        JOIN fact_booking AS b
            ON s.booking_id = b.booking_id
        """,
        con
    )

print("Total snapshots:", len(snapshots))
print("Unique bookings:", snapshots["booking_id"].nunique())

print("\nSnapshot lead-time values:")
print(
    snapshots["days_before_arrival"]
    .value_counts()
    .sort_index()
)

print("\nOn-books values:")
print(snapshots["is_on_books"].value_counts())

# Check unique snapshot keys
duplicates = snapshots.duplicated(
    ["booking_id", "snapshot_date_id"]
).sum()

print("\nDuplicate booking/date snapshots:", duplicates)

# Check timing consistency
arrival_dt = pd.to_datetime(
    snapshots["arrival_date_id"].astype(str),
    format="%Y%m%d"
)
snapshot_dt = pd.to_datetime(
    snapshots["snapshot_date_id"].astype(str),
    format="%Y%m%d"
)
booking_dt = pd.to_datetime(
    snapshots["booking_date_id"].astype(str),
    format="%Y%m%d"
)

actual_days_before = (
    arrival_dt - snapshot_dt
).dt.days

date_mismatches = (
    actual_days_before
    != snapshots["days_before_arrival"]
).sum()

print("Lead-time/date mismatches:", date_mismatches)

on_books_before_booking = (
    (snapshots["is_on_books"] == 1)
    & (snapshot_dt < booking_dt)
).sum()

print(
    "On-books before reservation creation:",
    on_books_before_booking
)

print("\nSnapshot quality check completed.")


# ============================================
# WEEK 11 — SNAPSHOT CONSISTENCY AUDIT
# ============================================

print("\n=== SNAPSHOT CONSISTENCY AUDIT ===")

audit = snapshots.copy()

for col in [
    "snapshot_date_id",
    "booking_date_id",
    "status_date_id",
]:
    audit[col + "_dt"] = pd.to_datetime(
        audit[col].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

# Reconstruct whether a reservation should be
# active at the snapshot date.
#
# For non-canceled bookings, assume the booking
# remains active before the scheduled arrival.
# For canceled bookings, use the cancellation date.

booking_exists = (
    audit["booking_date_id_dt"]
    <= audit["snapshot_date_id_dt"]
)

not_yet_canceled = (
    (audit["is_canceled"] == 0)
    |
    (
        audit["status_date_id_dt"]
        > audit["snapshot_date_id_dt"]
    )
)

audit["expected_on_books"] = (
    booking_exists & not_yet_canceled
).astype(int)

audit["mismatch"] = (
    audit["expected_on_books"]
    != audit["is_on_books"]
)

print(
    "Total inconsistent snapshots:",
    int(audit["mismatch"].sum())
)

print(
    "Mismatch rate:",
    round(audit["mismatch"].mean() * 100, 3),
    "%"
)

print("\nMismatch counts by snapshot horizon:")
print(
    audit.groupby("days_before_arrival")[
        "mismatch"
    ].agg(["count", "sum", "mean"])
    .round(4)
)

print("\nExample inconsistent records:")
print(
    audit.loc[audit["mismatch"], [
        "booking_id",
        "snapshot_date_id",
        "booking_date_id",
        "status_date_id",
        "is_canceled",
        "is_on_books",
        "expected_on_books",
    ]].head(10).to_string(index=False)
)

print("\nSnapshot consistency audit completed.")


print("\nDevelopment evaluation completed.")
print("Final test sets were reserved and not evaluated.")
