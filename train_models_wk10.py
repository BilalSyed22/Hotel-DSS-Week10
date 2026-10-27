"""
Week 10 — Predictive model training for the Hotel Revenue DSS.

Trains two Gradient Boosting models, one per KPI, and saves bundles that the
Streamlit app loads for prediction, SHAP explanations, and what-if analysis.

KPI 1: Booking cancellation probability   (classification)
KPI 2: Expected nightly occupancy rate     (regression)

Gradient Boosting is used for both because it captures non-linear effects and
interactions and pairs directly with SHAP TreeExplainer for fast, exact
per-prediction explanations.
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "final" / "hotel_dss.db"
SEED = 42

con = sqlite3.connect(DB)
bookings = pd.read_sql_query("SELECT * FROM fact_booking", con)
daily = pd.read_sql_query("SELECT * FROM fact_daily_stay", con)
dim_date = pd.read_sql_query("SELECT * FROM dim_date", con)
hotels = pd.read_sql_query("SELECT * FROM dim_hotel", con)
con.close()


def make_explainable_bundle(pipeline, raw_features):
    """Pull the preprocessor and raw tree model out of a fitted pipeline so the
    app can run SHAP TreeExplainer on the transformed features."""
    preproc = pipeline.named_steps["preprocessor"]
    tree = pipeline.named_steps["estimator"]
    names = list(preproc.get_feature_names_out())
    # clean the sklearn prefixes ("num__lead_time" -> "lead_time")
    clean = [n.split("__", 1)[1] if "__" in n else n for n in names]
    return {
        "model": pipeline,
        "features": raw_features,
        "preprocessor": preproc,
        "tree_model": tree,
        "feature_names_out": clean,
    }


# ======================================================================
# KPI 1 — Cancellation probability (classification)
# ======================================================================
num1 = ["lead_time", "adr", "adults", "children", "previous_cancellations",
        "is_repeated_guest", "total_special_requests", "total_nights"]
cat1 = ["hotel_id", "segment_code", "channel_code", "deposit_type_code"]
feat1 = num1 + cat1

df1 = bookings.sort_values("arrival_date_id").reset_index(drop=True)
X1, y1 = df1[feat1], df1["is_canceled"]
cut = int(len(df1) * 0.8)                       # chronological (out-of-time) split
X1tr, X1te, y1tr, y1te = X1.iloc[:cut], X1.iloc[cut:], y1.iloc[:cut], y1.iloc[cut:]

pre1 = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num1),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat1),
])
clf = Pipeline([("preprocessor", pre1),
                ("estimator", GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=SEED))])
clf.fit(X1tr, y1tr)
p = clf.predict_proba(X1te)[:, 1]
print("KPI1 Cancellation  ROC-AUC:", round(roc_auc_score(y1te, p), 4),
      " Brier:", round(brier_score_loss(y1te, p), 4))

b1 = make_explainable_bundle(clf, feat1)
b1["task"] = "classification"
# keep a small transformed background sample for display/averages
b1["background"] = pre1.transform(X1tr.sample(min(500, len(X1tr)), random_state=SEED))
joblib.dump(b1, ROOT / "cancellation_model.pkl")     # replaces the Week 9 model (same interface)
print("  saved cancellation_model.pkl (Gradient Boosting, SHAP-ready)")


# ======================================================================
# KPI 2 — Expected nightly occupancy rate (regression)
# ======================================================================
occ = daily.merge(dim_date, left_on="stay_date_id", right_on="date_id", how="left")
occ["occupancy_rate"] = occ["occupied_rooms"] / occ["capacity_rooms"]
num2 = ["month", "week_number", "day_of_month", "is_weekend", "year"]
cat2 = ["hotel_id", "season"]
feat2 = num2 + cat2

occ = occ.sort_values("stay_date_id").reset_index(drop=True)
X2, y2 = occ[feat2], occ["occupancy_rate"]
cut2 = int(len(occ) * 0.8)
X2tr, X2te, y2tr, y2te = X2.iloc[:cut2], X2.iloc[cut2:], y2.iloc[:cut2], y2.iloc[cut2:]

pre2 = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num2),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat2),
])
reg = Pipeline([("preprocessor", pre2),
                ("estimator", GradientBoostingRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED))])
reg.fit(X2tr, y2tr)
pr = reg.predict(X2te)
print("KPI2 Occupancy     R2:", round(r2_score(y2te, pr), 4),
      " MAE:", round(mean_absolute_error(y2te, pr), 4))

b2 = make_explainable_bundle(reg, feat2)
b2["task"] = "regression"
b2["background"] = pre2.transform(X2tr.sample(min(500, len(X2tr)), random_state=SEED))
b2["baseline_mean"] = float(y2tr.mean())
joblib.dump(b2, ROOT / "occupancy_model.pkl")
print("  saved occupancy_model.pkl (Gradient Boosting, SHAP-ready)")
print("Done.")
