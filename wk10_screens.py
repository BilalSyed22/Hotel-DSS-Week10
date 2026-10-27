"""
Week 10 — Predictive Modeling, Explainability, and What-If Analysis.

Adds a screen to the Hotel Revenue DSS that, for each KPI:
  - shows overall predictor importance (whole model, mean |SHAP|),
  - lets the user run a what-if scenario and compares it to a fixed baseline,
  - shows predictor contributions for that single prediction (local SHAP),
  - interprets direction / magnitude / practical meaning (not causation),
  - auto-generates a plain-language summary with implications and limitations.

Models (trained in train_models_wk10.py, Gradient Boosting + SHAP TreeExplainer):
  KPI 1  cancellation_model.pkl   cancellation probability   (classification)
  KPI 2  occupancy_model.pkl      nightly occupancy rate     (regression)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Model loading and SHAP helpers
# --------------------------------------------------------------------------
@st.cache_resource
def load_wk10_models():
    return (joblib.load(BASE_DIR / "cancellation_model.pkl"),
            joblib.load(BASE_DIR / "occupancy_model.pkl"))


def _transform(bundle, row_df):
    Xt = bundle["preprocessor"].transform(row_df)
    return Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)


def _shap_values(bundle, Xt):
    """Return SHAP values aligned to feature_names_out and the model's base value."""
    expl = shap.TreeExplainer(bundle["tree_model"])
    sv = np.array(expl.shap_values(Xt))
    if sv.ndim == 3:                                  # (rows, feats, classes)
        sv = sv[..., 1] if sv.shape[-1] > 1 else sv[..., 0]
    ev = np.atleast_1d(expl.expected_value)
    ev = float(ev[-1]) if ev.size > 1 else float(ev[0])
    return sv, ev


def predict(bundle, row_df):
    """One scalar prediction for a one-row input frame."""
    if bundle["task"] == "classification":
        return float(bundle["model"].predict_proba(row_df)[:, 1][0])
    return float(bundle["model"].predict(row_df)[0])


@st.cache_data
def global_importance(_bundle, X_sample, key):
    """Mean |SHAP| per feature across a sample = overall importance (whole model)."""
    Xt = _transform(_bundle, X_sample)
    sv, _ = _shap_values(_bundle, Xt)
    imp = np.abs(sv).mean(axis=0)
    out = (pd.DataFrame({"feature": _bundle["feature_names_out"], "importance": imp})
           .sort_values("importance", ascending=False).head(10))
    return out


def local_contributions(bundle, row_df):
    """SHAP contributions for one prediction (this case only)."""
    Xt = _transform(bundle, row_df)
    sv, ev = _shap_values(bundle, Xt)
    sv = np.asarray(sv).reshape(-1)
    out = pd.DataFrame({"feature": bundle["feature_names_out"], "contribution": sv})
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).head(8), ev


def fmt(bundle, v):
    return f"{v:.1%}" if bundle["task"] == "classification" else f"{v*100:.1f}%"


# --------------------------------------------------------------------------
# KPI metadata (justification text shown in the app)
# --------------------------------------------------------------------------
KPI1_WHY = (
    "**What it measures.** The probability that a single reservation will be "
    "cancelled before arrival.\n\n"
    "**Why it matters.** Cancellations are the core uncertainty the revenue "
    "manager faces. A reliable per-booking cancellation probability is what lets "
    "the hotel sell slightly more rooms than it has (overbooking) without walking "
    "guests, which protects both occupancy revenue and guest goodwill.")
KPI2_WHY = (
    "**What it measures.** The expected share of rooms occupied on a future "
    "stay-date (0–100%).\n\n"
    "**Why it matters.** Expected occupancy is the demand signal behind pricing "
    "and how aggressively to overbook. High expected occupancy supports higher "
    "rates and tighter inventory; low expected occupancy supports discounts and "
    "more cautious overbooking.")


# --------------------------------------------------------------------------
# What-if input panels (return a one-row DataFrame of raw features)
# --------------------------------------------------------------------------
def cancel_inputs(bookings, defaults):
    seg = sorted(bookings["segment_code"].dropna().unique())
    chan = sorted(bookings["channel_code"].dropna().unique())
    dep = sorted(bookings["deposit_type_code"].dropna().unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        lead = st.slider("Lead time (days before arrival)", 0, 600, defaults["lead_time"], 5)
        adr = st.number_input("ADR — room price ($)", 10.0, 600.0, float(defaults["adr"]), 5.0)
        nights = st.slider("Total nights", 1, 21, defaults["total_nights"])
    with c2:
        segment = st.selectbox("Market segment", seg, index=seg.index(defaults["segment_code"]))
        channel = st.selectbox("Booking channel", chan, index=chan.index(defaults["channel_code"]))
        deposit = st.selectbox("Deposit type", dep, index=dep.index(defaults["deposit_type_code"]))
    with c3:
        adults = st.slider("Adults", 1, 4, defaults["adults"])
        children = st.slider("Children", 0, 3, defaults["children"])
        prev = st.slider("Previous cancellations", 0, 5, defaults["previous_cancellations"])
        repeat = st.selectbox("Repeat guest?", [0, 1],
                              index=defaults["is_repeated_guest"],
                              format_func=lambda x: "Yes" if x else "No")
        special = st.slider("Special requests", 0, 5, defaults["total_special_requests"])
    return pd.DataFrame([{
        "lead_time": lead, "adr": adr, "adults": adults, "children": children,
        "previous_cancellations": prev, "is_repeated_guest": repeat,
        "total_special_requests": special, "total_nights": nights,
        "hotel_id": defaults["hotel_id"], "segment_code": segment,
        "channel_code": channel, "deposit_type_code": deposit}])


def occ_inputs(defaults):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.slider("Month (1–12)", 1, 12, defaults["month"])
        season = st.selectbox("Season", seasons, index=seasons.index(defaults["season"]))
    with c2:
        weekend = st.selectbox("Weekend night?", [0, 1], index=defaults["is_weekend"],
                               format_func=lambda x: "Yes" if x else "No")
        week = st.slider("Week number", 1, 53, defaults["week_number"])
    with c3:
        dom = st.slider("Day of month", 1, 31, defaults["day_of_month"])
        year = st.selectbox("Year", [2015, 2016, 2017],
                            index=[2015, 2016, 2017].index(defaults["year"]))
    return pd.DataFrame([{
        "month": month, "week_number": week, "day_of_month": dom,
        "is_weekend": weekend, "year": year,
        "hotel_id": defaults["hotel_id"], "season": season}])


# --------------------------------------------------------------------------
# Auto-generated summary
# --------------------------------------------------------------------------
def auto_summary(bundle, kpi_name, inputs_row, baseline_pred, scenario_pred, top_local):
    change = scenario_pred - baseline_pred
    direction = "no change" if abs(change) < 1e-6 else ("higher" if change > 0 else "lower")
    inp = ", ".join(f"{k} = {v}" for k, v in inputs_row.iloc[0].items())
    drivers = ", ".join(f"{r.feature} ({'+' if r.contribution >= 0 else ''}{r.contribution:.3f})"
                        for _, r in top_local.head(3).iterrows())
    unit = "percentage points" if bundle["task"] == "classification" else "occupancy points"
    delta = abs(change) * 100
    lines = [
        f"**Scenario inputs:** {inp}.",
        f"**Baseline {kpi_name}:** {fmt(bundle, baseline_pred)}.",
        f"**Scenario {kpi_name}:** {fmt(bundle, scenario_pred)} "
        f"({direction}" + (f", {delta:.1f} {unit} vs. baseline)." if direction != "no change" else ")."),
        f"**Main drivers of this prediction:** {drivers}.",
        "**Implications.** " + (
            "A higher predicted cancellation probability suggests this booking profile is "
            "riskier, so the manager might request a deposit or confirmation, or lean on it "
            "when sizing the overbooking buffer."
            if bundle["task"] == "classification" else
            "A higher predicted occupancy suggests stronger demand for this date, supporting "
            "firmer pricing and a tighter, more confident overbooking stance; lower predicted "
            "occupancy supports discounts and caution."),
        "**Limitations.** This is a model-based scenario, not a guaranteed real-world outcome. "
        "The contributions show association learned from historical synthetic data, not cause "
        "and effect. Changing an input shows how the model responds, not what would happen if "
        "the hotel actually changed that factor.",
    ]
    return "\n\n".join(lines)


def contribution_chart(df, title):
    d = df.sort_values("contribution")
    fig = px.bar(d, x="contribution", y="feature", orientation="h", title=title,
                 color="contribution", color_continuous_scale=["#c0392b", "#dddddd", "#1f6f3f"],
                 labels={"contribution": "Push on the prediction", "feature": ""})
    fig.update_layout(coloraxis_showscale=False, height=330, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Main screen
# --------------------------------------------------------------------------
def show_predictive_screen(bookings, hotel_id):
    st.header("Predictive Models & What-If Analysis")
    st.caption("Two KPIs, each with a predictive model, SHAP explanations, and an interactive what-if.")
    st.info("Results are model-based scenarios from historical synthetic data. "
            "Predictor contributions show association, not causation.")

    cancel_b, occ_b = load_wk10_models()

    kpi = st.radio("Choose a KPI", ["KPI 1 — Cancellation Probability",
                                    "KPI 2 — Expected Occupancy Rate"], horizontal=True)
    is_cancel = kpi.startswith("KPI 1")
    bundle = cancel_b if is_cancel else occ_b
    kpi_name = "cancellation probability" if is_cancel else "occupancy rate"

    # --- 1. KPI justification + model card ---
    st.subheader("1. The KPI and the model")
    st.markdown(KPI1_WHY if is_cancel else KPI2_WHY)
    if is_cancel:
        st.markdown("**Model:** Gradient Boosting Classifier · out-of-time ROC-AUC ≈ 0.88 · "
                    "Brier ≈ 0.12.  \n**Predictors:** lead time, ADR, party size, prior "
                    "cancellations, repeat-guest flag, special requests, nights, hotel, "
                    "segment, channel, deposit type.")
        st.caption("Why this algorithm: cancellation depends on non-linear interactions "
                   "(e.g., long lead time *and* non-refundable deposit); gradient boosting "
                   "captures these and pairs with SHAP for exact per-booking explanations.")
    else:
        st.markdown("**Model:** Gradient Boosting Regressor · out-of-time R² ≈ 0.18 · "
                    "MAE ≈ 10 occupancy points.  \n**Predictors:** month, season, week "
                    "number, day of month, weekend flag, year, hotel.")
        st.caption("Why this algorithm: demand is seasonal and non-linear across the "
                   "calendar; gradient boosting captures the seasonal shape. The modest R² "
                   "reflects real night-to-night demand noise, so the figure is a guide used "
                   "alongside manager judgment, not a guarantee.")

    # --- 2. Overall importance (whole model) ---
    st.subheader("2. Overall predictor importance (whole model)")
    st.caption("Average size of each predictor's effect across many cases — which factors the "
               "model relies on in general. This is different from the single-prediction "
               "contributions in section 4.")
    if is_cancel:
        Xs = bookings[bundle["features"]].sample(min(400, len(bookings)), random_state=1)
    else:
        import sqlite3
        con = sqlite3.connect(BASE_DIR / "data" / "final" / "hotel_dss.db")
        daily = pd.read_sql_query("SELECT * FROM fact_daily_stay", con)
        dd = pd.read_sql_query("SELECT * FROM dim_date", con); con.close()
        occ = daily.merge(dd, left_on="stay_date_id", right_on="date_id")
        Xs = occ[bundle["features"]].sample(min(400, len(occ)), random_state=1)
    gi = global_importance(bundle, Xs, key=kpi)
    fig = px.bar(gi.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title="Overall importance (mean |SHAP|)", labels={"importance": "Mean |SHAP|", "feature": ""})
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. What-if ---
    st.subheader("3. What-if analysis")
    st.caption("Adjust the inputs and compare the scenario prediction with the fixed baseline.")
    if is_cancel:
        defaults = dict(lead_time=90, adr=100.0, adults=2, children=0, previous_cancellations=0,
                        is_repeated_guest=0, total_special_requests=0, total_nights=3,
                        hotel_id=hotel_id, segment_code="Online TA", channel_code="TA/TO",
                        deposit_type_code="No Deposit")
        # guard defaults against values not present in data
        for k, col in [("segment_code", "segment_code"), ("channel_code", "channel_code"),
                       ("deposit_type_code", "deposit_type_code")]:
            vals = sorted(bookings[col].dropna().unique())
            if defaults[k] not in vals: defaults[k] = vals[0]
        baseline_row = pd.DataFrame([defaults])
        scenario_row = cancel_inputs(bookings, defaults)
    else:
        defaults = dict(month=7, week_number=28, day_of_month=15, is_weekend=0, year=2016,
                        hotel_id=hotel_id, season="Summer")
        baseline_row = pd.DataFrame([defaults])
        scenario_row = occ_inputs(defaults)

    baseline_pred = predict(bundle, baseline_row)
    scenario_pred = predict(bundle, scenario_row)
    change = scenario_pred - baseline_pred

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline prediction", fmt(bundle, baseline_pred))
    m2.metric("Scenario prediction", fmt(bundle, scenario_pred),
              delta=f"{change*100:+.1f} pts")
    m3.metric("Change vs. baseline", f"{change*100:+.1f} pts")
    st.caption("The baseline is a fixed, typical case. Your scenario is compared against it. "
               "An input change may or may not move the prediction.")

    # --- 4. Local contributions (this prediction only) ---
    st.subheader("4. Why this prediction (this case only)")
    st.caption("How each input pushed the scenario prediction up (green) or down (red), "
               "relative to the model's average case. This explains one prediction, not the "
               "whole model.")
    top_local, base_val = local_contributions(bundle, scenario_row)
    contribution_chart(top_local, "Predictor contributions for your scenario")

    # --- 5. Contextual interpretation ---
    st.subheader("5. Interpretation")
    top3 = top_local.head(3)
    bullet_txt = ""
    for _, r in top3.iterrows():
        d = "raises" if r.contribution >= 0 else "lowers"
        bullet_txt += (f"- **{r.feature}** {d} the predicted {kpi_name} by about "
                       f"{abs(r.contribution)*100:.1f} points in this scenario.\n")
    st.markdown(bullet_txt)
    st.markdown("_Direction_ is the sign of each bar, _magnitude_ is its length. These are "
                "associations the model learned from historical data; they are **not** evidence "
                "that changing the factor would cause the outcome in the real world.")

    # --- 6. Auto-generated summary ---
    st.subheader("6. Auto-generated scenario summary")
    st.markdown(auto_summary(bundle, kpi_name, scenario_row, baseline_pred, scenario_pred, top_local))
