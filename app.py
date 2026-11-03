
"""
Hotel Revenue Decision Support System
AA-5960 Master's Research Project

Application prototype with two screens:
1. Hotel Overview
2. Overbooking Decision Support

Data: Validated synthetic hotel booking dataset
Model: Logistic Regression cancellation prediction
Optimization: Monte Carlo risk-constrained cost minimization

Note:
This application uses historical synthetic reservations.
Recommendations represent simulation scenarios, not live
operational hotel decisions.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import sqlite3
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from overbooking import calculate_dynamic_buffer
from demand_aware import calculate_demand_aware_buffer
from wk10_screens import show_predictive_screen



# ============================================
# WEEK 11 — OCCUPANCY MODEL LOADER
# ============================================

@st.cache_resource
def load_occupancy_model_wk11():
    model_path = (
        Path(__file__).resolve().parent
        / "occupancy_model_wk11.pkl"
    )

    return joblib.load(model_path)


def predict_occupancy_wk11(model_bundle, hotel_id, stay_date):
    import pandas as pd

    selected_date = pd.Timestamp(stay_date)

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }

    input_row = pd.DataFrame([{
        "month": selected_date.month,
        "week_number": int(selected_date.isocalendar().week),
        "day_of_month": selected_date.day,
        "is_weekend": int(selected_date.weekday() >= 5),
        "year": selected_date.year,
        "hotel_id": hotel_id,
        "season": season_map[selected_date.month],
    }])

    prediction = model_bundle["model"].predict(
        input_row[model_bundle["features"]]
    )[0]

    return float(prediction)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "final" / "hotel_dss.db"
MODEL_PATH = BASE_DIR / "cancellation_model.pkl"

SIMULATIONS = 10000
RANDOM_SEED = 42

st.set_page_config(
    page_title="Hotel Revenue DSS",
    page_icon="🏨",
    layout="wide"
)


# ============================================================
# 3. DATA AND MODEL LOADING
# ============================================================

@st.cache_data
def load_data():
    """
    Load the validated hotel data from SQLite.

    Returns:
        hotels: Hotel information and room capacity
        bookings: Reservation-level data
        daily: Historical hotel-night occupancy
        recommendations: Original baseline recommendations
    """

    with sqlite3.connect(DB_PATH) as conn:

        hotels = pd.read_sql_query(
            "SELECT * FROM dim_hotel",
            conn
        )

        bookings = pd.read_sql_query(
            "SELECT * FROM fact_booking",
            conn
        )

        daily = pd.read_sql_query(
            "SELECT * FROM fact_daily_stay",
            conn
        )

        recommendations = pd.read_sql_query(
            "SELECT * FROM fact_overbooking_recommendation",
            conn
        )

    return hotels, bookings, daily, recommendations


@st.cache_resource
def load_model():
    """
    Load the trained cancellation prediction model.

    The saved bundle contains:
    - model: Trained sklearn pipeline
    - features: Required prediction columns
    """

    bundle = joblib.load(MODEL_PATH)

    return bundle["model"], bundle["features"]


@st.cache_data
def calculate_predictions(_model, booking_data, feature_names):
    """
    Estimate cancellation probability for each reservation.

    The leading underscore in _model tells Streamlit not
    to hash the machine-learning model for cache purposes.
    """

    result = booking_data.copy()

    result["predicted_cancel_prob"] = (
        _model.predict_proba(
            result[feature_names]
        )[:, 1]
    )

    return result


@st.cache_resource
def load_nightly_demand_model():
    model_path = (
        Path(__file__).resolve().parent
        / "nightly_demand_model_wk11.pkl"
    )
    return joblib.load(model_path)


def predict_nightly_demand(
    hotel_id,
    stay_date,
    capacity_rooms,
    on_books_30d
):
    
    
    
    bundle = load_nightly_demand_model()

    date = pd.Timestamp(stay_date)

    inputs = pd.DataFrame([{
        "hotel_id": hotel_id,
        "capacity_rooms": capacity_rooms,
        "on_books_30d": on_books_30d,
        "month": date.month,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
    }])

    predicted = bundle["model"].predict(
        inputs[bundle["features"]]
    )[0]

    return max(0.0, float(predicted))


@st.cache_data
def load_nightly_demand_data():
    path = (
        Path(__file__).resolve().parent
        / "nightly_future_demand_wk11.csv"
    )

    df = pd.read_csv(path)
    df["stay_date"] = pd.to_datetime(df["stay_date"])

    return df


def get_nightly_demand_inputs(hotel_id, stay_date):
    df = load_nightly_demand_data()

    target_date = pd.Timestamp(stay_date).normalize()

    matched = df[
        (df["hotel_id"] == int(hotel_id))
        & (df["stay_date"] == target_date)
    ]

    if len(matched) != 1:
        return None

    row = matched.iloc[0]

    return {
        "capacity_rooms": int(row["capacity_rooms"]),
        "on_books_30d": int(row["on_books_30d"]),
    }

# ============================================================
# 4. DATA PROCESSING FUNCTIONS
# ============================================================

def prepare_booking_dates(bookings):
    """
    Convert date identifiers into datetime values.

    A reservation covers the stay date when:
        arrival_date <= stay_date < checkout_date

    This includes guests who arrived on earlier dates
    but are still occupying rooms on the selected night.
    """

    result = bookings.copy()

    result["arrival_date"] = pd.to_datetime(
        result["arrival_date_id"].astype(str),
        format="%Y%m%d"
    )

    result["checkout_date"] = (
        result["arrival_date"]
        + pd.to_timedelta(
            result["total_nights"],
            unit="D"
        )
    )

    return result


def get_stay_bookings(bookings, hotel_id, stay_date):
    """
    Select all historical reservations covering a stay date.

    Important:
    This is a retrospective simulation population.
    It does not reconstruct which reservations were active
    at a specific historical decision timestamp.
    """

    selected_date = pd.Timestamp(stay_date)

    return bookings[
        (bookings["hotel_id"] == hotel_id)
        & (bookings["arrival_date"] <= selected_date)
        & (bookings["checkout_date"] > selected_date)
    ].copy()
    

def get_bookings_at_forecast_date(
    bookings,
    hotel_id,
    stay_date,
    days_before=30
):
    """
    Reconstruct reservations active at the
    historical 30-day-ahead decision date.

    Only includes reservations covering the
    selected stay night.
    """

    stay_date = pd.Timestamp(stay_date)
    forecast_date = (
        stay_date - pd.Timedelta(days=days_before)
    )

    result = bookings[
        (bookings["hotel_id"] == hotel_id)
        & (bookings["arrival_date"] <= stay_date)
        & (bookings["checkout_date"] > stay_date)
    ].copy()

    result["booking_date"] = pd.to_datetime(
        result["booking_date_id"].astype(str),
        format="%Y%m%d"
    )

    result["status_date"] = pd.to_datetime(
        result["status_date_id"].astype(str),
        format="%Y%m%d"
    )

    active = (
        (result["booking_date"] <= forecast_date)
        & (
            (result["is_canceled"] == 0)
            | (result["status_date"] > forecast_date)
        )
    )

    return result.loc[active].copy()



def prepare_daily_data(daily, hotel_id):
    """
    Prepare historical occupancy records for one hotel.
    """

    result = daily[
        daily["hotel_id"] == hotel_id
    ].copy()

    result["date"] = pd.to_datetime(
        result["stay_date_id"].astype(str),
        format="%Y%m%d"
    )

    result["occupancy_rate"] = (
        result["occupied_rooms"]
        / result["capacity_rooms"] * 100
    )

    return result


def get_baseline_data(daily, recommendations, hotel_id):
    """
    Join historical occupancy with baseline recommendations.

    The baseline uses a fixed hotel-level cancellation rate.
    It is kept for reference, not treated as an optimized
    dynamic recommendation.
    """

    hotel_daily = prepare_daily_data(daily, hotel_id)

    hotel_recommendations = recommendations[
        recommendations["hotel_id"] == hotel_id
    ].copy()

    combined = hotel_daily.merge(
        hotel_recommendations,
        on=["hotel_id", "stay_date_id"],
        how="inner",
        validate="one_to_one"
    )

    return combined.sort_values("date")


# ============================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================

def show_occupancy_chart(hotel_daily):
    """
    Display monthly average historical occupancy.
    """

    monthly = (
        hotel_daily
        .set_index("date")
        .resample("MS")["occupancy_rate"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="date",
        y="occupancy_rate",
        markers=True,
        title="Monthly Average Occupancy Rate",
        labels={
            "date": "Month",
            "occupancy_rate": "Occupancy Rate (%)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def show_cost_chart(simulation_df, recommended_k):
    """
    Compare expected spoilage, walk, and total costs.
    """

    cost_columns = [
        "expected_spoilage_cost",
        "expected_walk_cost",
        "expected_total_cost"
    ]

    fig = px.line(
        simulation_df,
        x="buffer_k",
        y=cost_columns,
        title="Expected Cost vs. Additional Bookings",
        labels={
            "buffer_k": "Additional Bookings",
            "value": "Expected Cost ($)",
            "variable": "Cost Component"
        }
    )

    fig.add_vline(
        x=recommended_k,
        line_dash="dash",
        annotation_text="Recommended"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def show_risk_chart(simulation_df, risk_limit, recommended_k):
    """
    Display walk probability across simulated booking levels.
    """

    fig = px.line(
        simulation_df,
        x="buffer_k",
        y="walk_probability",
        markers=True,
        title="Additional Bookings vs. Walk Risk",
        labels={
            "buffer_k": "Additional Bookings",
            "walk_probability": "Walk Probability"
        }
    )

    fig.add_hline(
        y=risk_limit,
        line_dash="dash",
        annotation_text="Risk Limit"
    )

    fig.add_vline(
        x=recommended_k,
        line_dash="dot",
        annotation_text="Recommended"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 6. SCREEN 1: HOTEL OVERVIEW
# ============================================================

def show_hotel_overview(
    hotel_bookings,
    hotel_daily,
    capacity,
    hotel_name
):
    """
    Display descriptive hotel performance indicators.

    These represent historical synthetic data,
    not forecasts of future hotel performance.
    """

    st.header("Hotel Overview")
    st.caption(f"Historical performance — {hotel_name}")

    total_bookings = len(hotel_bookings)

    cancellation_rate = (
        hotel_bookings["is_canceled"].mean()
    )

    average_adr = hotel_bookings["adr"].mean()

    # Main performance metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Room Capacity",
        capacity
    )

    col2.metric(
        "Total Bookings",
        f"{total_bookings:,}"
    )

    col3.metric(
        "Cancellation Rate",
        f"{cancellation_rate:.1%}"
    )

    col4.metric(
        "Mean Booking ADR",
        f"${average_adr:.2f}"
    )

    st.caption(
        "Mean Booking ADR is the arithmetic mean of the "
        "reservation ADR field, including canceled bookings. "
        "It is not a realized revenue-weighted hotel ADR."
    )

    # Historical occupancy visualization
    st.subheader("Historical Occupancy Trend")

    show_occupancy_chart(hotel_daily)

    # Dataset description
    st.subheader("About This Dataset")

    st.write(
        "This prototype uses a validated synthetic booking "
        "dataset calibrated to historical hotel reservations."
    )

    st.write(
        "The dataset supports cancellation prediction, "
        "room occupancy analysis, and simulation-based "
        "overbooking decisions."
    )


# ============================================================
# 7. SCREEN 2: OVERBOOKING DECISION SUPPORT
# ============================================================

def show_overbooking_support(
    bookings,
    daily,
    recommendations,
    hotel_id,
    capacity,
    average_adr
):
    """
    Display booking-level cancellation predictions,
    risk-constrained cost optimization, and baseline results.
    """

    st.header("Overbooking Decision Support")

    st.caption(
        "Booking-Level Cancellation Prediction "
        "& Risk-Constrained Cost Optimization"
    )

    st.info(
        "This application evaluates historical synthetic "
        "booking scenarios. It uses predicted cancellation "
        "probabilities and simulated financial costs."
    )

    # --------------------------------------------------------
    # 7.1 SELECT HOTEL STAY DATE
    # --------------------------------------------------------

    combined = get_baseline_data(
        daily,
        recommendations,
        hotel_id
    )

    if combined.empty:
        st.warning("No historical stay-date data available.")
        return

    available_dates = sorted(
        combined["date"].dt.date.unique()
    )

    selected_date = st.selectbox(
        "Select Stay Date",
        available_dates
    )

    selected = combined.loc[
        combined["date"].dt.date == selected_date
    ].iloc[0]

    # --------------------------------------------------------
    # 7.2 SELECT BOOKINGS COVERING THE STAY DATE
    # --------------------------------------------------------

    date_bookings = get_stay_bookings(
        bookings,
        hotel_id,
        selected_date
    )

    st.subheader("Booking-Level Cancellation Predictions")

    if date_bookings.empty:
        st.warning(
            "No bookings cover the selected stay date. "
            "The dynamic simulation cannot run."
        )

        return

    predicted_probs = date_bookings[
        "predicted_cancel_prob"
    ].to_numpy()

    avg_risk = predicted_probs.mean()

    expected_non_cancelled = (
        1 - date_bookings["predicted_cancel_prob"]
    ).sum()

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Bookings Covering Stay Date",
        len(date_bookings)
    )

    p2.metric(
        "Average Predicted Cancellation Risk",
        f"{avg_risk:.1%}"
    )

    p3.metric(
        "Expected Non-Cancelled Bookings",
        f"{expected_non_cancelled:.1f}"
    )

    # Optional booking-level details
    with st.expander("View Individual Booking Predictions"):

        st.dataframe(
            date_bookings[
                [
                    "booking_id",
                    "lead_time",
                    "adr",
                    "predicted_cancel_prob"
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # 7.3 FINANCIAL ASSUMPTIONS
    # --------------------------------------------------------

    st.subheader("Financial Assumptions")

    f1, f2, f3 = st.columns(3)

    with f1:
        simulation_adr = st.number_input(
            "Assumed ADR ($)",
            min_value=1.0,
            value=max(1.0, round(float(average_adr), 2)),
            step=5.0
        )

    with f2:
        contribution_margin = st.slider(
            "Contribution Margin (%)",
            min_value=40,
            max_value=90,
            value=70,
            step=5
        )

    with f3:
        walk_multiplier = st.slider(
            "Walk Cost (× ADR)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5
        )

    max_walk_risk_pct = st.slider(
        "Maximum Acceptable Walk Risk (%)",
        min_value=0,
        max_value=20,
        value=5,
        step=1
    )

    st.caption(
        "Financial parameters are adjustable simulation "
        "assumptions rather than observed property costs."
    )



    st.subheader("30-Day Future Demand Forecast")


    forecast_bookings = get_bookings_at_forecast_date(
        bookings=bookings,
        hotel_id=hotel_id,
        stay_date=selected_date,
        days_before=30
    )

    st.caption(
        "Historical reservations reconstructed as of "
        "30 days before the selected stay night."
    )


    demand_inputs = get_nightly_demand_inputs(
        hotel_id,
        selected_date
    )

    if demand_inputs is None:
        st.warning(
            "No historical 30-day demand snapshot is "
            "available for this hotel and stay date."
        )
    else:
        
        snapshot_count = demand_inputs["on_books_30d"]
        reconstructed_count = len(forecast_bookings)

        st.subheader("Forecast Snapshot Validation")

        v1, v2 = st.columns(2)

        v1.metric(
            "Dataset On-the-Books",
            snapshot_count
        )

        v2.metric(
            "Reconstructed Active Bookings",
            reconstructed_count
        )

        if reconstructed_count != snapshot_count:
            st.error(
                "Snapshot mismatch detected. "
                "Do not run demand-aware simulation "
                "until the booking populations match."
            )
        else:
            st.success(
                "Historical booking snapshot matches "
                "the demand forecasting dataset."
            )

        
        predicted_new = predict_nightly_demand(
            hotel_id=hotel_id,
            stay_date=selected_date,
            capacity_rooms=demand_inputs["capacity_rooms"],
            on_books_30d=demand_inputs["on_books_30d"]
        )

        estimated_total = (
            demand_inputs["on_books_30d"]
            + predicted_new
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "On-the-Books at Day -30",
            demand_inputs["on_books_30d"]
        )

        c2.metric(
            "Predicted New Bookings",
            f"{predicted_new:.1f}"
        )

        c3.metric(
            "Projected Booking Volume",
            f"{estimated_total:.1f}"
        )

        st.caption(
            "Projected booking volume is an approximate "
            "booking-count measure, not realized occupancy. "
            "It does not subtract future cancellations."
        )

        st.info(
            "The demand-aware simulation uses the "
            "30-day future demand forecast. "
            "The original Monte Carlo model remains "
            "available separately for comparison."
        )
        

        # ============================================
        # DEMAND-AWARE MONTE CARLO
        # ============================================

        st.subheader("Demand-Aware Overbooking Simulation")

        forecast_probs = forecast_bookings[
            "predicted_cancel_prob"
        ].to_numpy()

        demand_recommendation, demand_results = (
            calculate_demand_aware_buffer(
                cancellation_probs=forecast_probs,
                capacity=capacity,
                predicted_new_demand=predicted_new,
                max_walk_risk=max_walk_risk_pct / 100,
                max_buffer=250,
                simulations=SIMULATIONS,
                seed=RANDOM_SEED,
                adr=simulation_adr,
                contribution_margin=contribution_margin / 100,
                walk_cost_multiplier=walk_multiplier,
            )
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Additional Booking Limit",
            demand_recommendation["buffer_k"]
        )

        d2.metric(
            "Expected Accepted Bookings",
            f'{demand_recommendation["expected_accepted_new"]:.1f}'
        )

        d3.metric(
            "Simulated Walk Risk",
            f'{demand_recommendation["walk_probability"]:.2%}'
        )

        d4.metric(
            "Expected Total Cost",
            f'${demand_recommendation["expected_total_cost"]:,.2f}'
        )

        st.caption(
            "Experimental 30-day-ahead simulation. "
            "Future demand follows an assumed Poisson "
            "distribution. Future booking cancellations "
            "use an assumed rate. Results are simulated "
            "scenarios, not validated operational decisions."
        )

        if not demand_recommendation["feasible"]:
            st.error(
                "No evaluated policy satisfies the "
                "selected walk-risk constraint."
            )
        
        # ============================================
        # FAIR POLICY COMPARISON
        # ============================================

        st.subheader("30-Day-Ahead Policy Comparison")

        demand_df = pd.DataFrame(demand_results)

        current_30d = len(forecast_bookings)

        # Policies share the same 30-day snapshot
        # and demand-aware simulation environment.

        no_overbooking_k = max(
            0,
            capacity - current_30d
        )

        fixed_overbooking_k = max(
            0,
            capacity + 5 - current_30d
        )

        policy_limits = {
            "Stop Accepting New Bookings": 0,
            "No Overbooking": no_overbooking_k,
            "Fixed Overbooking (+5)": fixed_overbooking_k,
            "Demand-Aware Optimization": int(
                demand_recommendation["buffer_k"]
            ),
        }


        # The old model's selected limit is evaluated
        # as a transferred policy, not as a new optimum.
        original_limit = int(recommended["buffer_k"]) \
            if "recommended" in locals() else None

        if original_limit is not None:
            policy_limits["Original Policy Limit"] = original_limit

        comparison_rows = []

        for name, k in policy_limits.items():

            matched = demand_df.loc[
                demand_df["buffer_k"] == k
            ]

            if matched.empty:
                continue

            result = matched.iloc[0]

            comparison_rows.append({
                "Policy": name,
                "Additional Booking Limit": k,
                "Expected Accepted New": float(
                    result["expected_accepted_new"]
                ),
                "Walk Risk (%)": float(
                    result["walk_probability"]
                ) * 100,
                "Expected Cost ($)": float(
                    result["expected_total_cost"]
                ),
                "Risk Compliant": (
                    "Yes"
                    if result["walk_probability"]
                    <= max_walk_risk_pct / 100
                    else "No"
                ),
            })

        fair_comparison = pd.DataFrame(comparison_rows)

        st.dataframe(
            fair_comparison.round(2),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "All policies are evaluated using the same "
            "30-day-ahead booking population, simulated "
            "future demand, cancellation assumptions, "
            "room capacity, and financial parameters. "
            "The original policy limit, when available, "
            "is transferred into this common scenario."
        )

    

    # ============================================
    # WEEK 11 — OCCUPANCY FORECAST
    # ============================================

    occupancy_bundle = load_occupancy_model_wk11()

    predicted_occupancy = predict_occupancy_wk11(
        model_bundle=occupancy_bundle,
        hotel_id=hotel_id,
        stay_date=selected_date
    )

    st.subheader("Expected Nightly Occupancy")

    st.metric(
        "Predicted Occupancy Rate",
        f"{predicted_occupancy * 100:.1f}%"
    )

    st.caption(
        "Calendar-based forecast from the Week 11 "
        "Gradient Boosting model. Final test MAE: "
        "11.36 percentage points; R²: 0.0757. "
        "This forecast is based on synthetic historical "
        "data and does not incorporate live booking pace."
    )


    # --------------------------------------------------------
    # 7.4 RUN MONTE CARLO COST OPTIMIZATION
    # --------------------------------------------------------

    # Extend the search range far enough to include scenarios
    # in which simulated demand may exceed physical capacity.
    max_additional_bookings = max(
        100,
        capacity - len(date_bookings) + 150
    )

    recommended, simulation_results = (
        calculate_dynamic_buffer(
            cancellation_probs=predicted_probs,
            capacity=capacity,
            max_walk_risk=max_walk_risk_pct / 100,
            max_buffer=max_additional_bookings,
            simulations=SIMULATIONS,
            seed=RANDOM_SEED,
            adr=simulation_adr,
            contribution_margin=contribution_margin / 100,
            walk_cost_multiplier=walk_multiplier
        )
    )

    simulation_df = pd.DataFrame(simulation_results)
    
    
    # ============================================
    # WEEK 11 — OVERBOOKING POLICY COMPARISON
    # ============================================

    current_bookings = len(date_bookings)
    fixed_buffer = 5

    # Number of additional reservations allowed
    no_overbooking_k = max(0, capacity - current_bookings)

    fixed_k = max(
        0,
        capacity + fixed_buffer - current_bookings
    )

    dynamic_k = int(recommended["buffer_k"])

    policies = {
        "No Overbooking": no_overbooking_k,
        "Fixed Overbooking (+5)": fixed_k,
        "Dynamic Optimization": dynamic_k
    }

    policy_rows = []

    for strategy, k in policies.items():

        match = simulation_df[
            simulation_df["buffer_k"] == k
        ]

        if match.empty:
            st.error(
                f"Simulation range does not include "
                f"{strategy} (k={k})."
            )
            st.stop()

        result = match.iloc[0]

        risk = float(result["walk_probability"])

        policy_rows.append({
            "Strategy": strategy,
            "Additional Bookings": k,
            "Authorized Limit": current_bookings + k,
            "Expected Empty Rooms": float(
                result["expected_empty_rooms"]
            ),
            "Expected Walked Rooms": float(
                result["expected_walked_rooms"]
            ),
            "Walk Risk (%)": risk * 100,
            "Expected Cost ($)": float(
                result["expected_total_cost"]
            ),
            "Risk Compliant": (
                "Yes" if risk <= max_walk_risk_pct / 100
                else "No"
            )
        })

    policy_df = pd.DataFrame(policy_rows)

    st.subheader("Overbooking Strategy Comparison")

    st.dataframe(
        policy_df.round(2),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "All three policies use the same Monte Carlo "
        "simulation assumptions. Expected cost includes "
        "empty-room opportunity cost and guest relocation "
        "cost, not complete hotel profit."
    )

    # Compare dynamic policy with the no-overbooking policy
    cost_no = policy_df.loc[
        policy_df["Strategy"] == "No Overbooking",
        "Expected Cost ($)"
    ].iloc[0]

    cost_dynamic = policy_df.loc[
        policy_df["Strategy"] == "Dynamic Optimization",
        "Expected Cost ($)"
    ].iloc[0]

    savings = cost_no - cost_dynamic

    st.metric(
    "Expected Cost Reduction vs. No Overbooking",
    f"${savings:,.2f}"
    )
    
    st.caption(
    f"No Overbooking expected cost: ${cost_no:,.2f} | "
    f"Dynamic expected cost: ${cost_dynamic:,.2f}. "
    "The difference is simulated cost reduction, "
    "not guaranteed financial savings."
    )

    if not recommended.get("feasible", False):
        st.error(
            "The dynamic fallback does not satisfy "
            "the selected walk-risk constraint."
        )

    st.caption(
        "Positive cost difference means the dynamic "
        "policy has lower simulated expected cost. "
        "It is not a guaranteed real-world saving."
    )


    # The function's buffer_k means the additional number
    # of reservations accepted in this simulated scenario.
    additional_bookings = int(recommended["buffer_k"])

    authorized_bookings = (
        len(date_bookings) + additional_bookings
    )

    # Standard overbooking buffer is the authorized booking
    # limit above physical room capacity.
    overbooking_buffer = max(
        0,
        authorized_bookings - capacity
    )

    walk_risk = recommended["walk_probability"]
    
    
    # ============================================
    # WEEK 11 — COMBINED DECISION DASHBOARD
    # ============================================

    st.subheader("Integrated Decision Support")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Predicted Occupancy",
        f"{predicted_occupancy * 100:.1f}%"
    )

    m2.metric(
        "Average Cancellation Risk",
        f"{np.mean(predicted_probs) * 100:.1f}%"
        if len(predicted_probs) else "N/A"
    )

    m3.metric(
        "Additional Booking Limit",
        str(additional_bookings)
    )

    m4.metric(
        "Simulated Walk Risk",
        f"{walk_risk * 100:.1f}%"
    )

    st.caption(
        "The cancellation model supports arrival "
        "simulation and booking-limit optimization. "
        "The occupancy model provides an independent "
        "calendar-based demand reference. These outputs "
        "are not added together because that could "
        "double-count demand."
    )
    
    st.warning(
        "Demand feasibility has not been validated. "
        "The Monte Carlo optimization estimates a booking "
        "authorization limit assuming additional reservations "
        "can be obtained. The occupancy model provides a "
        "separate calendar-based forecast and does not "
        "predict future booking arrivals. Therefore, "
        "the authorized booking limit should not be interpreted "
        "as expected demand or a guaranteed revenue outcome."
    )    

    if recommended.get("feasible") is False:
        st.error(
            "No evaluated booking level meets the "
            "selected walk-risk constraint."
        )
    else:
        st.info(
            "The booking limit is feasible under the "
            "selected simulation assumptions. It is not "
            "a guarantee of future demand or profit."
        )

    
    if not recommended["feasible"]:
        st.error(
            "No simulated booking policy satisfies the "
            "selected walk-risk limit. Accepting zero "
            "additional bookings is only a fallback, "
            "not a risk-compliant recommendation."
        )
    else:
        st.success(
            "The recommended booking policy satisfies "
            "the selected simulated walk-risk constraint."
        )

    # --------------------------------------------------------
    # 7.5 DECISION SUMMARY
    # --------------------------------------------------------

    st.subheader("Decision Summary")

    st.markdown(
        f"**Simulated booking limit:** "
        f"Up to {authorized_bookings} total reservations "
        f"({additional_bookings} more than the current "
        f"{len(date_bookings)} reservations), "
        f"subject to the selected assumptions."
    )

    st.caption(
        "This is a conditional inventory limit, not a forecast "
        "that additional demand will materialize. "
        "Future booking arrivals and cancellation behavior "
        "have not been independently validated."
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric(
        "Additional Bookings",
        additional_bookings
    )

    d2.metric(
        "Overbooking Buffer",
        overbooking_buffer
    )

    d3.metric(
        "Authorized Booking Limit",
        authorized_bookings
    )

    d4.metric(
        "Estimated Walk Risk",
        f"{walk_risk:.1%}"
    )

    st.caption(
        "Calculations assume one room per reservation. "
        "The scenario does not reconstruct live inventory "
        "or decision-time booking availability."
    )

    # --------------------------------------------------------
    # 7.6 EXPECTED COST RESULTS
    # --------------------------------------------------------

    st.subheader("Cost Optimization Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Expected Total Cost",
        f"${recommended['expected_total_cost']:,.2f}"
    )

    c2.metric(
        "Expected Spoilage Cost",
        f"${recommended['expected_spoilage_cost']:,.2f}"
    )

    c3.metric(
        "Expected Walk Cost",
        f"${recommended['expected_walk_cost']:,.2f}"
    )

    show_cost_chart(
        simulation_df,
        additional_bookings
    )

    # --------------------------------------------------------
    # 7.7 RISK ANALYSIS
    # --------------------------------------------------------

    st.subheader("Walk Risk Analysis")

    show_risk_chart(
        simulation_df,
        max_walk_risk_pct / 100,
        additional_bookings
    )

    # --------------------------------------------------------
    # 7.8 BASELINE COMPARISON
    # --------------------------------------------------------

    st.subheader("Original Baseline Recommendation")

    baseline_k = int(
        selected["recommended_buffer_k"]
    )

    baseline_authorized = int(
        selected["authorized_rooms"]
    )

    b1, b2, b3 = st.columns(3)

    b1.metric(
        "Physical Room Capacity",
        capacity
    )

    b2.metric(
        "Baseline Buffer",
        baseline_k
    )

    b3.metric(
        "Baseline Authorized Rooms",
        baseline_authorized
    )

    st.caption(
        "The baseline uses the hotel's fixed cancellation "
        "rate. The dynamic simulation uses booking-level "
        "probabilities. These are exploratory results "
        "rather than a controlled policy comparison."
    )

    # --------------------------------------------------------
    # 7.9 HISTORICAL OUTCOME
    # --------------------------------------------------------

    st.subheader("Historical Outcome")

    occupied = int(selected["occupied_rooms"])

    h1, h2 = st.columns(2)

    h1.metric(
        "Realized Occupied Rooms",
        occupied
    )

    h2.metric(
        "Realized Occupancy Rate",
        f"{occupied / capacity:.1%}"
    )

    st.caption(
        "Historical occupancy is an observed outcome. "
        "It is not information available when predicting "
        "future room demand."
    )

    # --------------------------------------------------------
    # 7.10 LIMITATIONS
    # --------------------------------------------------------

    with st.expander("Simulation Assumptions and Limitations"):

        st.write(
            "- Uses synthetic historical hotel booking data.\n"
            "- Cancellation probabilities come from the "
            "trained Logistic Regression model.\n"
            "- Each reservation is assumed to occupy one room.\n"
            "- Cancellation events are simulated independently.\n"
            "- Additional bookings are assumed to have the "
            "mean cancellation probability of existing bookings.\n"
            "- No-show behavior is not separately modeled.\n"
            "- The selected stay-date population is retrospective, "
            "not an as-of-date active booking ledger.\n"
            "- Overlapping stay dates are not jointly optimized.\n"
            "- Cost parameters are illustrative assumptions.\n"
            "- Results should not be used for real bookings."
        )


# ============================================================
# 8. APPLICATION ENTRY POINT
# ============================================================

def main():
    """
    Run the Streamlit application and route the selected page.
    """

    # Check required files before loading.
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.stop()

    if not MODEL_PATH.exists():
        st.error(f"Prediction model not found: {MODEL_PATH}")
        st.stop()

    # Load historical data and trained model.
    hotels, bookings, daily, recommendations = load_data()

    model, features = load_model()

    # Generate booking-level cancellation probabilities.
    bookings = calculate_predictions(
        model,
        bookings,
        features
    )

    # Prepare stay-date coverage once per application run.
    bookings = prepare_booking_dates(bookings)

    # Application title and navigation.
    st.title("🏨 Hotel Revenue Decision Support System")

    st.caption("AA-5960 Master's Research Project")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Hotel Overview",
            "Overbooking Decision Support",
            "Predictive Models & What-If"
        ]
    )

    # Select hotel for both screens.
    hotel_id = st.sidebar.selectbox(
        "Select Hotel",
        hotels["hotel_id"].tolist(),
        format_func=lambda x: hotels.loc[
            hotels["hotel_id"] == x,
            "hotel_name"
        ].iloc[0]
    )

    hotel = hotels.loc[
        hotels["hotel_id"] == hotel_id
    ].iloc[0]

    hotel_name = hotel["hotel_name"]
    capacity = int(hotel["capacity_rooms"])

    hotel_bookings = bookings[
        bookings["hotel_id"] == hotel_id
    ].copy()

    hotel_daily = prepare_daily_data(
        daily,
        hotel_id
    )

    average_adr = hotel_bookings["adr"].mean()

    # Route to the selected screen.
    if page == "Hotel Overview":

        show_hotel_overview(
            hotel_bookings=hotel_bookings,
            hotel_daily=hotel_daily,
            capacity=capacity,
            hotel_name=hotel_name
        )

    elif page == "Overbooking Decision Support":

        show_overbooking_support(
            bookings=bookings,
            daily=daily,
            recommendations=recommendations,
            hotel_id=hotel_id,
            capacity=capacity,
            average_adr=average_adr
        )

    else:

        show_predictive_screen(
            bookings=bookings,
            hotel_id=hotel_id
        )


if __name__ == "__main__":
    main()




