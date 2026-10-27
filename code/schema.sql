-- ============================================================
-- Synthetic Hotel Overbooking DSS - relational schema (SQLite)
-- Near-3NF. Canonical tables enforce PK/FK/CHECK integrity.
-- Outcome fields (known only AFTER the decision) are marked.
-- ============================================================

PRAGMA foreign_keys = ON;

-- ---------- Dimension / reference tables ----------
CREATE TABLE dim_hotel (
    hotel_id            INTEGER PRIMARY KEY,
    hotel_name          TEXT    NOT NULL,
    hotel_type          TEXT    NOT NULL CHECK (hotel_type IN ('Resort','City')),
    capacity_rooms      INTEGER NOT NULL CHECK (capacity_rooms > 0),
    capacity_percentile REAL    NOT NULL           -- percentile of occupancy used to set capacity
);

CREATE TABLE dim_date (
    date_id       INTEGER PRIMARY KEY,             -- yyyymmdd
    full_date     TEXT    NOT NULL UNIQUE,
    year          INTEGER NOT NULL,
    month         INTEGER NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name    TEXT    NOT NULL,
    week_number   INTEGER NOT NULL,
    day_of_month  INTEGER NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_week   INTEGER NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    is_weekend    INTEGER NOT NULL CHECK (is_weekend IN (0,1)),
    season        TEXT    NOT NULL
);

CREATE TABLE ref_market_segment (
    segment_code TEXT PRIMARY KEY,
    description  TEXT NOT NULL
);

CREATE TABLE ref_distribution_channel (
    channel_code TEXT PRIMARY KEY,
    description  TEXT NOT NULL
);

CREATE TABLE ref_deposit_type (
    deposit_type_code TEXT PRIMARY KEY,
    cancel_risk_note  TEXT
);

-- GENERATED economic parameters, with provenance label (assumption vs calibrated)
CREATE TABLE dim_cost_parameter (
    param_id                INTEGER PRIMARY KEY,
    hotel_id                INTEGER NOT NULL REFERENCES dim_hotel(hotel_id),
    contribution_margin_pct REAL    NOT NULL CHECK (contribution_margin_pct BETWEEN 0 AND 1),
    spoilage_cost_per_room  REAL    NOT NULL CHECK (spoilage_cost_per_room >= 0),
    walk_cost_mult_low      REAL    NOT NULL CHECK (walk_cost_mult_low >= 0),
    walk_cost_mult_high     REAL    NOT NULL CHECK (walk_cost_mult_high >= walk_cost_mult_low),
    provenance              TEXT    NOT NULL,       -- e.g. expert_assumption / source_calibrated
    effective_date          TEXT    NOT NULL
);

-- ---------- Core fact table: bookings (calibrated to Antonio) ----------
CREATE TABLE fact_booking (
    booking_id              INTEGER PRIMARY KEY,
    hotel_id                INTEGER NOT NULL REFERENCES dim_hotel(hotel_id),
    booking_date_id         INTEGER NOT NULL REFERENCES dim_date(date_id),
    arrival_date_id         INTEGER NOT NULL REFERENCES dim_date(date_id),
    segment_code            TEXT    NOT NULL REFERENCES ref_market_segment(segment_code),
    channel_code            TEXT    NOT NULL REFERENCES ref_distribution_channel(channel_code),
    deposit_type_code       TEXT    NOT NULL REFERENCES ref_deposit_type(deposit_type_code),
    reserved_room_type      TEXT    NOT NULL,
    lead_time               INTEGER NOT NULL CHECK (lead_time >= 0),
    stays_week_nights       INTEGER NOT NULL CHECK (stays_week_nights >= 0),
    stays_weekend_nights    INTEGER NOT NULL CHECK (stays_weekend_nights >= 0),
    total_nights            INTEGER NOT NULL CHECK (total_nights >= 1),
    adults                  INTEGER NOT NULL CHECK (adults >= 0),
    children                INTEGER NOT NULL CHECK (children >= 0),
    babies                  INTEGER NOT NULL CHECK (babies >= 0),
    adr                     REAL    NOT NULL CHECK (adr >= 0),      -- no negative ADR (unlike raw data)
    is_repeated_guest       INTEGER NOT NULL CHECK (is_repeated_guest IN (0,1)),
    previous_cancellations  INTEGER NOT NULL CHECK (previous_cancellations >= 0),
    booking_changes         INTEGER NOT NULL CHECK (booking_changes >= 0),
    total_special_requests  INTEGER NOT NULL CHECK (total_special_requests >= 0),
    customer_type           TEXT    NOT NULL,
    meal                    TEXT    NOT NULL,
    country                 TEXT    NOT NULL,
    -- ----- OUTCOME fields: known only AFTER the decision. Exclude from predictive features. -----
    is_canceled             INTEGER NOT NULL CHECK (is_canceled IN (0,1)),
    reservation_status      TEXT    NOT NULL CHECK (reservation_status IN ('Check-Out','Canceled','No-Show')),
    status_date_id          INTEGER NOT NULL REFERENCES dim_date(date_id),
    CHECK (total_nights = stays_week_nights + stays_weekend_nights),
    CHECK (adults + children > 0)
);

-- ---------- GENERATED decision-time state (leakage-safe booking pace) ----------
-- One row per booking per snapshot offset while the booking was on the books.
-- is_on_books reflects CONTEMPORANEOUS state only; the final outcome is NOT used here.
CREATE TABLE fact_reservation_snapshot (
    snapshot_id        INTEGER PRIMARY KEY,
    booking_id         INTEGER NOT NULL REFERENCES fact_booking(booking_id),
    snapshot_date_id   INTEGER NOT NULL REFERENCES dim_date(date_id),
    days_before_arrival INTEGER NOT NULL CHECK (days_before_arrival >= 0),
    is_on_books        INTEGER NOT NULL CHECK (is_on_books IN (0,1))
);

-- ---------- Derived nightly mart (one row per hotel per stay-date) ----------
CREATE TABLE fact_daily_stay (
    hotel_id       INTEGER NOT NULL REFERENCES dim_hotel(hotel_id),
    stay_date_id   INTEGER NOT NULL REFERENCES dim_date(date_id),
    occupied_rooms INTEGER NOT NULL CHECK (occupied_rooms >= 0),   -- realized (arrived) room-nights
    arrivals       INTEGER NOT NULL CHECK (arrivals >= 0),
    capacity_rooms INTEGER NOT NULL CHECK (capacity_rooms > 0),
    PRIMARY KEY (hotel_id, stay_date_id)
);

-- ---------- GENERATED overbooking outcomes (walks) under historical baseline ----------
CREATE TABLE fact_walk_event (
    walk_event_id   INTEGER PRIMARY KEY,
    hotel_id        INTEGER NOT NULL REFERENCES dim_hotel(hotel_id),
    stay_date_id    INTEGER NOT NULL REFERENCES dim_date(date_id),
    occupied_rooms  INTEGER NOT NULL CHECK (occupied_rooms >= 0),
    capacity_rooms  INTEGER NOT NULL CHECK (capacity_rooms > 0),
    guests_walked   INTEGER NOT NULL CHECK (guests_walked > 0),
    avg_adr         REAL    NOT NULL CHECK (avg_adr >= 0),
    relocation_cost REAL    NOT NULL CHECK (relocation_cost >= 0)
);

-- ---------- Model-ops: what the DSS would recommend (derived, illustrative) ----------
CREATE TABLE fact_overbooking_recommendation (
    rec_id                INTEGER PRIMARY KEY,
    hotel_id              INTEGER NOT NULL REFERENCES dim_hotel(hotel_id),
    stay_date_id          INTEGER NOT NULL REFERENCES dim_date(date_id),
    expected_cancellations REAL   NOT NULL CHECK (expected_cancellations >= 0),
    recommended_buffer_k  INTEGER NOT NULL CHECK (recommended_buffer_k >= 0),
    authorized_rooms      INTEGER NOT NULL CHECK (authorized_rooms > 0)
);

-- ---------- Indexes for joins / dashboard filters ----------
CREATE INDEX idx_booking_hotel_arr ON fact_booking(hotel_id, arrival_date_id);
CREATE INDEX idx_booking_status    ON fact_booking(is_canceled, reservation_status);
CREATE INDEX idx_snapshot_booking  ON fact_reservation_snapshot(booking_id, snapshot_date_id);
CREATE INDEX idx_daily_hotel_date  ON fact_daily_stay(hotel_id, stay_date_id);
CREATE INDEX idx_walk_hotel_date   ON fact_walk_event(hotel_id, stay_date_id);
