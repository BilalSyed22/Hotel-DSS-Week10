#!/usr/bin/env python3
"""Validation: database integrity + statistical fidelity vs real Antonio targets.
Writes reports/quality_report.md and reports/validation_metrics.csv. Exit 1 on any FAIL."""
import os, sqlite3, json
import pandas as pd, numpy as np

HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB=os.path.join(HERE,"data","final","hotel_dss.db")
REP=os.path.join(HERE,"reports"); os.makedirs(REP,exist_ok=True)

# real targets extracted from H1/H2 (see calibration_targets.json)
TARGET={"cancel_resort":0.278,"cancel_city":0.417,"cancel_overall":0.370,
        "noshow_resort":0.0073,"noshow_city":0.0115,
        "nonrefund_cancel_min":0.90,"walk_rate_lo":0.005,"walk_rate_hi":0.025}

con=sqlite3.connect(DB)
T={t:pd.read_sql(f"SELECT * FROM {t}",con) for t in
   ["dim_hotel","dim_date","fact_booking","fact_reservation_snapshot",
    "fact_daily_stay","fact_walk_event","fact_overbooking_recommendation"]}
rows=[]
def chk(name,ok,detail=""):
    rows.append({"check":name,"status":"PASS" if ok else "FAIL","detail":detail})

# ---------- BASELINE RECOMMENDATION VALIDATION ----------

rec = T["fact_overbooking_recommendation"].copy()
hotel = T["dim_hotel"]

# Recommended buffer must be non-negative
chk(
    "recommended buffer >= 0",
    (rec["recommended_buffer_k"] >= 0).all()
)

# Authorized rooms must equal capacity + buffer
merged = rec.merge(
    hotel[["hotel_id", "capacity_rooms"]],
    on="hotel_id",
    how="left"
)

chk(
    "authorized rooms = capacity + buffer",
    (
        merged["authorized_rooms"]
        == merged["capacity_rooms"] + merged["recommended_buffer_k"]
    ).all()
)

# Each hotel-date must have one recommendation
chk(
    "unique hotel-date recommendation",
    ~rec.duplicated(["hotel_id", "stay_date_id"]).any()
)

# Recommendation dates must correspond to daily stay records
daily_keys = set(
    zip(
        T["fact_daily_stay"]["hotel_id"],
        T["fact_daily_stay"]["stay_date_id"]
    )
)

rec_keys = set(
    zip(rec["hotel_id"], rec["stay_date_id"])
)

chk(
    "recommendation keys match daily stay",
    rec_keys == daily_keys
)

# ---------- INTEGRITY ----------
fb=T["fact_booking"]
chk("booking_id unique & non-null", fb["booking_id"].is_unique and fb["booking_id"].notna().all())
dids=set(T["dim_date"]["date_id"])
for col in ["booking_date_id","arrival_date_id","status_date_id"]:
    orph=(~fb[col].isin(dids)).sum(); chk(f"FK {col} -> dim_date (orphans=0)",orph==0,f"{orph} orphans")
chk("FK snapshot.booking_id -> booking", T["fact_reservation_snapshot"]["booking_id"].isin(set(fb["booking_id"])).all())
chk("ADR >= 0", (fb["adr"]>=0).all())
chk("lead_time >= 0", (fb["lead_time"]>=0).all())
chk("adults+children > 0", ((fb["adults"]+fb["children"])>0).all())
chk("total_nights = week+weekend", (fb["total_nights"]==fb["stays_week_nights"]+fb["stays_weekend_nights"]).all())
chk("booking_date <= arrival_date", (fb["booking_date_id"]<=fb["arrival_date_id"]).all())
# chronology: cancellations happen on/before arrival
cxl=fb[fb["is_canceled"]==1]
chk("cancel status_date <= arrival", (cxl["status_date_id"]<=cxl["arrival_date_id"]).all())
# snapshots never after arrival
sn=T["fact_reservation_snapshot"].merge(fb[["booking_id","arrival_date_id"]],on="booking_id")
chk("snapshot_date <= arrival_date", (sn["snapshot_date_id"]<=sn["arrival_date_id"]).all())

# ---------- STATISTICAL FIDELITY ----------
hid={r.hotel_type:r.hotel_id for r in T["dim_hotel"].itertuples()}
def rate(h): s=fb[fb["hotel_id"]==hid[h]]; return (s["is_canceled"].mean())
def tol(name,val,tgt,t,unit="pp"):
    ok=abs(val-tgt)<=t
    chk(name,ok,f"synthetic={val:.3f} target={tgt:.3f} tol=±{t}")
    return ok
tol("cancel rate Resort (±0.02)", rate("Resort"), TARGET["cancel_resort"],0.02)
tol("cancel rate City (±0.02)",   rate("City"),   TARGET["cancel_city"],  0.02)
tol("cancel rate overall (±0.02)",fb["is_canceled"].mean(),TARGET["cancel_overall"],0.02)
chk("City cancels more than Resort", rate("City")>rate("Resort"),
    f"city={rate('City'):.3f} > resort={rate('Resort'):.3f}")
# no-show
ns=lambda h:(fb[(fb.hotel_id==hid[h])&(fb.reservation_status=='No-Show')].shape[0]/fb[fb.hotel_id==hid[h]].shape[0])
tol("no-show Resort (±0.01)", ns("Resort"),TARGET["noshow_resort"],0.01)
tol("no-show City (±0.01)",   ns("City"),  TARGET["noshow_city"],0.01)
# lead time: canceled > arrived
lt_c=fb[fb.is_canceled==1]["lead_time"].mean(); lt_a=fb[fb.is_canceled==0]["lead_time"].mean()
chk("lead time higher for cancellations", lt_c>lt_a, f"canceled={lt_c:.0f} vs arrived={lt_a:.0f}")
# deposit effect: Non Refund cancels heavily
nr=fb[fb.deposit_type_code=="Non Refund"]
if len(nr): chk("Non-Refund cancel rate > 0.90", nr["is_canceled"].mean()>TARGET["nonrefund_cancel_min"],
                f"{nr['is_canceled'].mean():.3f}")
# walk rate within evidence band
wr=len(T["fact_walk_event"])/len(T["fact_daily_stay"])
chk("walk-night rate in 0.5-2.5%", TARGET["walk_rate_lo"]<=wr<=TARGET["walk_rate_hi"], f"{wr:.3%}")
# capacity sane (archetype 100-300)
chk("capacity in 100-300 rooms", T["dim_hotel"]["capacity_rooms"].between(100,300).all(),
    str(list(T["dim_hotel"]["capacity_rooms"])))
con.close()

# ---------- BOOKING PACE VALIDATION ----------

snap = T["fact_reservation_snapshot"]

allowed_offsets = {120, 90, 60, 30, 14, 7, 3, 1}

chk(
    "snapshot offsets are valid",
    snap["days_before_arrival"].isin(allowed_offsets).all()
)

chk(
    "snapshot on-books flag is binary",
    snap["is_on_books"].isin([0, 1]).all()
)

chk(
    "unique booking-snapshot offset",
    ~snap.duplicated(
        ["booking_id", "days_before_arrival"]
    ).any()
)

# Every snapshot must belong to an existing booking
chk(
    "all snapshot bookings exist",
    snap["booking_id"].isin(fb["booking_id"]).all()
)

# ---------- write report ----------
dfm=pd.DataFrame(rows); dfm.to_csv(os.path.join(REP,"validation_metrics.csv"),index=False)
n_pass=(dfm.status=="PASS").sum(); n_fail=(dfm.status=="FAIL").sum()
with open(os.path.join(REP,"quality_report.md"),"w") as f:
    f.write(f"# Synthetic Database — Validation Report\n\n")
    f.write(f"**{n_pass} passed, {n_fail} failed** of {len(dfm)} checks.\n\n")
    f.write("Validity labels: integrity = `verified_implementation`; fidelity vs real = "
            "`matched_external_aggregate_benchmarks`; capacity/walk/cost layers = "
            "`internally_validated_simulation` (real-world fidelity `not_yet_validated`).\n\n")
    f.write("| Check | Status | Detail |\n|---|---|---|\n")
    for r in rows: f.write(f"| {r['check']} | {r['status']} | {r['detail']} |\n")
print(dfm.to_string(index=False))
print(f"\n{n_pass} PASS / {n_fail} FAIL")
raise SystemExit(1 if n_fail else 0)
