#!/usr/bin/env python3
"""
Reproducible synthetic data generator for the Hotel Overbooking DSS.

Design: the booking layer is GENERATED but CALIBRATED to the real Antonio et al.
(2019) hotel-booking dataset (cancellation rates, lead-time effect, deposit/segment
effects, seasonality, ADR, no-show). The capacity / walk / cost / booking-pace
layers are GENERATED to fill gaps the real data does not contain.

Everything is driven by a master seed with one RNG stream per mechanism, so the
database regenerates identically. The real dataset is a CALIBRATION source only;
no real records are copied.
"""
import json, sqlite3, hashlib, sys, os, datetime as dt
import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))        # code/
HERE = os.path.dirname(CODE)                              # submission root
CFG  = os.path.join(CODE, "default.json")
OUT  = os.path.join(HERE, "data", "final")
ORA  = os.path.join(HERE, "oracle")
DB   = os.path.join(HERE, "data", "final", "hotel_dss.db")

MONTHS = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]

def season(m):
    return {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
            6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"}[m]

def pick(rng, mapping, n):
    keys = list(mapping.keys()); p = np.array(list(mapping.values()), float); p /= p.sum()
    return rng.choice(keys, size=n, p=p)

def calibrate_intercept(lin, target_rate, lo=-12, hi=12):
    """Bisection on an added intercept so mean(sigmoid(lin+b)) == target_rate."""
    for _ in range(60):
        mid = (lo+hi)/2
        if np.mean(1/(1+np.exp(-(lin+mid)))) < target_rate: lo = mid
        else: hi = mid
    return (lo+hi)/2

def gen_hotel(name, hc, cfg, seeds, start, end):
    n = hc["n_bookings"]; hid = hc["hotel_id"]
    r_time  = np.random.default_rng(seeds["time"])
    r_attr  = np.random.default_rng(seeds["attr"])
    r_price = np.random.default_rng(seeds["price"])
    r_cxl   = np.random.default_rng(seeds["cxl"])

    # ---- arrival dates via monthly seasonality ----
    all_days = pd.date_range(start, end, freq="D")
    mw = np.array([hc["month_arrival_prop"][MONTHS[d.month-1]] for d in all_days], float)
    day_w = mw / mw.sum()
    arrival = r_time.choice(all_days, size=n, p=day_w)
    arrival = pd.to_datetime(arrival)

    # ---- lead time (gamma, calibrated mean), booking date = arrival - lead ----
    shape = 1.15
    lead = r_time.gamma(shape, hc["lead_time_mean"]/shape, n).astype(int)
    lead = np.clip(lead, 0, 729)
    booking = arrival - pd.to_timedelta(lead, "D")

    # ---- stay nights ----
    wk = r_attr.poisson(hc["stay_week_mean"], n)
    we = r_attr.poisson(hc["stay_weekend_mean"], n)
    tot = wk + we
    bump = tot < 1; wk[bump] += 1; tot = wk + we            # ensure >= 1 night

    # ---- categorical attributes ----
    seg   = pick(r_attr, hc["segment_prop"], n)
    chan  = pick(r_attr, hc["channel_prop"], n)
    dep   = pick(r_attr, hc["deposit_prop"], n)
    cust  = pick(r_attr, hc["customer_prop"], n)
    meal  = pick(r_attr, hc["meal_prop"], n)
    rtp   = pick(r_attr, {"A":0.55,"B":0.07,"D":0.20,"E":0.10,"F":0.05,"G":0.03}, n)
    country = pick(r_attr, {"PRT":0.45,"GBR":0.12,"FRA":0.10,"ESP":0.08,"DEU":0.07,
                            "IRL":0.04,"ITA":0.03,"USA":0.03,"Other":0.08}, n)

    adults   = np.clip(r_attr.poisson(1.9, n), 1, 4)
    children = (r_attr.random(n) < 0.08) * r_attr.integers(1, 3, n)
    babies   = (r_attr.random(n) < 0.01) * 1
    repeated = (r_attr.random(n) < hc["repeated_guest_rate"]).astype(int)
    prev_cxl = (r_attr.random(n) < 0.06) * r_attr.integers(1, 4, n)
    special  = r_attr.poisson(hc["special_req_mean"], n)
    changes  = (r_attr.random(n) < 0.15) * r_attr.integers(1, 4, n)

    # ---- ADR: lognormal calibrated to mean, with summer uplift ----
    cv = hc["adr_cv"]; sigma = np.sqrt(np.log(1+cv**2)); mu = np.log(hc["adr_mean"]) - sigma**2/2
    adr = r_price.lognormal(mu, sigma, n)
    season_mult = np.array([1.25 if d.month in (7,8) else 1.12 if d.month in (6,9) else
                            0.85 if d.month in (12,1,2) else 1.0 for d in arrival])
    adr = np.round(adr * season_mult, 2)
    adr = np.clip(adr, 0, None)

    # ---- cancellation model (logit), coefficients chosen to reproduce real effects ----
    z = lambda x: (x - x.mean())/ (x.std()+1e-9)
    seg_eff = {"Online TA":0.55,"Offline TA/TO":-0.15,"Direct":-0.95,"Corporate":-0.75,
               "Groups":1.15,"Complementary":-0.6,"Aviation":-0.5}
    dep_eff = {"No Deposit":0.0,"Refundable":-0.4,"Non Refund":6.0}   # Non-Refund ~ always cancels
    lin = (1.05*z(lead.astype(float))
           + np.array([seg_eff.get(s,0) for s in seg])
           + np.array([dep_eff.get(d,0) for d in dep])
           + 0.8*prev_cxl - 0.9*repeated - 0.25*special + 0.1*z(adr))
    b0 = calibrate_intercept(lin, hc["target_cancel_rate"])
    p_true = 1/(1+np.exp(-(lin+b0)))
    is_canceled = (r_cxl.random(n) < p_true).astype(int)

    # ---- no-show (among non-canceled), calibrated to overall rate ----
    noshow_cond = hc["noshow_rate_overall"] / max(1e-9, (1 - is_canceled.mean()))
    is_noshow = ((is_canceled == 0) & (r_cxl.random(n) < noshow_cond)).astype(int)

    status = np.where(is_canceled == 1, "Canceled",
             np.where(is_noshow == 1, "No-Show", "Check-Out"))

    # ---- status date: cancel before arrival; checkout = arrival + nights; no-show = arrival ----
    frac = r_cxl.random(n)                       # fraction of lead-time elapsed before cancel
    days_before_cxl = np.ceil(frac * lead).astype(int)
    status_date = arrival.copy()
    cxl_mask = is_canceled == 1
    status_date = np.where(cxl_mask, arrival - pd.to_timedelta(days_before_cxl, "D"), status_date)
    co_mask = status == "Check-Out"
    status_date = np.where(co_mask, arrival + pd.to_timedelta(tot, "D"), status_date)
    status_date = pd.to_datetime(status_date)
    # effective date a booking leaves the books (cancel date, else arrival/checkin)
    leave_date = np.where(cxl_mask, status_date, arrival)
    leave_date = pd.to_datetime(leave_date)

    df = pd.DataFrame({
        "hotel_id": hid, "booking_date": booking, "arrival_date": arrival,
        "segment_code": seg, "channel_code": chan, "deposit_type_code": dep,
        "reserved_room_type": rtp, "lead_time": lead,
        "stays_week_nights": wk, "stays_weekend_nights": we, "total_nights": tot,
        "adults": adults, "children": children, "babies": babies, "adr": adr,
        "is_repeated_guest": repeated, "previous_cancellations": prev_cxl,
        "booking_changes": changes, "total_special_requests": special,
        "customer_type": cust, "meal": meal, "country": country,
        "is_canceled": is_canceled, "reservation_status": status,
        "status_date": status_date, "leave_date": leave_date, "p_true": np.round(p_true,5),
    })
    return df

def main(cfg_path=CFG, scale=None):
    cfg = json.load(open(cfg_path))
    start = pd.Timestamp(cfg["date_start"]); end = pd.Timestamp(cfg["date_end"])
    ms = np.random.SeedSequence(cfg["master_seed"])

    # ---- generate bookings per hotel ----
    frames = []
    for i,(name,hc) in enumerate(cfg["hotels"].items()):
        if scale == "small": hc = {**hc, "n_bookings": max(2000, hc["n_bookings"]//10)}
        child = ms.spawn(1)[0].spawn(5)
        seeds = {k:int(child[j].generate_state(1)[0]) for j,k in
                 enumerate(["time","attr","price","cxl","misc"])}
        f = gen_hotel(name, hc, cfg, seeds, start, end); frames.append(f)
    bk = pd.concat(frames, ignore_index=True)
    bk.insert(0, "booking_id", np.arange(1, len(bk)+1))

    # ---- date dimension spanning everything actually used ----
    dmin = min(bk["booking_date"].min(), start)
    dmax = max(bk["status_date"].max(), (bk["arrival_date"]+pd.to_timedelta(bk["total_nights"],"D")).max(), end)
    dates = pd.date_range(dmin.normalize(), dmax.normalize(), freq="D")
    dim_date = pd.DataFrame({
        "date_id":[int(d.strftime("%Y%m%d")) for d in dates],
        "full_date":[d.strftime("%Y-%m-%d") for d in dates],
        "year":dates.year, "month":dates.month, "month_name":[MONTHS[m-1] for m in dates.month],
        "week_number":dates.isocalendar().week.astype(int).values,
        "day_of_month":dates.day, "day_of_week":dates.dayofweek,
        "is_weekend":(dates.dayofweek>=5).astype(int), "season":[season(m) for m in dates.month],
    })
    did = lambda s: pd.to_datetime(s).dt.strftime("%Y%m%d").astype(int)

    # ---- assemble fact_booking with date FKs ----
    fb = pd.DataFrame({
        "booking_id":bk["booking_id"], "hotel_id":bk["hotel_id"],
        "booking_date_id":did(bk["booking_date"]), "arrival_date_id":did(bk["arrival_date"]),
        "segment_code":bk["segment_code"], "channel_code":bk["channel_code"],
        "deposit_type_code":bk["deposit_type_code"], "reserved_room_type":bk["reserved_room_type"],
        "lead_time":bk["lead_time"], "stays_week_nights":bk["stays_week_nights"],
        "stays_weekend_nights":bk["stays_weekend_nights"], "total_nights":bk["total_nights"],
        "adults":bk["adults"], "children":bk["children"], "babies":bk["babies"], "adr":bk["adr"],
        "is_repeated_guest":bk["is_repeated_guest"], "previous_cancellations":bk["previous_cancellations"],
        "booking_changes":bk["booking_changes"], "total_special_requests":bk["total_special_requests"],
        "customer_type":bk["customer_type"], "meal":bk["meal"], "country":bk["country"],
        "is_canceled":bk["is_canceled"], "reservation_status":bk["reservation_status"],
        "status_date_id":did(bk["status_date"]),
    })

    # ---- reservation snapshots (leakage-safe on-the-books state) ----
    offs = cfg["snapshot_offsets_days"]; snaps = []
    arr = bk["arrival_date"].values.astype("datetime64[D]")
    bkd = bk["booking_date"].values.astype("datetime64[D]")
    lvd = bk["leave_date"].values.astype("datetime64[D]")
    for off in offs:
        sdate = arr - np.timedelta64(off,"D")
        exists = (sdate >= bkd)                      # booking already made
        on_books = exists & (sdate < lvd)            # not yet cancelled / not yet arrived
        idx = np.where(exists)[0]
        if len(idx)==0: continue
        snaps.append(pd.DataFrame({
            "booking_id":bk["booking_id"].values[idx],
            "snapshot_date_id":pd.to_datetime(sdate[idx]).strftime("%Y%m%d").astype(int),
            "days_before_arrival":off, "is_on_books":on_books[idx].astype(int),
        }))
    fsnap = pd.concat(snaps, ignore_index=True)
    fsnap.insert(0, "snapshot_id", np.arange(1, len(fsnap)+1))

    # ---- nightly occupancy, capacity, walks, recommendations ----
    daily_rows=[]; walk_rows=[]; rec_rows=[]; hotels_meta=[]; cost_rows=[]
    arrived = bk[(bk["is_canceled"]==0) & (bk["reservation_status"]=="Check-Out")].copy()
    pct = cfg["capacity_percentile"]
    for name,hc in cfg["hotels"].items():
        hid=hc["hotel_id"]; sub=arrived[arrived["hotel_id"]==hid]
        # expand each stay across occupied nights
        occ=np.zeros(len(dates),dtype=int); arrv=np.zeros(len(dates),dtype=int)
        base=dates[0].to_datetimeindex() if hasattr(dates[0],'to_datetimeindex') else dates[0]
        d0=dates[0]
        a_idx=((sub["arrival_date"]-d0).dt.days).values
        nights=sub["total_nights"].values
        for s,ln in zip(a_idx,nights):
            if s<0: continue
            e=min(s+ln,len(occ)); occ[s:e]+=1
            if 0<=s<len(arrv): arrv[s]+=1
        capacity=int(np.ceil(np.quantile(occ[occ>0], pct))) if (occ>0).any() else 50
        hotels_meta.append((hid,f"{name} Hotel",name,capacity,pct))
        # cost params (generated; assumption-labelled)
        mean_adr=float(sub["adr"].mean())
        cost_rows.append((hid, cfg["contribution_margin_pct"],
                          round(cfg["contribution_margin_pct"]*mean_adr,2),
                          cfg["walk_cost_multiplier_low"], cfg["walk_cost_multiplier_high"],
                          "expert_assumption (margin to be sourced)", cfg["date_start"]))
        # daily + walks
        for j,d in enumerate(dates):
            if occ[j]==0 and arrv[j]==0: continue
            sdid=int(d.strftime("%Y%m%d"))
            daily_rows.append((hid,sdid,int(occ[j]),int(arrv[j]),capacity))
            if occ[j]>capacity:
                gw=int(occ[j]-capacity)
                m=np.random.default_rng(cfg["master_seed"]+sdid+hid).uniform(
                    cfg["walk_cost_multiplier_low"],cfg["walk_cost_multiplier_high"])
                walk_rows.append((hid,sdid,int(occ[j]),capacity,gw,round(mean_adr,2),
                                  round(gw*m*mean_adr,2)))
        # simple recommendation: authorize capacity / (1 - cancel_rate) rounded
        cr=hc["target_cancel_rate"]
        for (h,sdid,o,a,cap) in [r for r in daily_rows if r[0]==hid]:
            auth=int(round(cap/(1-cr))); k=max(0,auth-cap)
            rec_rows.append((hid,sdid,round(cap*cr,1),k,auth))

    dim_hotel=pd.DataFrame(hotels_meta,columns=["hotel_id","hotel_name","hotel_type","capacity_rooms","capacity_percentile"])
    dim_cost=pd.DataFrame(cost_rows,columns=["hotel_id","contribution_margin_pct","spoilage_cost_per_room","walk_cost_mult_low","walk_cost_mult_high","provenance","effective_date"])
    dim_cost.insert(0,"param_id",np.arange(1,len(dim_cost)+1))
    fdaily=pd.DataFrame(daily_rows,columns=["hotel_id","stay_date_id","occupied_rooms","arrivals","capacity_rooms"])
    fwalk=pd.DataFrame(walk_rows,columns=["hotel_id","stay_date_id","occupied_rooms","capacity_rooms","guests_walked","avg_adr","relocation_cost"])
    fwalk.insert(0,"walk_event_id",np.arange(1,len(fwalk)+1))
    frec=pd.DataFrame(rec_rows,columns=["hotel_id","stay_date_id","expected_cancellations","recommended_buffer_k","authorized_rooms"])
    frec.insert(0,"rec_id",np.arange(1,len(frec)+1))

    # ---- reference tables ----
    seg_all=sorted(set(fb["segment_code"]))
    ref_seg=pd.DataFrame({"segment_code":seg_all,"description":[s+" market segment" for s in seg_all]})
    chan_all=sorted(set(fb["channel_code"]))
    ref_chan=pd.DataFrame({"channel_code":chan_all,"description":[c+" distribution channel" for c in chan_all]})
    dep_all=sorted(set(fb["deposit_type_code"]))
    notes={"No Deposit":"no prepayment","Non Refund":"prepaid, non-refundable (very high cancel risk)","Refundable":"refundable deposit"}
    ref_dep=pd.DataFrame({"deposit_type_code":dep_all,"cancel_risk_note":[notes.get(d,"") for d in dep_all]})

    oracle=pd.DataFrame({"booking_id":bk["booking_id"],"true_cancel_prob":bk["p_true"]})

    # ---- write CSVs ----
    os.makedirs(OUT,exist_ok=True); os.makedirs(ORA,exist_ok=True)
    tables={"dim_hotel":dim_hotel,"dim_date":dim_date,"ref_market_segment":ref_seg,
            "ref_distribution_channel":ref_chan,"ref_deposit_type":ref_dep,
            "dim_cost_parameter":dim_cost,"fact_booking":fb,"fact_reservation_snapshot":fsnap,
            "fact_daily_stay":fdaily,"fact_walk_event":fwalk,"fact_overbooking_recommendation":frec}
    manifest={"generated_at":dt.datetime.now().isoformat(timespec="seconds"),
              "master_seed":cfg["master_seed"],"python":sys.version.split()[0],
              "numpy":np.__version__,"pandas":pd.__version__,"tables":{}}
    for nm,df in tables.items():
        path=os.path.join(OUT,nm+".csv"); df.to_csv(path,index=False)
        h=hashlib.md5(open(path,"rb").read()).hexdigest()
        manifest["tables"][nm]={"rows":len(df),"md5":h}
    oracle.to_csv(os.path.join(ORA,"oracle_booking_truth.csv"),index=False)
    json.dump({"master_seed":cfg["master_seed"],"cancellation_coeffs":"see generate.py gen_hotel()",
               "capacity_percentile":pct}, open(os.path.join(ORA,"truth_parameters.json"),"w"),indent=2)

    # ---- load SQLite from schema ----
    if os.path.exists(DB): os.remove(DB)
    con=sqlite3.connect(DB); con.executescript(open(os.path.join(CODE,"schema.sql")).read())
    for nm,df in tables.items(): df.to_sql(nm,con,if_exists="append",index=False)
    con.commit(); con.close()

    json.dump(manifest, open(os.path.join(HERE,"run_manifest.json"),"w"),indent=2)
    print("rows:",{k:v["rows"] for k,v in manifest["tables"].items()})
    print("DB:",DB)

if __name__=="__main__":
    main(scale=(sys.argv[1] if len(sys.argv)>1 else None))
