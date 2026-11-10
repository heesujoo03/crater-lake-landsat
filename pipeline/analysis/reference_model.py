# -*- coding: utf-8 -*-
"""Checks on the relative-temperature model (synthetic reference + ERA5-Land adjustment).

Pipeline modules and existing records are not modified. The module re-implements
the fitting path of `s3b_synthetic_control._core` + `s9b_event_stats.frame` with an
optional extra exclusion mask on the fitting sample, checks that without extra
exclusion the re-implementation reproduces the primary series, and then computes:

  1. implementation facts (donor list, weights, fit sample)       -> donor_weights.csv
  2. weather-model coefficients                                    -> weather_coefficients.csv
  3. fit-sample leakage into eruption windows + corrected designs  -> fit_leakage.csv,
     corrected_contrasts.csv
  4. simple seasonal baseline and other reference definitions      -> baseline_comparison.csv
  5. non-stationarity: trends and split-half stability             -> trends_*.csv, split_half.csv
  6. residual seasonality / seasonal amplitude                     -> seasonal_*.csv
  7. Taal caldera-lake reference around 2020                       -> taal_reference_*.csv

Outputs: results/11_analysis/reference_model/.

    python -m pipeline.analysis.reference_model
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from pipeline import s3b_synthetic_control as SC
from pipeline import s9b_event_stats as B9
from pipeline import s10_post_eruption as S10
from pipeline.common import baseline as B
from pipeline.config import settings as S
from pipeline.config import thermal_episodes as TE
from pipeline.config.sites import get_site

OUT = S.RECORDS_DIR / "11_analysis" / "reference_model"
OUT.mkdir(parents=True, exist_ok=True)
SITES3 = ("taal", "santa_ana", "ruapehu")
F_MIN = 0.70
BUF = 180
WINS = (180, 365)
EXCL_W = 365


def donor_label(site, mode, k):
    if k == "control_lake":
        if mode == "bg_ref":
            return "surrounding ring (annulus 1.6-3.0 lake radii; enters as T_control in bg_ref mode)", "ring"
        return "caldera lake, open-water point (Taal Lake, 7.5 km, r = 1500 m)", "caldera_lake"
    if k == "background_ring":
        return "surrounding ring (annulus 1.6-3.0 lake radii)", "ring"
    if k == "seasonal_cycle":
        return "seasonal donor: lake's own quiet-period annual + semi-annual harmonic fit (K)", "seasonal"
    c = k.split()[-1]
    return f"nearby water body {c} (1500 m circle, Landsat ST mean)", f"water_{c}"


def donor_meta(site, k):
    if not k.startswith("nearby water body"):
        return {}
    c = k.split()[-1]
    d = pd.read_csv(S.INTERIM_DIR / f"{site}_donor_lst.csv")
    g = d[d["donor"] == c].iloc[0]
    return dict(lat=g["lat"], lon=g["lon"], dist_km=g["dist_km"], area_km2=g["area_km2"],
                elev_diff_m=g["elev_diff_m"], jrc_occurrence=g["jrc_occurrence"])


def raw_donor(site, k, P):
    c = k.split()[-1]
    d = pd.read_csv(S.INTERIM_DIR / f"{site}_donor_lst.csv")
    d = d[d["donor"] == c]
    d["day"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    return d.groupby("day")["T"].median().reindex(P["date"].to_numpy()).to_numpy(float)


# ---------------------------------------------------------------------------
#  data preparation (identical to s3b._core / s9b.frame up to the fits)
# ---------------------------------------------------------------------------
def prep(site_key, f_min=F_MIN, buf=BUF):
    site = get_site(site_key)
    d = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_deltaT.csv")
    d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
    d = d[d["qc_flag"].eq("ok")]
    if f_min is not None:
        d = d[pd.to_numeric(d["f_est_act"], errors="coerce") >= float(f_min)]
    d = d.sort_values("t").reset_index(drop=True)
    ke = list(site.known_events or [])
    brep = {}
    quiet = B.in_baseline(d["t"], ke, buf, qc_flag=d.get("qc_flag"), min_n=60,
                          report=brep).to_numpy()
    cls = B.classify(d["t"], ke, buf).to_numpy()
    y = pd.to_numeric(d["T_active_K"], errors="coerce").to_numpy(float)
    ctl = pd.to_numeric(d.get("T_control_K"), errors="coerce").to_numpy(float)
    bg = pd.to_numeric(d.get("bg_active_K"), errors="coerce").to_numpy(float)
    dow = SC._doy_terms(d["t"])
    extra = SC._donor_pool(site_key, d["t"])
    date = pd.to_datetime(d["t"], utc=True).dt.tz_localize(None).dt.normalize()
    ds = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_dataset.csv")
    ds["date"] = pd.to_datetime(ds["date"], errors="coerce").dt.normalize()
    keep = ["date"] + [c for c in B9.MET_COLS if c in ds.columns]
    met = pd.DataFrame({"date": date}).merge(ds[keep].drop_duplicates("date"),
                                             on="date", how="left")
    absent = TE.absent_mask(site_key, date)
    dnum = date.to_numpy("datetime64[D]").astype("int64")
    return dict(site=site_key, d=d, ke=ke, quiet=quiet, cls=cls, brep=brep, y=y,
                ctl=ctl, bg=bg, dow=dow, extra=extra, date=date, met=met,
                absent=absent, dnum=dnum, mode=str(d["mode"].iloc[0]))


def fit(P, excl=None, drop_donors=(), return_detail=False):
    """Synthetic reference + weather OLS. `excl` (bool, len = rows of P['d'])
    removes rows from BOTH fitting samples. Returns a frame aligned to the
    non-absent rows (as s9b.frame does)."""
    y, ctl, bg, dow, quiet = P["y"], P["ctl"], P["bg"], P["dow"], P["quiet"]
    n = len(y)
    excl = np.zeros(n, bool) if excl is None else np.asarray(excl, bool)
    ok = np.isfinite(y)
    fmask = quiet & ok & np.isfinite(ctl) & np.isfinite(bg) & ~excl
    Xc = np.c_[np.ones(n), dow]
    bc, *_ = np.linalg.lstsq(Xc[fmask], y[fmask], rcond=None)
    clim = Xc @ bc
    donors = [ctl, bg, clim]
    names = ["control_lake", "background_ring", "seasonal_cycle"]
    for nm, ser in P["extra"].items():
        donors.append(ser)
        names.append(nm)
    D = np.column_stack(donors)
    keepi, seen = [], []
    for i in range(D.shape[1]):
        if names[i] in drop_donors:
            continue
        col = D[:, i]
        if not np.isfinite(col[fmask]).all():
            continue
        if any(np.allclose(col[fmask], D[:, j][fmask], atol=1e-9) for j in seen):
            continue
        seen.append(i)
        keepi.append(i)
    D, names = D[:, keepi], [names[i] for i in keepi]
    w = SC._simplex_fit(y[fmask], D[fmask])
    ref = D @ w
    dT_syn = y - ref

    # --- s9b.frame part (after dropping lake-absent rows) ---
    keep = ~P["absent"]
    df = pd.DataFrame({"date": P["date"][keep].to_numpy(), "T_active_K": y[keep],
                       "dT_syn": dT_syn[keep], "ref": ref[keep],
                       "dT_simple": (y - ctl)[keep], "dT_ring": (y - bg)[keep],
                       "dT_clim": (y - clim)[keep], "clim": clim[keep],
                       "T_control_K": ctl[keep], "bg_active_K": bg[keep],
                       "_quiet": quiet[keep], "_excl": excl[keep],
                       "_fit_syn": fmask[keep]})
    met = P["met"][keep].reset_index(drop=True)
    cols = [c for c in B9.MET_COLS if c in met.columns
            and pd.to_numeric(met[c], errors="coerce").notna().mean() > 0.7]
    X = met[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    yy = df["dT_syn"].to_numpy(float)
    f2 = df["_quiet"].to_numpy(bool) & np.isfinite(yy) & ~df["_excl"].to_numpy(bool)
    for j in range(X.shape[1]):
        med = np.nanmedian(X[f2, j]) if f2.any() else np.nanmedian(X[:, j])
        X[~np.isfinite(X[:, j]), j] = med if np.isfinite(med) else 0.0
    Xd = np.c_[np.ones(len(yy)), X]
    good = f2 & np.isfinite(Xd).all(axis=1)
    beta, *_ = np.linalg.lstsq(Xd[good], yy[good], rcond=None)
    ghat = Xd @ beta
    df["g_hat"] = ghat
    df["dT_syn_weather"] = yy - ghat
    df["_fit_w"] = good
    for c in cols:
        df[c] = met[c].to_numpy()
    df["dnum"] = df["date"].to_numpy("datetime64[D]").astype("int64")
    if not return_detail:
        return df
    return df, dict(weights=dict(zip(names, w)), bc=bc, beta=beta, cols=cols,
                    Xd=Xd, good=good, yy=yy, n_fit_syn=int(fmask.sum()),
                    fmask=fmask)


# ---------------------------------------------------------------------------
#  contrasts (the s10 machinery, re-used verbatim)
# ---------------------------------------------------------------------------
def events_in_record(dnum, ke, w):
    t_first, t_last = int(dnum.min()), int(dnum.max())
    eps = []
    for o in S10.onsets(ke, w):
        t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
        if t0n - w < t_first or t0n > t_last:
            continue
        eps.append(dict(t0=o["t0"], t0n=t0n, n_merged=o["n_merged"]))
    return eps


def contrasts(dnum, y, ke, w, cands=None):
    eps = events_in_record(dnum, ke, w)
    if cands is None:
        cands = S10.placebo_candidates(dnum, y, [e["t0n"] for e in eps], w)
    rows = []
    for e in eps:
        d, npre, npost = S10.before_after(dnum, y, e["t0n"], w)
        ok = npre >= S10.MIN_SCENES and npost >= S10.MIN_SCENES
        r = dict(onset=str(e["t0"].date()), window_d=w, n_merged=e["n_merged"],
                 n_pre=npre, n_post=npost, delta=(d if ok else np.nan),
                 p_rank=np.nan, n_years=0, pl_min=np.nan, pl_max=np.nan,
                 pl_q025=np.nan, pl_q975=np.nan, max_exceed=False)
        if ok:
            pool, pool_yr, ny = S10._matched_pool(cands, e["t0"].dayofyear, npre, npost)
            r["n_years"] = ny
            r["p_rank"] = S10._p_two_sided(pool_yr, d)
            if ny:
                r["pl_min"], r["pl_max"] = float(np.min(pool_yr)), float(np.max(pool_yr))
            if len(pool) >= 10:
                r["pl_q025"], r["pl_q975"] = np.percentile(pool, [2.5, 97.5])
            r["max_exceed"] = bool(ny >= S10.MIN_YEARS and
                                   abs(r["p_rank"] - 1 / (ny + 1)) < 2e-4)
        rows.append(r)
    return rows


def lwo_candidates(P, df_c1, excl_c1, w, col="dT_syn_weather"):
    """Leave-window-out placebo values: every placebo date c is evaluated on a
    series refitted with [c-W, c+W] removed from the fitting sample (in addition
    to the actual-event exclusion excl_c1)."""
    dnum = df_c1["dnum"].to_numpy()
    y1 = df_c1[col].to_numpy(float)
    eps = events_in_record(dnum, P["ke"], w)
    real = [e["t0n"] for e in eps]
    lo, hi = int(dnum.min()) + w, int(dnum.max()) - w
    cand = np.arange(lo, hi + 1, 3)
    far = np.ones(len(cand), bool)
    for t in real:
        far &= np.abs(cand - t) > 2 * w
    cand = cand[far]
    ts = pd.to_datetime(cand.astype("datetime64[D]"))
    doy = ts.dayofyear.to_numpy()
    near = np.zeros(len(cand), bool)
    for e in eps:
        dd = np.abs(doy - e["t0"].dayofyear)
        near |= np.minimum(dd, 365 - dd) <= S10.SEASON_HALF_WIDTH_D
    vals = np.full(len(cand), np.nan)
    npre = np.zeros(len(cand), int)
    npost = np.zeros(len(cand), int)
    dn_all = P["dnum"]
    fit_rows_c1 = df_c1["_fit_w"].to_numpy(bool)
    n_refit = 0
    for i, c in enumerate(cand):
        d0, a, b = S10.before_after(dnum, y1, int(c), w)
        npre[i], npost[i] = a, b
        if not (a >= S10.MIN_SCENES and b >= S10.MIN_SCENES) or not near[i]:
            continue
        win_fit = P["quiet"] & ~excl_c1 & (np.abs(dn_all - c) <= w)
        if not win_fit.any():
            vals[i] = d0                      # window not in fit sample: C1 value
            continue
        ex = excl_c1 | (np.abs(dn_all - c) <= w)
        dfi = fit(P, ex)
        vals[i], _, _ = S10.before_after(dfi["dnum"].to_numpy(),
                                         dfi[col].to_numpy(float), int(c), w)
        n_refit += 1
    okm = np.isfinite(vals)
    return (vals[okm], doy[okm], ts.year.to_numpy()[okm], npre[okm], npost[okm]), n_refit


def event_excl(P, W=EXCL_W):
    dn = P["dnum"]
    m = np.zeros(len(dn), bool)
    for e in P["ke"]:
        s = int(np.datetime64(pd.Timestamp(e.get("start") or e["date"]).date(), "D").astype("int64"))
        en = int(np.datetime64(pd.Timestamp(e.get("end") or e.get("start") or e["date"]).date(), "D").astype("int64"))
        m |= (dn >= s - W) & (dn <= en + W)
    return m


# ---------------------------------------------------------------------------
def slope_decade(t_year, v, n_boot=2000, seed=1):
    ok = np.isfinite(t_year) & np.isfinite(v)
    t, v = t_year[ok], v[ok]
    if len(v) < 8:
        return np.nan, np.nan, np.nan, int(len(v))
    b = np.polyfit(t, v, 1)[0] * 10
    rng = np.random.default_rng(seed)
    yrs = np.floor(t).astype(int)
    uy = np.unique(yrs)
    bs = []
    for _ in range(n_boot):                    # resample whole years (block bootstrap)
        pick = rng.choice(uy, len(uy), replace=True)
        idx = np.concatenate([np.flatnonzero(yrs == u) for u in pick])
        if np.ptp(t[idx]) < 1:
            continue
        bs.append(np.polyfit(t[idx], v[idx], 1)[0] * 10)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b), float(lo), float(hi), int(len(v))


def decyear(dates):
    dt = pd.DatetimeIndex(pd.to_datetime(dates))
    return np.asarray(dt.year + (dt.dayofyear - 0.5) / 365.25, float)


def main():
    pub = pd.read_csv(S.RECORDS_DIR / "10_post_eruption" / "per_eruption_delta_paper.csv",
                      encoding="utf-8-sig")
    rows_w, rows_b, rows_leak, rows_c, rows_base, rows_tr, rows_era, rows_split = ([] for _ in range(8))
    rows_mon, rows_amp, facts = [], [], {}
    Ps = {}
    for site in SITES3:
        P = prep(site)
        Ps[site] = P
        df, det = fit(P, return_detail=True)
        # ---- 0. check reproduction of the primary series
        ref = B9.frame(site, P["ke"], BUF, f_min=F_MIN)
        same = np.allclose(ref["dT_syn_weather"].to_numpy(float),
                           df["dT_syn_weather"].to_numpy(float), equal_nan=True, atol=1e-9)
        dn, yw = df["dnum"].to_numpy(), df["dT_syn_weather"].to_numpy(float)
        fr = df.loc[df["_fit_w"]]
        fs = det["fmask"]
        facts[site] = dict(
            reproduces_s9b_frame=bool(same), mode=P["mode"],
            n_scenes_after_screen=int(len(P["d"])), n_after_absent=int(len(df)),
            baseline_rule=P["brep"].get("baseline_rule"),
            n_quiet_pre_first_event=P["brep"].get("n_strict"),
            n_fit_synthetic=int(fs.sum()),
            fit_synthetic_period=[str(P["date"][fs].min().date()), str(P["date"][fs].max().date())],
            n_fit_weather=int(df["_fit_w"].sum()),
            fit_weather_period=[str(fr["date"].min().date()), str(fr["date"].max().date())],
            harmonic_coef_K=[round(float(v), 4) for v in det["bc"]],
            weather_cols=det["cols"])
        # ---- 1. donors
        for k, v in det["weights"].items():
            lab, key_en = donor_label(site, P["mode"], k)
            meta = donor_meta(site, k)
            if k in P["extra"]:
                cov = float(np.isfinite(raw_donor(site, k, P)).mean())
                covq = float(np.isfinite(raw_donor(site, k, P)[fs]).mean())
            else:
                cov = float(np.isfinite({"control_lake": P["ctl"], "background_ring": P["bg"]}.get(k, P["y"])).mean())
                covq = 1.0
            rows_w.append(dict(site=site, donor_key=key_en, donor=lab, weight=round(float(v), 4),
                               observed_frac_all=round(cov, 3), observed_frac_fit=round(covq, 3),
                               **meta,
                               n_fit=int(fs.sum()),
                               fit_start=facts[site]["fit_synthetic_period"][0],
                               fit_end=facts[site]["fit_synthetic_period"][1]))
        # all candidate donors, including ones dropped
        cand_all = ["control_lake", "background_ring", "seasonal_cycle"] + list(P["extra"].keys())
        facts[site]["candidate_donors"] = [donor_label(site, P["mode"], k)[1] for k in cand_all]
        facts[site]["retained_donors"] = [donor_label(site, P["mode"], k)[1] for k in det["weights"]]
        facts[site]["weights"] = {donor_label(site, P["mode"], k)[1]: round(float(v), 4) for k, v in det["weights"].items()}
        # ---- 2. weather coefficients
        Xd, good, yy = det["Xd"], det["good"], det["yy"]
        Xg, yg = Xd[good], yy[good]
        res = yg - Xg @ det["beta"]
        dof = len(yg) - Xg.shape[1]
        s2 = float(res @ res / dof)
        cov = s2 * np.linalg.pinv(Xg.T @ Xg)
        r2 = 1 - float(res @ res) / float(((yg - yg.mean()) ** 2).sum())
        for j, nm in enumerate(["intercept"] + det["cols"]):
            sdx = float(np.std(Xg[:, j], ddof=1)) if j else np.nan
            rows_b.append(dict(site=site, term=nm, coef=det["beta"][j],
                               se=float(np.sqrt(cov[j, j])),
                               t=det["beta"][j] / np.sqrt(cov[j, j]),
                               fit_mean=float(Xg[:, j].mean()) if j else np.nan,
                               fit_sd=sdx,
                               K_per_1sd=(det["beta"][j] * sdx if j else np.nan),
                               n_fit=int(good.sum()), R2=r2,
                               resid_sd_K=float(np.sqrt(s2)),
                               sd_ghat_fit_K=float(np.std(Xg @ det["beta"], ddof=1))))
        # ---- 3a. leakage count
        fitw = df["_fit_w"].to_numpy(bool)
        fits = df["_fit_syn"].to_numpy(bool)
        for w in WINS:
            for e in events_in_record(dn, P["ke"], w):
                t0 = e["t0n"]
                pre = (dn >= t0 - w) & (dn <= t0 - 1) & np.isfinite(yw)
                post = (dn >= t0) & (dn <= t0 + w) & np.isfinite(yw)
                rows_leak.append(dict(site=site, onset=str(e["t0"].date()), window_d=w,
                                      n_pre=int(pre.sum()), n_pre_in_fit=int((pre & fitw).sum()),
                                      n_pre_in_fit_syn=int((pre & fits).sum()),
                                      n_post=int(post.sum()), n_post_in_fit=int((post & fitw).sum()),
                                      pre_fit_dates="|".join(str(x.date()) for x in df["date"][pre & fitw]),
                                      post_fit_dates="|".join(str(x.date()) for x in df["date"][post & fitw])))
        # ---- 3b. designs
        ex1 = event_excl(P, EXCL_W)
        df1 = fit(P, ex1)
        facts[site]["n_fit_weather_C1"] = int(df1["_fit_w"].sum())
        facts[site]["n_fit_syn_C1"] = int(df1["_fit_syn"].sum())
        _, det1 = fit(P, ex1, return_detail=True)
        facts[site]["weights_C1"] = {donor_label(site, P["mode"], k)[1]: round(float(v), 3) for k, v in det1["weights"].items()}
        for w in WINS:
            for design, dfx in (("published_repro", df), ("C1_excl_actual_365", df1)):
                for r in contrasts(dfx["dnum"].to_numpy(), dfx["dT_syn_weather"].to_numpy(float), P["ke"], w):
                    rows_c.append(dict(site=site, design=design, **r))
            cands, nref = lwo_candidates(P, df1, ex1, w)
            for r in contrasts(df1["dnum"].to_numpy(), df1["dT_syn_weather"].to_numpy(float),
                               P["ke"], w, cands=cands):
                rows_c.append(dict(site=site, design="C2_leave_window_out", n_refits=nref, **r))
            print(f"  {site} W={w}: LWO refits {nref}", flush=True)
        # ---- 4. baseline comparison (primary fit, and C1 fit)
        for tag, dfx in (("published_fit", df), ("C1_fit", df1)):
            ests = dict(dT_syn_weather=dfx["dT_syn_weather"], dT_syn=dfx["dT_syn"],
                        seasonal_baseline=dfx["dT_clim"], ring_only=dfx["dT_ring"],
                        lake_minus_control=dfx["dT_simple"], T_lake=dfx["T_active_K"])
            # ring-only + weather: OLS of (lake - ring) on the same ERA5 set, quiet only
            cols = det["cols"]
            X = dfx[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            f2 = dfx["_fit_w"].to_numpy(bool)
            for j in range(X.shape[1]):
                X[~np.isfinite(X[:, j]), j] = np.nanmedian(X[f2, j])
            Xd2 = np.c_[np.ones(len(X)), X]
            yr_ = dfx["dT_ring"].to_numpy(float)
            g2 = f2 & np.isfinite(yr_)
            bb, *_ = np.linalg.lstsq(Xd2[g2], yr_[g2], rcond=None)
            ests["ring_only_weather"] = pd.Series(yr_ - Xd2 @ bb)
            # month-matched: remove the quiet (fit-sample) calendar-month median of the main estimand
            mon = pd.to_datetime(dfx["date"]).dt.month.to_numpy()
            ym = dfx["dT_syn_weather"].to_numpy(float)
            mm = pd.Series(ym[f2]).groupby(mon[f2]).median()
            ests["month_matched"] = pd.Series(ym - pd.Series(mon).map(mm).fillna(0).to_numpy())
            # seasonal baseline with weather: lake minus own climatology, then ERA5 OLS
            yc = dfx["dT_clim"].to_numpy(float)
            g3 = f2 & np.isfinite(yc)
            b3, *_ = np.linalg.lstsq(Xd2[g3], yc[g3], rcond=None)
            ests["seasonal_baseline_weather"] = pd.Series(yc - Xd2 @ b3)
            for nm, ser in ests.items():
                v = pd.to_numeric(ser, errors="coerce").to_numpy(float)
                qsd = float(np.nanstd(v[f2], ddof=1))
                for w in WINS:
                    for r in contrasts(dfx["dnum"].to_numpy(), v, P["ke"], w):
                        rows_base.append(dict(site=site, fit=tag, estimator=nm,
                                              quiet_fit_sd_K=qsd, **r))
        # ---- 5. non-stationarity
        ty = decyear(df["date"])
        for nm in ("dT_syn_weather", "dT_syn", "g_hat", "ref", "T_active_K", "dT_clim"):
            v = df[nm].to_numpy(float)
            for sub, m in (("weather_fit_sample", fitw),):
                b, lo, hi, n = slope_decade(ty[m], v[m])
                rows_tr.append(dict(site=site, series=nm, sample=sub, slope_K_per_decade=b,
                                    ci_lo=lo, ci_hi=hi, n=n,
                                    span=f"{df['date'][m].min().date()}..{df['date'][m].max().date()}"))
        b, lo, hi, n = slope_decade(ty, df["g_hat"].to_numpy(float))
        rows_tr.append(dict(site=site, series="g_hat", sample="all_scenes", slope_K_per_decade=b,
                            ci_lo=lo, ci_hi=hi, n=n, span=f"{df['date'].min().date()}..{df['date'].max().date()}"))
        for c in det["cols"]:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            b, lo, hi, n = slope_decade(ty, v)
            bf, lof, hif, nf = slope_decade(ty[fitw], v[fitw])
            beta_c = float(det["beta"][1 + det["cols"].index(c)])
            rows_era.append(dict(site=site, predictor=c, slope_per_decade_all=b, ci_lo_all=lo,
                                 ci_hi_all=hi, n_all=n, slope_per_decade_fit=bf, ci_lo_fit=lof,
                                 ci_hi_fit=hif, coef=beta_c,
                                 implied_ghat_K_per_decade_all=beta_c * b))
        # split halves of the fit sample (both fits restricted to one half)
        fdates = P["date"][det["fmask"]]
        tmid = fdates.iloc[len(fdates) // 2]
        facts[site]["split_date"] = str(tmid.date())
        for half, exh in (("early", P["date"].to_numpy() >= np.datetime64(tmid)),
                          ("late", P["date"].to_numpy() < np.datetime64(tmid))):
            try:
                dfh, deth = fit(P, exh | ex1, return_detail=True)
            except Exception as exc:                              # noqa: BLE001
                rows_split.append(dict(site=site, half=half, error=str(exc)))
                continue
            rec = dict(site=site, half=half, n_fit_syn=deth["n_fit_syn"],
                       n_fit_weather=int(deth["good"].sum()),
                       weights=json.dumps({donor_label(site, P["mode"], k)[1]: round(float(v), 3) for k, v in deth["weights"].items()},
                                          ensure_ascii=False))
            for j, nm in enumerate(["intercept"] + deth["cols"]):
                rec[f"b_{nm}"] = float(deth["beta"][j])
            for w in WINS:
                for r in contrasts(dfh["dnum"].to_numpy(), dfh["dT_syn_weather"].to_numpy(float), P["ke"], w):
                    rec[f"delta_{r['onset']}_{w}"] = r["delta"]
            rows_split.append(rec)
        # ---- 6. residual seasonality by month in the fit sample
        mon = pd.to_datetime(df["date"]).dt.month.to_numpy()
        for nm in ("dT_syn_weather", "dT_syn", "dT_ring"):
            v = df[nm].to_numpy(float)
            for m_ in range(1, 13):
                sel = fitw & (mon == m_) & np.isfinite(v)
                rows_mon.append(dict(site=site, series=nm, month=m_, n=int(sel.sum()),
                                     median_K=float(np.median(v[sel])) if sel.any() else np.nan,
                                     iqr_K=float(np.subtract(*np.percentile(v[sel], [75, 25]))) if sel.sum() > 2 else np.nan))
        # per-year annual amplitude (fit sample, lake temperature, ring and ERA5 Ta)
        yrs = pd.to_datetime(df["date"]).dt.year.to_numpy()
        doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy(float)
        for nm in ("T_active_K", "bg_active_K", "Ta_K", "dT_syn_weather"):
            v = pd.to_numeric(df[nm], errors="coerce").to_numpy(float)
            for u in np.unique(yrs[fitw]):
                sel = fitw & (yrs == u) & np.isfinite(v)
                if sel.sum() < 6 or len(np.unique(mon[sel])) < 4:
                    continue
                A = np.c_[np.ones(sel.sum()), np.sin(2 * np.pi * doy[sel] / 365.25),
                          np.cos(2 * np.pi * doy[sel] / 365.25)]
                c, *_ = np.linalg.lstsq(A, v[sel], rcond=None)
                rows_amp.append(dict(site=site, series=nm, year=int(u), n=int(sel.sum()),
                                     amp_K=float(np.hypot(c[1], c[2])),
                                     phase_doy=float((np.arctan2(c[1], c[2]) / (2 * np.pi) * 365.25) % 365.25),
                                     mean_K=float(c[0])))
        # store for Taal-specific analysis
        P["df"], P["det"] = df, det

    # ---------------- 7. Taal caldera-lake reference ------------------------
    P = Ps["taal"]
    df, det = P["df"], P["det"]
    rows_tr7, rows_7 = [], []
    ctl = df["T_control_K"].to_numpy(float)
    fitw = df["_fit_w"].to_numpy(bool)
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy(float)
    H = np.c_[np.ones(len(df)), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
              np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]
    for nm, v in (("caldera_lake_T", ctl), ("ring_T", df["bg_active_K"].to_numpy(float)),
                  ("ERA5_Ta", pd.to_numeric(df["Ta_K"], errors="coerce").to_numpy(float)),
                  ("caldera_lake_minus_ERA5_Ta", ctl - pd.to_numeric(df["Ta_K"], errors="coerce").to_numpy(float))):
        g = fitw & np.isfinite(v)
        c, *_ = np.linalg.lstsq(H[g], v[g], rcond=None)
        an = v - H @ c
        qsd = float(np.std(an[g], ddof=1))
        dn = df["dnum"].to_numpy()
        for w in WINS:
            for r in contrasts(dn, an, P["ke"], w):
                if r["onset"] != "2020-01-12":
                    continue
                rows_7.append(dict(series=nm + "_anomaly", quiet_sd_K=qsd, **r))
        # scene list 2018-07 .. 2021-06
        m = (df["date"] >= "2018-07-01") & (df["date"] <= "2021-06-30")
        for dd_, vv, aa in zip(df["date"][m], v[m], an[m]):
            rows_tr7.append(dict(series=nm, date=str(dd_.date()), value_K=vv, anomaly_K=aa))
    # extra donors around 2020 (raw donor series, anomalies vs own quiet harmonic)
    for k, ser in P["extra"].items():
        v = ser[~P["absent"]]
        g = fitw & np.isfinite(v)
        c, *_ = np.linalg.lstsq(H[g], v[g], rcond=None)
        an = v - H @ c
        for r in contrasts(df["dnum"].to_numpy(), an, P["ke"], 365):
            if r["onset"] == "2020-01-12":
                rows_7.append(dict(series=f"donor_{k.split()[-1]}_anomaly(filled)",
                                   quiet_sd_K=float(np.std(an[g], ddof=1)), **r))
    # refits without the caldera-lake donor (and without all water donors)
    water = ["control_lake"] + list(P["extra"].keys())
    for tag, drop in (("no_caldera_lake_donor", ("control_lake",)),
                      ("no_Taal_Lake_donors(caldera+c01)", ("control_lake", "nearby water body c01")),
                      ("ring_and_seasonal_only", tuple(water))):
        for dtag, ex in (("published_fit", None), ("C1_fit", event_excl(P))):
            dfx, detx = fit(P, ex, drop_donors=drop, return_detail=True)
            wts = json.dumps({donor_label("taal", "paired", k)[1]: round(float(v), 3)
                              for k, v in detx["weights"].items()}, ensure_ascii=False)
            for w in WINS:
                for r in contrasts(dfx["dnum"].to_numpy(), dfx["dT_syn_weather"].to_numpy(float), P["ke"], w):
                    rows_7.append(dict(series=f"dT_syn_weather[{tag},{dtag}]", weights=wts,
                                       quiet_sd_K=float(np.std(dfx["dT_syn_weather"][dfx["_fit_w"]], ddof=1)), **r))

    # ---------------- write ----------------
    def w_(rows, name):
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    w_(rows_w, "donor_weights.csv")
    w_(rows_b, "weather_coefficients.csv")
    w_(rows_leak, "fit_leakage.csv")
    w_(rows_c, "corrected_contrasts.csv")
    w_(rows_base, "baseline_comparison.csv")
    w_(rows_tr, "trends_relative_temperature.csv")
    w_(rows_era, "trends_era5_predictors.csv")
    w_(rows_split, "split_half.csv")
    w_(rows_mon, "seasonal_residual_by_month.csv")
    w_(rows_amp, "seasonal_amplitude_by_year.csv")
    w_(rows_7, "taal_reference_2020_contrasts.csv")
    w_(rows_tr7, "taal_reference_2018_2021_scenes.csv")
    (OUT / "implementation_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    # comparison with the primary Stage 10 records
    pr = pub[pub["estimator"].eq("dT_syn_weather") & pub["window_d"].isin(WINS)]
    rep = pd.DataFrame(rows_c)
    rep = rep[rep["design"].eq("published_repro")]
    mg = pr.merge(rep, on=["site", "onset", "window_d"], how="left", suffixes=("_pub", "_rep"))
    mg["delta_match"] = np.isclose(mg["delta_pub"], mg["delta_rep"], atol=1e-3) | (mg["delta_pub"].isna() & mg["delta_rep"].isna())
    mg["p_match"] = np.isclose(mg["p_placebo_year"], mg["p_rank"], atol=1e-4) | (mg["p_placebo_year"].isna() & mg["p_rank"].isna())
    mg.to_csv(OUT / "repro_check.csv", index=False, encoding="utf-8-sig")
    print(mg[["site", "onset", "window_d", "delta_pub", "delta_rep", "p_placebo_year", "p_rank",
              "delta_match", "p_match"]].to_string())
    print(json.dumps(facts, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------
#  item 6 add-on: residual seasonality tests and seasonal-misfit bias bounds
# ---------------------------------------------------------------------------
def _harm(doy, k2=True):
    a = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    if k2:
        a += [np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]
    return np.column_stack(a)


def _ftest(y, X0, X1):
    from scipy import stats as st
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r0 = ((y - X0 @ b0) ** 2).sum()
    r1 = ((y - X1 @ b1) ** 2).sum()
    df1 = X1.shape[1] - X0.shape[1]
    df2 = len(y) - X1.shape[1]
    F = ((r0 - r1) / df1) / (r1 / df2)
    return float(F), int(df1), int(df2), float(st.f.sf(F, df1, df2))


def seasonal_tests():
    rows, rows_b = [], []
    for site in SITES3:
        P = prep(site)
        df = fit(P)
        f = df["_fit_w"].to_numpy(bool)
        dt = pd.DatetimeIndex(df["date"])
        doy = dt.dayofyear.to_numpy(float)
        yr = dt.year.to_numpy()
        for nm in ("dT_syn_weather", "T_active_K"):
            v = df[nm].to_numpy(float)
            m = f & np.isfinite(v)
            y, H = v[m], _harm(doy[m])
            one = np.ones((m.sum(), 1))
            F, d1, d2, p = _ftest(y, one, np.c_[one, H])
            b, *_ = np.linalg.lstsq(np.c_[one, H], y, rcond=None)
            # year-specific annual cycle: years with >= 6 scenes in >= 4 months
            yy = yr[m]
            mon = dt.month.to_numpy()[m]
            good_years = [u for u in np.unique(yy)
                          if (yy == u).sum() >= 6 and len(np.unique(mon[yy == u])) >= 4]
            k = np.isin(yy, good_years)
            YF = pd.get_dummies(yy[k]).to_numpy(float)
            Hk = _harm(doy[m][k])
            X0 = np.c_[YF, Hk]
            inter = np.column_stack([YF[:, j:j + 1] * Hk[:, :2] for j in range(1, YF.shape[1])])
            Fi, d1i, d2i, pi = _ftest(y[k], X0, np.c_[X0, inter])
            rows.append(dict(site=site, series=nm, n_fit=int(m.sum()),
                             annual_amp_K=float(np.hypot(b[1], b[2])),
                             semiannual_amp_K=float(np.hypot(b[3], b[4])),
                             F_seasonal=F, df=f"{d1},{d2}", p_seasonal=p,
                             n_years_interaction=len(good_years), F_year_x_annual=Fi,
                             df_int=f"{d1i},{d2i}", p_year_x_annual=pi))
        # bias bound: evaluate (a) residual harmonic, (b) year-specific deviations of the
        # lake's own annual cycle from its mean cycle, at the actual pre/post scene dates
        v = df["dT_syn_weather"].to_numpy(float)
        m = f & np.isfinite(v)
        br, *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), _harm(doy[m])], v[m], rcond=None)
        ta = df["T_active_K"].to_numpy(float)
        mt = f & np.isfinite(ta)
        bm, *_ = np.linalg.lstsq(np.c_[np.ones(mt.sum()), _harm(doy[mt], False)], ta[mt], rcond=None)
        devs = []
        for u in np.unique(yr[mt]):
            s_ = mt & (yr == u)
            if s_.sum() < 6 or len(np.unique(dt.month.to_numpy()[s_])) < 4:
                continue
            bu, *_ = np.linalg.lstsq(np.c_[np.ones(s_.sum()), _harm(doy[s_], False)], ta[s_], rcond=None)
            devs.append(bu[1:] - bm[1:])
        dn = df["dnum"].to_numpy()
        for w in WINS:
            for e in events_in_record(dn, P["ke"], w):
                t0 = e["t0n"]
                pre = (dn >= t0 - w) & (dn <= t0 - 1) & np.isfinite(v)
                post = (dn >= t0) & (dn <= t0 + w) & np.isfinite(v)
                if pre.sum() < 2 or post.sum() < 2:
                    continue
                hr = _harm(doy) @ br[1:]
                bias_res = float(np.median(hr[post]) - np.median(hr[pre]))
                bs = []
                for dv in devs:
                    h = _harm(doy, False) @ dv
                    bs.append(float(np.median(h[post]) - np.median(h[pre])))
                bs = np.array(bs)
                rows_b.append(dict(site=site, onset=str(e["t0"].date()), window_d=w,
                                   bias_from_residual_seasonality_K=bias_res,
                                   n_years_dev=len(bs),
                                   yearly_cycle_dev_bias_sd_K=float(np.std(bs, ddof=1)) if len(bs) > 2 else np.nan,
                                   yearly_cycle_dev_bias_maxabs_K=float(np.max(np.abs(bs))) if len(bs) else np.nan))
    pd.DataFrame(rows).to_csv(OUT / "seasonal_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_b).to_csv(OUT / "seasonal_bias_bounds.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).round(4).to_string())
    print(pd.DataFrame(rows_b).round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--seasonal-only" not in sys.argv:
        main()
    seasonal_tests()
