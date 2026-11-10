# -*- coding: utf-8 -*-
"""Ground validation against the GeoNet lake temperature and energy scales.

Pipeline functions are imported and called, not modified; variants are
re-implemented here. All outputs go to results/11_analysis/validation_energy/.

    python -m pipeline.analysis.validation_energy
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from pipeline import s3b_synthetic_control as SC
from pipeline import s4c_energy_balance as EB
from pipeline import s9b_event_stats as B9
from pipeline import s9d_retrieval_unc as D9
from pipeline import s10_post_eruption as P
from pipeline.config import settings as S
from pipeline.config.sites import get_site
from pipeline.s3_delta_t import planck_radiance, dL_dT, unmix

warnings.filterwarnings("ignore")
OUT = S.ROOT / "results" / "11_analysis" / "validation_energy"
OUT.mkdir(parents=True, exist_ok=True)
RAW = S.ROOT / "data" / "raw" / "ground_thermal" / "ruapehu"
RNG_SEED = 20260927
SUMMARY: dict = {}


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


# =============================================================================
# A. GROUND DATA
# =============================================================================
def ground_inventory() -> pd.DataFrame:
    rows = []
    for fn in ("tilde_RU001_lake_temperature_envirosensor.csv",
               "tilde_RU001_lake_temperature_manualcollect.csv"):
        g = pd.read_csv(RAW / fn, dtype=str)
        g["t"] = pd.to_datetime(g["timestamp"], utc=True, errors="coerce", format="ISO8601")
        g["qualityControl"] = g["qualityControl"].fillna("")
        for key, s in g.groupby(["domain", "sensorCode", "method", "aspect"]):
            rows.append(dict(domain=key[0], sensorCode=key[1], method=key[2],
                             aspect=key[3], n=len(s),
                             first=str(s["t"].min())[:19], last=str(s["t"].max())[:19],
                             n_distinct_days=int(s["t"].dt.floor("D").nunique()),
                             qc_values="|".join(sorted(s["qualityControl"].unique())),
                             error_values="|".join(sorted(s["error"].unique()))[:40],
                             frac_timestamp_midnight_utc=round(float(
                                 (s["t"].dt.strftime("%H:%M") == "00:00").mean()), 3)))
    return pd.DataFrame(rows)


def ground_native(which: str) -> pd.DataFrame:
    """Converted GeoNet series at native time, station subset `which`.

    all      : what the pipeline uses (every station, incl. MC02)
    clean    : excludes MC02 (method 'satellite-sensing', whole-lake)
    tele     : envirosensor 'snapshot' series only (2009-)
    manual   : thermometer / thermocouple hand samples only (MC01, MC03, MC04)
    """
    g = pd.read_csv(RAW / "geonet_converted.csv")
    g = g[g["variable"].eq("temperature")]
    st = g["station"].astype(str)
    if which == "clean":
        g = g[~st.str.endswith("MC02")]
    elif which == "tele":
        g = g[st.str.contains("envirosensor")]
    elif which == "manual":
        g = g[st.str.contains("manualcollect") & ~st.str.endswith("MC02")]
    t = pd.to_datetime(g["datetime"], utc=True, errors="coerce", format="mixed")
    d = pd.DataFrame({"t": t, "insitu_C": pd.to_numeric(g["value"], errors="coerce"),
                      "station": g["station"].to_numpy()}).dropna(subset=["t", "insitu_C"])
    return d


def ground_daily(which: str) -> pd.Series:
    d = ground_native(which)
    d["t"] = d["t"].dt.floor("D")
    return d.groupby("t")["insitu_C"].median().sort_index()


def manual_frequency() -> pd.DataFrame:
    d = ground_native("manual")
    d = d[(d["t"] >= "1999-01-01") & (d["t"] < "2009-05-01")]
    days = d["t"].dt.floor("D").drop_duplicates()
    per_m = days.dt.to_period("M").value_counts()
    idx = pd.period_range("1999-01", "2009-04", freq="M")
    per_m = per_m.reindex(idx, fill_value=0)
    gaps = days.sort_values().diff().dt.days.dropna()
    return pd.DataFrame([dict(
        period="1999-01..2009-04 (manual, excl. MC02)", n_sample_days=len(days),
        months=len(idx), median_samples_per_month=float(per_m.median()),
        frac_months_0=round(float((per_m == 0).mean()), 3),
        frac_months_1_3=round(float(((per_m >= 1) & (per_m <= 3)).mean()), 3),
        frac_months_gt3=round(float((per_m > 3).mean()), 3),
        median_gap_days=float(gaps.median()), p90_gap_days=float(gaps.quantile(.9)),
        max_gap_days=float(gaps.max()))])


# ---------------------------------------------------------------------------
def satellite_scenes(f_min=None) -> pd.DataFrame:
    d = pd.read_csv(S.PROCESSED_DIR / "ruapehu_deltaT.csv")
    d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
    d = d[d["qc_flag"].eq("ok")]
    d = d[pd.to_numeric(d["T_active_K"], errors="coerce").notna()]
    if f_min is not None:
        d = d[pd.to_numeric(d["f_est_act"], errors="coerce") >= f_min]
    return d.sort_values("t").reset_index(drop=True)


def match(sat: pd.DataFrame, ins: pd.DataFrame, tol_h: float) -> pd.DataFrame:
    ins = ins.groupby("t", as_index=False).agg(
        insitu_C=("insitu_C", "median"),
        stations=("station", lambda s: "|".join(sorted(set(s)))))
    sat = sat[(sat["t"] >= ins["t"].min()) & (sat["t"] <= ins["t"].max())]
    cols = [c for c in ("t", "date", "T_active_K", "f_est_act", "spacecraft",
                        "bg_active_K") if c in sat.columns]
    m = pd.merge_asof(sat[cols].sort_values("t"),
                      ins.rename(columns={"t": "t_insitu"})
                      .assign(t=lambda x: x["t_insitu"]).sort_values("t"),
                      on="t", direction="nearest",
                      tolerance=pd.Timedelta(hours=tol_h)).dropna(subset=["insitu_C"])
    m["sat_C"] = pd.to_numeric(m["T_active_K"], errors="coerce") - 273.15
    m["dt_h"] = (m["t"] - m["t_insitu"]).dt.total_seconds() / 3600
    return m


def reg_stats(x, y, n_boot=4000, seed=RNG_SEED) -> dict:
    """OLS y (satellite) on x (ground), with analytic and pair-bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8:
        return dict(n=n)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    s2 = float(res @ res / (n - 2))
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_b = np.sqrt(s2 / sxx)
    se_a = np.sqrt(s2 * (1 / n + x.mean() ** 2 / sxx))
    rng = np.random.default_rng(seed)
    bb, aa = [], []
    for _ in range(n_boot):
        i = rng.integers(n, size=n)
        if np.std(x[i]) == 0:
            continue
        b1, a1 = np.polyfit(x[i], y[i], 1)
        bb.append(b1); aa.append(a1)
    d = y - x
    # reduced-major-axis slope (symmetric) as a direction-free reference
    r = float(np.corrcoef(x, y)[0, 1])
    rma = np.sign(r) * np.std(y) / np.std(x)
    return dict(n=n, bias_K=d.mean(), median_diff_K=np.median(d), sd_K=d.std(ddof=1),
                rmse_K=np.sqrt(np.mean(d ** 2)), mae_K=np.mean(np.abs(d)), r=r,
                slope=b, slope_ci95_lo=b - 1.96 * se_b, slope_ci95_hi=b + 1.96 * se_b,
                slope_boot_lo=np.percentile(bb, 2.5), slope_boot_hi=np.percentile(bb, 97.5),
                intercept_C=a, intercept_ci95_lo=a - 1.96 * se_a,
                intercept_ci95_hi=a + 1.96 * se_a,
                intercept_boot_lo=np.percentile(aa, 2.5),
                intercept_boot_hi=np.percentile(aa, 97.5),
                slope_rma=rma, loa_lo_K=d.mean() - 1.96 * d.std(ddof=1),
                loa_hi_K=d.mean() + 1.96 * d.std(ddof=1),
                x_min=x.min(), x_max=x.max())


def matched_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pairs_keep = [], None
    for gset in ("all", "clean", "tele", "manual"):
        ins = ground_native(gset)
        for fscreen in (None, 0.70):
            sat = satellite_scenes(fscreen)
            for tol in (1.0, 3.0, 24.0):
                m = match(sat, ins, tol)
                st = reg_stats(m["insitu_C"], m["sat_C"])
                n_man = int(m["stations"].str.contains("manualcollect").sum())
                n_mc02 = int(m["stations"].str.contains("MC02").sum())
                rows.append(dict(ground_set=gset,
                                 f_screen=("none" if fscreen is None else f">={fscreen}"),
                                 tolerance_h=tol, n_pairs_with_manual=n_man,
                                 n_pairs_with_MC02=n_mc02,
                                 median_abs_dt_min=round(float(m["dt_h"].abs().median() * 60), 1),
                                 first=str(m["t"].min())[:10], last=str(m["t"].max())[:10],
                                 **{k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                                    for k, v in st.items()}))
                if gset == "all" and fscreen is None and tol == 3.0:
                    pairs_keep = m
    return pd.DataFrame(rows), pairs_keep


# ---------------------------------------------------------------------------
def relative_vs_telemetry() -> pd.DataFrame:
    """Does relative temperature track telemetry anomalies? (telemetry era)"""
    df, ke, buf = P.load("ruapehu", P.F_MIN)
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    tele = ground_native("tele")
    rows = []
    # deseasonalised telemetry: daily median minus annual+semi-annual harmonic fit
    dd = tele.assign(day=tele["t"].dt.floor("D")).groupby("day")["insitu_C"].median()
    doy = dd.index.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    X = np.c_[np.ones(len(dd)), np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy)]
    beta, *_ = np.linalg.lstsq(X, dd.to_numpy(float), rcond=None)
    for tol in (3.0, 24.0):
        sat = df[["t", "date", "T_active_K", "dT_syn_weather", "dT_syn"]].copy()
        m = match(sat.assign(f_est_act=np.nan), tele, tol)
        m = m.merge(df[["t", "dT_syn_weather", "dT_syn"]], on="t", how="left")
        dy = m["t_insitu"].dt.dayofyear.to_numpy() / 365.25 * 2 * np.pi
        Xm = np.c_[np.ones(len(m)), np.sin(dy), np.cos(dy), np.sin(2 * dy), np.cos(2 * dy)]
        m["ground_anom_K"] = m["insitu_C"] - Xm @ beta
        for ycol, xcol, lab in (
                ("sat_C", "insitu_C", "lake ST vs telemetry (absolute)"),
                ("dT_syn_weather", "insitu_C", "relative temperature vs telemetry temperature"),
                ("dT_syn_weather", "ground_anom_K", "relative temperature vs deseasonalised telemetry anomaly"),
                ("dT_syn", "ground_anom_K", "relative (no weather adj.) vs deseasonalised telemetry anomaly")):
            x = m[xcol].to_numpy(float); y = m[ycol].to_numpy(float)
            st = reg_stats(x, y)
            ok = np.isfinite(x) & np.isfinite(y)
            cen = (y[ok] - y[ok].mean()) - (x[ok] - x[ok].mean())
            st["centred_rmsd_K"] = float(np.sqrt(np.mean(cen ** 2)))
            rows.append(dict(tolerance_h=tol, comparison=lab, f_screen=">=0.70",
                             **{k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                                for k, v in st.items()}))
    return pd.DataFrame(rows)


# =============================================================================
# A. RETRIEVAL-ERROR PROPAGATION to the relative temperature
# =============================================================================
def propagate(site_key: str, weights: dict, n_mc=600, seed=D9.SEED):
    """Monte Carlo propagation of the per-scene retrieval errors (s9d_retrieval_unc)
    to the unmixed lake temperature, the lake-minus-ring difference and the
    synthetic-reference difference. In the last, the atmospheric (ST_QA) error
    cancels only for donors observed in the same scene (ring, other water
    bodies) and not for the fitted seasonal harmonic."""
    ftab = D9.f_uncertainty_table([site_key])
    sigma_f = float(ftab["sigma_f"].iloc[0]) if len(ftab) else 0.10
    d = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_deltaT.csv")
    d = d[d["qc_flag"].eq("ok")].reset_index(drop=True)
    _dlt = pd.to_numeric(d.get("unmix_delta_active_K"), errors="coerce").fillna(0.0)
    T_pix = (pd.to_numeric(d["T_active_K"], errors="coerce") - _dlt).to_numpy(float)
    T_bg = pd.to_numeric(d["bg_active_K"], errors="coerce").to_numpy(float)
    f0 = pd.to_numeric(d["f_est_act"], errors="coerce").to_numpy(float)
    ss = D9.scene_sigmas(site_key)
    d = d.merge(ss, on="date", how="left")
    SEk = np.clip(D9._col(d, "se_K", 0.3).to_numpy(float), 0.05, 3.0)
    QAk = np.clip(D9._col(d, "st_qa_K", D9.SIGMA_ATM_FALLBACK_K).to_numpy(float), 0.2, 10.0)
    BGk = np.clip(D9._col(d, "bg_std_K", 2.0).to_numpy(float), 0.5, 12.0)
    ok = np.isfinite(T_pix) & np.isfinite(T_bg) & np.isfinite(f0)
    Tp, Tb, F = T_pix[ok], T_bg[ok], f0[ok]
    SEk, QAk, BGk = SEk[ok], QAk[ok], BGk[ok]
    n = len(Tp)
    rng = np.random.default_rng(seed)
    w_same = float(weights.get("same_scene", 0.0))      # ring + other water bodies
    w_ring = float(weights.get("ring", 0.0))

    def draw(which):
        tp = Tp[None, :] + (rng.normal(0, SEk, (n_mc, n)) if "pixel" in which else 0)
        bg_se = BGk / np.sqrt(D9.N_BG_PIXELS)
        tb_local = Tb[None, :] + (rng.normal(0, BGk, (n_mc, n)) if "background" in which else 0)
        tb = Tb[None, :] + (rng.normal(0, bg_se, (n_mc, n)) if "background" in which else 0)
        common = np.zeros((n_mc, n))
        if "atm" in which:
            common = rng.normal(0, QAk, (n_mc, n))
            tp, tb, tb_local = tp + common, tb + common, tb_local + common
        if "emis" in which:
            dL = rng.normal(0, D9.SIGMA_EPS, (n_mc, n)) * planck_radiance(Tp[None, :])
            tp = tp + dL / dL_dT(Tp[None, :])
        ff = np.clip(F[None, :] + (rng.normal(0, sigma_f, (n_mc, n)) if "f" in which else 0), 0.05, 1.0)
        T, _s, _a = unmix(tp, tb_local, ff, np.broadcast_to(SEk, tp.shape),
                          np.broadcast_to(BGk, tp.shape), sigma_f=sigma_f)
        # synthetic-reference difference: lake - [w_ring*ring + w_other*(other same-scene
        # water, carrying only the common atmospheric error) + w_clim*clim (no error)]
        w_other = max(w_same - w_ring, 0.0)
        ref_err = w_ring * (tb - Tb[None, :]) + w_other * common
        rel_syn = T - (Tb[None, :] * 0 + ref_err)
        return T, T - tb, rel_syn

    ALL = {"pixel", "atm", "emis", "background", "f"}
    full, full_rel, full_syn = draw(ALL)
    tot = np.nanmean(np.nanvar(full, axis=0))
    tot_r = np.nanmean(np.nanvar(full_rel, axis=0))
    tot_s = np.nanmean(np.nanvar(full_syn, axis=0))
    per = pd.DataFrame(dict(date=d.loc[ok, "date"].to_numpy(), f=F,
                            sigma_abs=np.nanstd(full, axis=0, ddof=1),
                            sigma_ring=np.nanstd(full_rel, axis=0, ddof=1),
                            sigma_syn=np.nanstd(full_syn, axis=0, ddof=1)))
    rows = []
    for nm, which in (("f", {"f"}), ("background", {"background"}),
                      ("lake-mean SE", {"pixel"}), ("atmospheric (ST_QA)", {"atm"}),
                      ("emissivity", {"emis"})):
        a, r, s = draw(which)
        rows.append(dict(site=site_key, source=nm,
                         sigma_absolute_K=np.sqrt(np.nanmean(np.nanvar(a, axis=0))),
                         sigma_lake_minus_ring_K=np.sqrt(np.nanmean(np.nanvar(r, axis=0))),
                         sigma_lake_minus_synthetic_ref_K=np.sqrt(np.nanmean(np.nanvar(s, axis=0)))))
    rows.append(dict(site=site_key, source="ALL (joint)", sigma_absolute_K=np.sqrt(tot),
                     sigma_lake_minus_ring_K=np.sqrt(tot_r),
                     sigma_lake_minus_synthetic_ref_K=np.sqrt(tot_s)))
    out = pd.DataFrame(rows)
    out["sigma_f_used"] = sigma_f
    out["n_scenes_all_qc_ok"] = n
    out["w_same_scene_donors"] = w_same
    out["w_ring"] = w_ring
    return out, per


def donor_weights(site_key) -> dict:
    ke = list(get_site(site_key).known_events or [])
    got = SC.series(site_key, ke, 180, f_min=P.F_MIN)
    meta = got[1]
    w = {str(k): float(v) for k, v in meta["weights"].items()}
    ring = w.get("background_ring", 0.0)
    clim = w.get("seasonal_cycle", 0.0)
    same = sum(v for k, v in w.items() if k != "seasonal_cycle")
    return dict(raw=w, ring=ring, clim=clim, same_scene=same)


# =============================================================================
# A. HEATING EPISODES
# =============================================================================
def heating_events_from(ins: pd.Series, rise_K=3.0, window_d=60, t_min=None):
    """Copy of s9b.insitu_heating_events for an arbitrary daily ground series."""
    grid = ins.resample("D").mean().interpolate(limit=30)
    rise = grid - grid.shift(window_d)
    hot = (rise > rise_K).fillna(False)
    out, run = [], None
    for t, v in hot.items():
        if v and run is None:
            run = t
        elif not v and run is not None:
            out.append((run, t - pd.Timedelta(days=1))); run = None
    if run is not None:
        out.append((run, hot.index[-1]))
    merged = []
    for s, e in out:
        if merged and (s - merged[-1][1]).days <= 30:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    ev = [dict(date=str(s.date()), start=str(s.date()), end=str(e.date()),
               label="in-situ heating", severity="unrest", kind="insitu_heating")
          for s, e in merged if (e - s).days >= 7]
    if t_min is not None:
        ev = [e for e in ev if pd.Timestamp(e["start"]) >= pd.Timestamp(t_min)]
    return ev


def heating_test(ev, label, restrict_null_to_span=False, f_min=P.F_MIN) -> dict:
    df = B9.frame("ruapehu", ev, 0, f_min=f_min)
    y = pd.to_numeric(df[P.PRIMARY], errors="coerce").to_numpy(float)
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    wins = [(int(np.datetime64(e["start"], "D").astype("int64")),
             int(np.datetime64(e["end"], "D").astype("int64"))) for e in ev]
    n_in_span = sum(1 for a, b in wins if b >= dnum.min() and a <= dnum.max())
    wins_null = ([(a, b) for a, b in wins if b >= dnum.min() and a <= dnum.max()]
                 if restrict_null_to_span else wins)
    unrest = B9._labels_from_windows(dnum, wins)
    quiet = df["_quiet"].to_numpy(bool) & ~unrest
    ok = np.isfinite(y)
    a, b = y[quiet & ok], y[unrest & ok]
    _, p_shift, used = B9.circular_shift_p(df["date"], y, wins_null)
    lo, hi = B9.block_bootstrap_ci(df["date"], y, unrest)
    per = []
    for s0, s1 in wins:
        m = (dnum >= s0) & (dnum <= s1) & ok
        if m.any():
            per.append(float(np.median(y[m])) - float(np.median(a)))
    heat_days = sum(b_ - a_ + 1 for a_, b_ in wins_null
                    if True) if wins_null else 0
    span = int(dnum.max() - dnum.min()) + 1
    return dict(variant=label, n_scenes=int(ok.sum()), n_episodes_total=len(ev),
                n_episodes_overlapping_satellite_span=n_in_span,
                n_episodes_used_in_null=len(wins_null),
                heating_days_in_null_windows=int(heat_days), satellite_span_days=span,
                n_episodes_with_scenes=len(per), n_positive=int(sum(v > 0 for v in per)),
                n_quiet=int(len(a)), n_heating=int(len(b)),
                diff_K=round(float(np.median(b) - np.median(a)), 3),
                boot_ci_lo=lo, boot_ci_hi=hi, p_window_shift=round(float(p_shift), 5),
                n_valid_shifts=used)


TELE_START = "2009-04-15"


def heating_test_era(ev, label, f_min=P.F_MIN) -> dict:
    """Heating contrast restricted to the telemetry era. A dummy event before the
    satellite record forces the pipeline's baseline rule to 'outside event
    windows', so the reference and weather fits use every non-heating scene;
    statistics (median difference, circular-shift null, 90-d block bootstrap)
    are then computed on telemetry-era scenes and episodes only."""
    ev = [e for e in ev if pd.Timestamp(e["end"]) >= pd.Timestamp(TELE_START)]
    dummy = dict(date="1980-01-01", start="1980-01-01", end="1980-01-08",
                 label="dummy", severity="unrest", kind="dummy")
    df = B9.frame("ruapehu", [dummy] + ev, 0, f_min=f_min)
    df = df[df["date"] >= pd.Timestamp(TELE_START)].reset_index(drop=True)
    y = pd.to_numeric(df[P.PRIMARY], errors="coerce").to_numpy(float)
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    wins = [(max(int(np.datetime64(e["start"], "D").astype("int64")), int(dnum.min())),
             int(np.datetime64(e["end"], "D").astype("int64"))) for e in ev]
    unrest = B9._labels_from_windows(dnum, wins)
    ok = np.isfinite(y)
    a, b = y[~unrest & ok], y[unrest & ok]
    _, p_shift, used = B9.circular_shift_p(df["date"], y, wins)
    lo, hi = B9.block_bootstrap_ci(df["date"], y, unrest)
    per = []
    for s0, s1 in wins:
        m = (dnum >= s0) & (dnum <= s1) & ok
        if m.any():
            per.append(float(np.median(y[m])) - float(np.median(a)))
    return dict(variant=label, n_scenes=int(ok.sum()), n_episodes_total=len(ev),
                n_episodes_overlapping_satellite_span=len(ev),
                n_episodes_used_in_null=len(wins),
                heating_days_in_null_windows=int(sum(b_ - a_ + 1 for a_, b_ in wins)),
                satellite_span_days=int(dnum.max() - dnum.min()) + 1,
                n_episodes_with_scenes=len(per), n_positive=int(sum(v > 0 for v in per)),
                n_quiet=int(len(a)), n_heating=int(len(b)),
                diff_K=round(float(np.median(b) - np.median(a)), 3),
                boot_ci_lo=lo, boot_ci_hi=hi, p_window_shift=round(float(p_shift), 5),
                n_valid_shifts=used)


# =============================================================================
# A. TABLE 4 + 2007 DECOMPOSITION
# =============================================================================
def table4_and_decomposition():
    df, ke, _ = P.load("ruapehu", P.F_MIN)
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    series = {k: ground_daily(k) for k in ("all", "clean")}
    for k in series:
        series[k].index = series[k].index.tz_localize(None)
    grid = {k: v.resample("D").mean().interpolate(limit=30) for k, v in series.items()}
    rows, dec, scenes = [], [], []
    onsets_ = [pd.Timestamp("2006-10-04"), pd.Timestamp("2007-09-25")]
    for w in P.WINDOWS:
        for t in onsets_:
            if w == 365 and t.year == 2007:
                pass
            t0n = int(np.datetime64(t.date(), "D").astype("int64"))
            rec = dict(onset=str(t.date()), window_d=w)
            for col in (P.PRIMARY, "T_active_K"):
                y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
                d_, a_, b_ = P.before_after(dnum, y, t0n, w)
                # s10.insitu_check (insitu_before_after.csv) accepts >= 1 scene per side
                rec[f"sat_{col}_delta"] = d_ if (a_ >= 1 and b_ >= 1) else np.nan
                rec[f"sat_{col}_meets_min2"] = bool(a_ >= 2 and b_ >= 2)
                rec["n_sat_pre"], rec["n_sat_post"] = a_, b_
            for k, ins in series.items():
                pre = ins[(ins.index >= t - pd.Timedelta(days=w)) & (ins.index < t)]
                post = ins[(ins.index >= t) & (ins.index <= t + pd.Timedelta(days=w))]
                rec[f"ground_{k}_n_pre"], rec[f"ground_{k}_n_post"] = len(pre), len(post)
                rec[f"ground_{k}_delta"] = (post.median() - pre.median()
                                            if len(pre) and len(post) else np.nan)
            rows.append(rec)
            # decomposition, using the 'clean' and the 'all' ground series
            m_pre = (dnum >= t0n - w) & (dnum <= t0n - 1)
            m_post = (dnum >= t0n) & (dnum <= t0n + w)
            if m_pre.sum() < 2 or m_post.sum() < 2:
                continue
            for k in ("all", "clean"):
                ins = series[k]
                sd = df.loc[m_pre | m_post, ["date", "T_active_K", P.PRIMARY]].copy()
                sd["side"] = np.where(m_pre[m_pre | m_post], "pre", "post")
                sd["ground_interp_C"] = grid[k].reindex(sd["date"]).to_numpy()
                # nearest actual sample and its distance
                idx = ins.index.to_numpy()
                near, gap = [], []
                for dt_ in sd["date"]:
                    j = np.argmin(np.abs(idx - np.datetime64(dt_)))
                    near.append(float(ins.iloc[j])); gap.append(int(abs((ins.index[j] - dt_).days)))
                sd["ground_nearest_C"], sd["nearest_gap_d"] = near, gap
                sd["sat_lake_C"] = sd["T_active_K"] - 273.15
                sd["onset"], sd["window_d"], sd["ground_set"] = str(t.date()), w, k
                scenes.append(sd)
                G_full = rec[f"ground_{k}_delta"]
                for gcol, lab in (("ground_interp_C", "interp (limit 30 d)"),
                                  ("ground_nearest_C", "nearest sample")):
                    g_pre = sd.loc[sd.side == "pre", gcol].median()
                    g_post = sd.loc[sd.side == "post", gcol].median()
                    G_sub = g_post - g_pre
                    S_l = rec["sat_T_active_K_delta"]
                    S_r = rec[f"sat_{P.PRIMARY}_delta"]
                    dec.append(dict(onset=str(t.date()), window_d=w, ground_set=k,
                                    ground_at_sat_dates=lab,
                                    n_sat_pre=int(m_pre.sum()), n_sat_post=int(m_post.sum()),
                                    ground_all_samples_delta=G_full,
                                    ground_at_sat_dates_delta=G_sub,
                                    sat_lake_delta=S_l, sat_relative_delta=S_r,
                                    sampling_part=G_full - G_sub,
                                    retrieval_part=G_sub - S_l,
                                    reference_part=S_l - S_r,
                                    total_gap=G_full - S_r,
                                    max_nearest_gap_d=int(sd["nearest_gap_d"].max())))
    return (pd.DataFrame(rows), pd.DataFrame(dec),
            pd.concat(scenes, ignore_index=True) if scenes else pd.DataFrame())


# =============================================================================
# B. ENERGY
# =============================================================================
TABLE7 = [("taal", "2020-01-12", 365), ("taal", "2021-07-01", 365),
          ("taal", "2022-10-05", 365), ("taal", "2024-04-12", 365),
          ("ruapehu", "2007-09-25", 180), ("santa_ana", "2005-06-16", 180)]
VOLUME = {  # value, source, epoch  (as coded in pipeline/s10_magma.py GEOM)
    "taal": (4.2e7, "Perttu et al. (2023)", "pre-eruption lake, before 12 Jan 2020"),
    "ruapehu": (9.0e6, "Christenson (1994)", "1991 survey"),
    "santa_ana": (4.7e5, "Bernard et al. (2004)", "2000-2002 surveys"),
}
RHO_W, C_W = 1000.0, 4186.0
RHO_M, L_M, C_M, DT_M = 2600.0, 3.5e5, 1200.0, 300.0


def sig2(x):
    return float(f"{x:.2g}") if np.isfinite(x) else np.nan


def scene_energy_frame(site_key):
    """Paper relative-temperature frame (f >= 0.70) joined to the scene-level
    energy-balance inputs, with fluxes recomputed from the current T_active_K."""
    df, ke, _ = P.load(site_key, P.F_MIN)
    eb = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_energy_balance.csv")
    eb = eb[eb["kind"].eq("active")].copy()
    eb["t"] = pd.to_datetime(eb["datetime"], utc=True)
    met = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_met.csv")
    met["t"] = pd.to_datetime(met["datetime"], utc=True, format="mixed")
    eb = eb.merge(met[["t", "Psfc_hPa"]].drop_duplicates("t"), on="t", how="left")
    x = df[["t", "date", "T_active_K", P.PRIMARY, "Q_total_MW"]].copy()
    x["t"] = pd.to_datetime(x["t"], utc=True)
    x = x.merge(eb[["t", "T_lake_K", "Q_total_MW", "Q_total_Wm2", "Q_evap_Wm2",
                    "Q_sens_Wm2", "Q_rad_Wm2", "Q_solar_net_Wm2", "area_m2", "Ta_K",
                    "RH", "u2_ms", "SWdown_Wm2", "LWdown_Wm2", "Psfc_hPa", "qc_flag"]]
                .rename(columns={"Q_total_MW": "Q_total_MW_eb"}), on="t", how="left")
    fl = EB.surface_fluxes(x["T_active_K"].to_numpy(float), x["Ta_K"].to_numpy(float),
                           np.clip(x["RH"].to_numpy(float), 0.01, 1.0),
                           x["Psfc_hPa"].to_numpy(float) * 100.0, x["u2_ms"].to_numpy(float),
                           x["SWdown_Wm2"].to_numpy(float), x["LWdown_Wm2"].to_numpy(float))
    for k in ("Q_evap_Wm2", "Q_sens_Wm2", "Q_rad_Wm2", "Q_solar_net_Wm2", "Q_total_Wm2"):
        x[k + "_recomp"] = fl[k]
    x["Q_total_MW_recomp"] = fl["Q_total_Wm2"] * x["area_m2"] / 1e6
    return x


def energy_table():
    rows, orders, checks = [], [], []
    frames = {}
    for site, onset, w in TABLE7:
        if site not in frames:
            frames[site] = scene_energy_frame(site)
            x = frames[site]
            dT = (x["T_lake_K"] - x["T_active_K"]).abs()
            checks.append(dict(site=site, n_scenes=len(x),
                               n_with_eb=int(x["T_lake_K"].notna().sum()),
                               max_abs_T_eb_minus_T_current_K=float(dT.max()),
                               n_T_mismatch_gt_0p01K=int((dT > 0.01).sum()),
                               max_abs_Q_recomp_minus_eb_MW=float(
                                   (x["Q_total_MW_recomp"] - x["Q_total_MW_eb"]).abs().max()),
                               max_abs_Qframe_minus_eb_MW=float(
                                   (x["Q_total_MW"] - x["Q_total_MW_eb"]).abs().max())))
        x = frames[site]
        dnum = x["date"].to_numpy("datetime64[D]").astype("int64")
        t0n = int(np.datetime64(onset, "D").astype("int64"))
        pre = (dnum >= t0n - w) & (dnum <= t0n - 1)
        post = (dnum >= t0n) & (dnum <= t0n + w)
        med = lambda c, m: float(np.nanmedian(pd.to_numeric(x.loc[m, c], errors="coerce")))
        mean = lambda c, m: float(np.nanmean(pd.to_numeric(x.loc[m, c], errors="coerce")))
        dT_lake = med("T_active_K", post) - med("T_active_K", pre)
        dT_rel = med(P.PRIMARY, post) - med(P.PRIMARY, pre)
        # order 1 (as implemented): scene flux (MW) -> median(post) - median(pre)
        o1 = med("Q_total_MW", post) - med("Q_total_MW", pre)
        o1r = med("Q_total_MW_recomp", post) - med("Q_total_MW_recomp", pre)
        o1_mean = mean("Q_total_MW_recomp", post) - mean("Q_total_MW_recomp", pre)

        # order 2: flux of window-median inputs (T, met, area) post minus pre
        def q_at(m, Tval=None, metmask=None):
            mm = m if metmask is None else metmask
            T = med("T_active_K", m) if Tval is None else Tval
            f = EB.surface_fluxes(np.array([T]), np.array([med("Ta_K", mm)]),
                                  np.array([np.clip(med("RH", mm), 0.01, 1)]),
                                  np.array([med("Psfc_hPa", mm) * 100]),
                                  np.array([med("u2_ms", mm)]),
                                  np.array([med("SWdown_Wm2", mm)]),
                                  np.array([med("LWdown_Wm2", mm)]))
            return float(f["Q_total_Wm2"][0]), f
        q2_post, _ = q_at(post); q2_pre, _ = q_at(pre)
        o2 = q2_post * med("area_m2", post) / 1e6 - q2_pre * med("area_m2", pre) / 1e6
        # order 3: temperature-only effect — both medians evaluated under pooled
        # median meteorology and a common (pooled median) area
        both = pre | post
        A = med("area_m2", both)
        q3_post, fpost = q_at(post, med("T_active_K", post), both)
        q3_pre, fpre = q_at(pre, med("T_active_K", pre), both)
        o3 = (q3_post - q3_pre) * A / 1e6
        terms = {k: (float(fpost[k][0]) - float(fpre[k][0])) * A / 1e6
                 for k in ("Q_evap_Wm2", "Q_sens_Wm2", "Q_rad_Wm2")}
        # order 1 with a fixed area (median over both windows)
        o1_fixedA = (med("Q_total_Wm2_recomp", post) - med("Q_total_Wm2_recomp", pre)) * A / 1e6
        V, vsrc, vep = VOLUME[site]
        E = RHO_W * C_W * V * dT_lake
        Vm = E / (RHO_M * (L_M + C_M * DT_M))
        rows.append(dict(
            site=site, onset=onset, window_d=w, n_pre=int(pre.sum()), n_post=int(post.sum()),
            T_lake_median_pre_C=med("T_active_K", pre) - 273.15,
            T_lake_median_post_C=med("T_active_K", post) - 273.15,
            dT_lake_K=dT_lake, dT_relative_K=dT_rel,
            area_median_pre_m2=med("area_m2", pre), area_median_post_m2=med("area_m2", post),
            dQ_MW_order1_scene_flux_median_diff=o1,
            dQ_MW_order1_recomputed_current_T=o1r,
            dQ_MW_order1_fixed_area=o1_fixedA,
            dQ_MW_scene_flux_mean_diff=o1_mean,
            dQ_MW_order2_flux_of_median_inputs=o2,
            dQ_MW_order3_temperature_only_pooled_met=o3,
            dQ_evap_MW_T_only=terms["Q_evap_Wm2"], dQ_sens_MW_T_only=terms["Q_sens_Wm2"],
            dQ_rad_MW_T_only=terms["Q_rad_Wm2"],
            lake_volume_m3=V, volume_source=vsrc, volume_epoch=vep,
            rho_w=RHO_W, c_w=C_W, stored_energy_J=E, stored_energy_J_2sf=sig2(E),
            magma_equiv_m3=Vm, magma_equiv_m3_2sf=sig2(Vm),
            rho_m=RHO_M, L_m=L_M, c_m=C_M, dT_m=DT_M))
    return pd.DataFrame(rows), pd.DataFrame(checks), frames


# =============================================================================
def main():
    import time
    t0 = time.time()
    inv = save(ground_inventory(), "A1_ground_inventory")
    freq = save(manual_frequency(), "A1_manual_sampling_frequency")
    print("A1 done", time.time() - t0)

    mc, pairs = matched_comparison()
    save(mc, "A2_matched_satellite_ground")
    save(pairs, "A2_pairs_pipeline_rule_all_pm3h")
    print("A2 done", time.time() - t0)
    rt = save(relative_vs_telemetry(), "A3_relative_vs_telemetry")
    print("A3 done", time.time() - t0)

    wts = donor_weights("ruapehu")
    SUMMARY["ruapehu_donor_weights"] = wts
    budg, per = propagate("ruapehu", wts)
    save(budg.round(3), "A4_uncertainty_budget_ruapehu")
    per_sum = []
    for lab, m in (("all qc-ok scenes", np.ones(len(per), bool)),
                   ("f >= 0.70", per["f"] >= 0.70)):
        p = per[m]
        per_sum.append(dict(subset=lab, n=len(p),
                            rms_sigma_abs=np.sqrt(np.mean(p.sigma_abs ** 2)),
                            median_sigma_abs=p.sigma_abs.median(),
                            rms_sigma_lake_minus_ring=np.sqrt(np.mean(p.sigma_ring ** 2)),
                            median_sigma_lake_minus_ring=p.sigma_ring.median(),
                            rms_sigma_lake_minus_synref=np.sqrt(np.mean(p.sigma_syn ** 2)),
                            median_sigma_lake_minus_synref=p.sigma_syn.median()))
    save(pd.DataFrame(per_sum).round(3), "A4_uncertainty_summary_ruapehu")
    for s in ("taal", "santa_ana"):
        try:
            w_ = donor_weights(s)
            SUMMARY[f"{s}_donor_weights"] = w_
            b_, p_ = propagate(s, w_)
            save(b_.round(3), f"A4_uncertainty_budget_{s}")
        except Exception as exc:                                    # noqa: BLE001
            SUMMARY[f"{s}_propagation_error"] = repr(exc)
    print("A4 done", time.time() - t0)

    # heating episodes
    ev_pipe = B9.insitu_heating_events("ruapehu")
    all_d = ground_daily("all"); clean_d = ground_daily("clean"); tele_d = ground_daily("tele")
    ev_clean = heating_events_from(clean_d)
    ev_tele = heating_events_from(tele_d)
    evrows = [heating_test(ev_pipe, "pipeline (all stations incl. MC02), as reported"),
              heating_test(ev_pipe, "pipeline events, null restricted to episodes within satellite span",
                           restrict_null_to_span=True),
              heating_test(ev_clean, "MC02 excluded"),
              heating_test(ev_clean, "MC02 excluded, null restricted to satellite span",
                           restrict_null_to_span=True),
              heating_test(ev_tele, "telemetry-only episodes (2009-), pipeline quiet rule (quiet = pre-2009 scenes)",
                         restrict_null_to_span=True),
            heating_test_era(ev_tele, "telemetry era only: episodes, quiet and null all within 2009-04-15..end"),
            heating_test_era(ev_pipe, "pipeline episodes evaluated on telemetry-era scenes only")]
    save(pd.DataFrame(evrows), "A5_heating_episode_tests")
    epi = pd.DataFrame(ev_pipe)[["start", "end"]]
    epi["duration_d"] = (pd.to_datetime(epi["end"]) - pd.to_datetime(epi["start"])).dt.days + 1
    epi["era"] = np.where(pd.to_datetime(epi["start"]) >= "2009-04-15", "telemetry", "manual")
    save(epi, "A5_heating_episodes_pipeline")
    SUMMARY["episodes_by_era"] = epi["era"].value_counts().to_dict()
    SUMMARY["episode_duration_days"] = epi["duration_d"].describe().round(1).to_dict()
    print("A5 done", time.time() - t0)

    t4, dec, sc = table4_and_decomposition()
    save(t4.round(3), "A6_ground_comparison")
    save(dec.round(3), "A6_gap_decomposition")
    save(sc.round(3), "A6_decomposition_scenes")
    print("A6 done", time.time() - t0)

    et, chk, frames = energy_table()
    save(chk, "B0_energy_input_consistency")
    save(et, "B1_dimensional_scales")
    for s, x in frames.items():
        save(x, f"B2_scene_energy_{s}")
    print("B done", time.time() - t0)
    (OUT / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
