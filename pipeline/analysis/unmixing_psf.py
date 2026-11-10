# -*- coding: utf-8 -*-
"""Radiance-mixture correction, PSF and sensor heterogeneity.

(a) with vs without sub-pixel unmixing on the same retained scenes
(b) sensor-specific spatial-response models for f (TM 120 m, ETM+ 60 m, TIRS 100 m)
(c) per-sensor consistency of the relative-temperature series in quiet periods,
    cross-sensor near-coincident pairs, and the Santa Ana single-sensor check.

Pipeline modules and records are not modified. Scenario series are written to a
scratch "processed" directory and the unchanged Stage 10 code (s10_post_eruption.run_site)
is pointed at it by re-binding settings.PROCESSED_DIR in this process only.

Outputs: results/11_analysis/unmixing_psf/.

Run:  python -m pipeline.analysis.unmixing_psf
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import settings as S
from pipeline.config.sites import get_site
from pipeline import s10_post_eruption as S10
from pipeline import s2b_water_extent as W
from pipeline.s3_delta_t import unmix
from pipeline.analysis import unmixing_psf_f as PF

SITES = ("taal", "santa_ana", "ruapehu")
ACTIVE = {"taal": "main_crater", "santa_ana": "crater_lake", "ruapehu": "crater_lake"}
CONTROL = {"taal": "taal_lake"}
OUT = S.RECORDS_DIR / "11_analysis" / "unmixing_psf" if hasattr(S, "RECORDS_DIR") \
    else Path("results/11_analysis/unmixing_psf")
ORIG_PROCESSED = S.PROCESSED_DIR
SC_MAP = {"LANDSAT_5": "TM", "LANDSAT_7": "ETM+", "LANDSAT_8": "TIRS", "LANDSAT_9": "TIRS-2"}
LAMBDA_SC = {"LANDSAT_5": 11.45, "LANDSAT_7": 11.335, "LANDSAT_8": 10.895, "LANDSAT_9": 10.895}
NATIVE = {"LANDSAT_5": 120.0, "LANDSAT_7": 60.0, "LANDSAT_8": 100.0, "LANDSAT_9": 100.0}

# primary event contrasts evaluated (site, onset, window)
T5 = [("taal", "2020-01-12", 365), ("taal", "2021-07-01", 365), ("taal", "2022-10-05", 180),
      ("taal", "2022-10-05", 365), ("taal", "2024-04-12", 365),
      ("santa_ana", "2005-06-16", 180), ("santa_ana", "2005-06-16", 365),
      ("ruapehu", "2006-10-04", 180), ("ruapehu", "2006-10-04", 365),
      ("ruapehu", "2007-09-25", 180)]
EST = ["dT_syn_weather", "dT_syn", "T_active_K"]


# ---------------------------------------------------------------------------
#  spatial-response scenarios -> kernel (sigma, box, gaussian_only) per sensor
# ---------------------------------------------------------------------------
def kernels(scn: str, sc: str):
    n = NATIVE[sc]
    s_enc = PF.encircled_sigma(n)            # 0.58 * native (pipeline 37.4 % convention)
    if scn == "baseline":
        return 58.0, 100.0, False
    if scn == "native_box_scaled":           # box = native, sigma scaled to native
        return s_enc, n, False
    if scn == "native_box_scaled_x0.7":
        return 0.7 * s_enc, n, False
    if scn == "native_box_scaled_x1.3":
        return 1.3 * s_enc, n, False
    if scn == "gauss_fwhm_native":           # Gaussian only, FWHM = native resolution
        return n / 2.3548, 0.0, True
    if scn == "common_tm_worst":             # every sensor treated as TM (broadest)
        return PF.encircled_sigma(120.0), 120.0, False
    raise KeyError(scn)


PSF_SCN = ["baseline", "native_box_scaled", "native_box_scaled_x0.7",
           "native_box_scaled_x1.3", "gauss_fwhm_native", "common_tm_worst"]


def f_ref_table():
    """f_max per lake x sensor x scenario, calibrated to the recorded value:
    f_ref_new = f_ref_recorded * f_local(scn) / f_local(baseline)   (removes grid-phase offset)."""
    rows = []
    cache = {}
    for site in SITES:
        lakes = [ACTIVE[site]] + ([CONTROL[site]] if site in CONTROL else [])
        lst = pd.read_csv(S.INTERIM_DIR / f"{site}_lst.csv")
        for lk in lakes:
            f_ee = float(pd.to_numeric(lst.loc[lst["lake"] == lk, "f_est"],
                                       errors="coerce").median())
            for scn in PSF_SCN:
                for sc in NATIVE:
                    key = (site, lk, kernels(scn, sc))
                    if key not in cache:
                        sg, bx, go = kernels(scn, sc)
                        cache[key] = PF.f_stats(site, lk, sg, bx, gaussian_only=go,
                                                dx=5.0 if lk != "taal_lake" else 20.0)
                    st = cache[key]
                    b = cache[(site, lk, kernels("baseline", sc))]
                    ratio = st["f_max"] / b["f_max"] if b["f_max"] > 0 else np.nan
                    sg, bx, go = kernels(scn, sc)
                    rows.append(dict(site=site, lake=lk, scenario=scn, spacecraft=sc,
                                     sigma_m=round(sg, 1), box_m=bx, gaussian_only=go,
                                     f_max_local=round(st["f_max"], 4),
                                     f_max_phase_range=f"{st['f_max_phase_min']:.3f}-{st['f_max_phase_max']:.3f}",
                                     f_mean_local=round(st["f_mean"], 4),
                                     f_ref_ee_baseline=round(f_ee, 4),
                                     f_ref_new=round(min(1.0, f_ee * ratio), 4)))
    return pd.DataFrame(rows)


def f_timevarying_for(site, fref_by_lake: dict):
    """Re-run Stage 2b's f assignment with a replaced reference f (per lake)."""
    s = get_site(site)
    w = pd.read_csv(S.INTERIM_DIR / f"{site}_water_extent.csv")
    inv = pd.read_csv(S.INTERIM_DIR / f"{site}_scene_inventory.csv")
    lst = pd.read_csv(S.INTERIM_DIR / f"{site}_lst.csv")
    for lk, fr in fref_by_lake.items():
        m = w["lake"] == lk
        f = fr * w.loc[m, "area_ratio_baseline"]
        w.loc[m, "f_tir"] = np.where(np.isfinite(f), np.clip(f, W.F_TIR_FLOOR, 1.0), np.nan)
        lst.loc[lst["lake"] == lk, "f_est"] = fr
        lst.loc[lst["lake"] == lk, "f_max"] = fr
    return W.build_f_timevarying(w, s, inv, lst)


# ---------------------------------------------------------------------------
#  scenario deltaT tables
# ---------------------------------------------------------------------------
def load_deltaT(site):
    d = pd.read_csv(ORIG_PROCESSED / f"{site}_deltaT.csv")
    d["T_raw_act"] = d["T_active_K"] - d["unmix_delta_active_K"].fillna(0.0)
    if d["mode"].iloc[0] == "paired":
        d["T_raw_ctl"] = d["T_raw_act"] - d["dT_raw_K"]
    else:
        d["T_raw_ctl"] = d["T_control_K"]
    return d


def sanity_unmix(site):
    d = load_deltaT(site)
    T, _, ok = unmix(d["T_raw_act"], d["bg_active_K"], d["f_est_act"], 1.0, 1.0)
    return float(np.nanmax(np.abs(T - d["T_active_K"])))


def scenario_deltaT(site, scn, fref_tab=None, hold_screen=False):
    d = load_deltaT(site).copy()
    if scn == "no_unmix":
        d["T_active_K"] = d["T_raw_act"]
        if d["mode"].iloc[0] == "paired":
            d["T_control_K"] = d["T_raw_ctl"]
        return d
    if scn == "baseline_repro":
        return d
    if scn == "lambda_sensor":
        # same f, same scenes; Planck conversion at each sensor's band centre
        from pipeline.s3_delta_t import planck_radiance as PR, planck_temperature as PT
        lam = d["spacecraft"].map(LAMBDA_SC).to_numpy(float)
        f = pd.to_numeric(d["f_est_act"], errors="coerce").to_numpy(float)
        ok = np.isfinite(f) & (f >= 0.15) & (f <= 1.0) & np.isfinite(d["bg_active_K"].to_numpy(float))
        f_ = np.where(ok, f, 1.0)
        Lp, Lb = PR(d["T_raw_act"].to_numpy(float), lam), PR(d["bg_active_K"].to_numpy(float), lam)
        Tl = PT(np.where(Lb + (Lp - Lb) / f_ > 0, Lb + (Lp - Lb) / f_, np.nan), lam)
        d["T_active_K"] = np.where(ok, Tl, d["T_raw_act"])
        return d
    # PSF scenario: per-sensor f_ref -> time-varying f -> per-scene f by spacecraft
    f_old = d["f_est_act"].copy()
    lakes = [("act", ACTIVE[site])] + ([("ctl", CONTROL[site])] if site in CONTROL else [])
    for sfx, lk in lakes:
        fnew = np.full(len(d), np.nan)
        for sc in NATIVE:
            fr = float(fref_tab[(fref_tab.site == site) & (fref_tab.lake == lk)
                                & (fref_tab.scenario == scn)
                                & (fref_tab.spacecraft == sc)]["f_ref_new"].iloc[0])
            ftv = f_timevarying_for(site, {lk: fr})
            ftv = ftv[ftv["lake"] == lk][["date", "f_tir"]]
            ftv["date"] = pd.to_datetime(ftv["date"]).dt.strftime("%Y-%m-%d")
            mm = d[["date"]].merge(ftv, on="date", how="left")["f_tir"].to_numpy()
            mm = np.where(np.isfinite(mm), mm, fr)          # geometric_constant fallback
            sel = (d["spacecraft"] == sc).to_numpy()
            fnew[sel] = mm[sel]
        if sfx == "ctl" and d["mode"].iloc[0] != "paired":
            continue
        T, _, _ = unmix(d[f"T_raw_{sfx}"], d[f"bg_{'active' if sfx=='act' else 'control'}_K"],
                        fnew, 1.0, 1.0)
        if sfx == "act":
            d["T_active_K"] = T
            d["f_est_act"] = fnew
        else:
            d["T_control_K"] = T
            d["f_est_ctl"] = fnew
    if hold_screen:
        d["f_est_act"] = f_old
    return d


def run_s10(site, d, f_min=S10.F_MIN, tmp=None):
    """Point the unchanged Stage-10 code at a scratch processed dir holding `d`."""
    for fn in (f"{site}_dataset.csv", f"{site}_energy_balance.csv"):
        src = ORIG_PROCESSED / fn
        if src.exists() and not (tmp / fn).exists():
            shutil.copy2(src, tmp / fn)
    d.drop(columns=["T_raw_act", "T_raw_ctl"], errors="ignore").to_csv(
        tmp / f"{site}_deltaT.csv", index=False)
    S.PROCESSED_DIR = tmp
    try:
        st, ev, epo, meta = S10.run_site(site, f_min)
        df, _, _ = S10.load(site, f_min)
    finally:
        S.PROCESSED_DIR = ORIG_PROCESSED
    return ev, df, meta


def pick_t5(ev, tag, only_site=None):
    rows = []
    for site, onset, w in T5:
        if only_site and site != only_site:
            continue
        for est in EST:
            e = ev[(ev["site"] == site) & (ev["onset"] == onset) & (ev["window_d"] == w)
                   & (ev["estimator"] == est)]
            if len(e):
                r = e.iloc[0]
                rows.append(dict(scenario=tag, site=site, onset=onset, window_d=w,
                                 estimator=est, n_pre=int(r["n_pre"]), n_post=int(r["n_post"]),
                                 delta=r["delta"], p_rank=r["p_placebo_year"],
                                 n_years=int(r["placebo_n_years"]),
                                 max_rank=bool(np.isfinite(r["p_placebo_year"])
                                               and r["placebo_n_years"] >= S10.MIN_YEARS
                                               and abs(r["p_placebo_year"]
                                                       - 1.0 / (r["placebo_n_years"] + 1)) < 2e-4)))
            else:
                rows.append(dict(scenario=tag, site=site, onset=onset, window_d=w,
                                 estimator=est, delta=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  (c) sensor consistency
# ---------------------------------------------------------------------------
def sensor_consistency(df_base: dict, dT: dict):
    rows, pair_rows, overlap_rows = [], [], []
    for site in SITES:
        df = df_base[site].copy()
        d = dT[site][["datetime", "spacecraft", "bg_active_K", "T_raw_act", "scene_id"]].copy()
        df = df.merge(d.drop_duplicates("datetime"), on="datetime", how="left")
        df["lake_minus_bg_unmixed"] = df["T_active_K"] - df["bg_active_K"]
        df["lake_minus_bg_raw"] = df["T_raw_act"] - df["bg_active_K"]
        q = df[df["_quiet"]]
        for sc, g in q.groupby("spacecraft"):
            for col in ("dT_syn_weather", "lake_minus_bg_raw", "lake_minus_bg_unmixed"):
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                rows.append(dict(site=site, spacecraft=sc, sensor=SC_MAP.get(sc), quantity=col,
                                 n=len(v), years=f"{g['date'].dt.year.min()}-{g['date'].dt.year.max()}",
                                 median=round(v.median(), 2), iqr=round(v.quantile(.75) - v.quantile(.25), 2),
                                 sd=round(v.std(ddof=1), 2) if len(v) > 2 else np.nan))
        # overlap-period comparison (quiet only), each pair of sensors, common years
        from scipy.stats import mannwhitneyu
        scs = sorted(q["spacecraft"].dropna().unique())
        for i, a in enumerate(scs):
            for b in scs[i + 1:]:
                ya = set(q.loc[q.spacecraft == a, "date"].dt.year)
                yb = set(q.loc[q.spacecraft == b, "date"].dt.year)
                yrs = ya & yb
                if not yrs:
                    continue
                qq = q[q["date"].dt.year.isin(yrs)]
                for col in ("dT_syn_weather", "lake_minus_bg_raw"):
                    va = pd.to_numeric(qq.loc[qq.spacecraft == a, col], errors="coerce").dropna()
                    vb = pd.to_numeric(qq.loc[qq.spacecraft == b, col], errors="coerce").dropna()
                    if len(va) < 3 or len(vb) < 3:
                        continue
                    rng = np.random.default_rng(1)
                    bs = [np.median(rng.choice(vb, len(vb))) - np.median(rng.choice(va, len(va)))
                          for _ in range(2000)]
                    overlap_rows.append(dict(site=site, sensor_a=SC_MAP[a], sensor_b=SC_MAP[b],
                                             common_years=f"{min(yrs)}-{max(yrs)} ({len(yrs)} yr)",
                                             quantity=col, n_a=len(va), n_b=len(vb),
                                             median_a=round(va.median(), 2), median_b=round(vb.median(), 2),
                                             diff_b_minus_a=round(vb.median() - va.median(), 2),
                                             ci95=f"{np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}",
                                             p_mwu=round(float(mannwhitneyu(va, vb).pvalue), 4)))
        # near-coincident pairs (<= 8 d), quiet, cross- vs same-sensor
        qs = q.sort_values("date").reset_index(drop=True)
        t = qs["date"].to_numpy("datetime64[D]").astype(int)
        for col in ("dT_syn_weather", "lake_minus_bg_raw"):
            y = pd.to_numeric(qs[col], errors="coerce").to_numpy(float)
            sc = qs["spacecraft"].to_numpy()
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    lag = t[j] - t[i]
                    if lag > 8:
                        break
                    if lag < 1 or not (np.isfinite(y[i]) and np.isfinite(y[j])):
                        continue
                    pair_rows.append(dict(site=site, quantity=col, lag_d=int(lag),
                                          a=SC_MAP[sc[i]], b=SC_MAP[sc[j]],
                                          same=sc[i] == sc[j],
                                          # orient as later-generation minus earlier
                                          diff=(y[j] - y[i]) if sc[j] >= sc[i] else (y[i] - y[j])))
    pr = pd.DataFrame(pair_rows)
    psum = (pr.assign(pair=np.where(pr["same"], "same sensor",
                                    pr[["a", "b"]].apply(lambda r: "-".join(sorted(r, key=list(SC_MAP.values()).index)), axis=1)))
            .groupby(["site", "quantity", "pair"])["diff"]
            .agg(n="count", median_diff="median",
                 mean_diff="mean", sd=lambda s: s.std(ddof=1))
            .round(2).reset_index())
    return pd.DataFrame(rows), pd.DataFrame(overlap_rows), psum


def santa_ana_within_sensor(df, dT):
    d = dT[["datetime", "spacecraft", "T_raw_act"]].copy()
    df = df.merge(d.drop_duplicates("datetime"), on="datetime", how="left")
    t0 = pd.Timestamp("2005-06-16")
    rows = []
    for sc in ("LANDSAT_5", "LANDSAT_7", "ALL"):
        g = df if sc == "ALL" else df[df["spacecraft"] == sc]
        for w in (180, 365, 730, None):
            pre = g[(g.date < t0) & ((g.date >= t0 - pd.Timedelta(days=w)) if w else True)]
            post = g[(g.date >= t0) & ((g.date <= t0 + pd.Timedelta(days=w)) if w else True)]
            if sc != "ALL" and w is None:
                # whole overlap of that sensor
                pass
            for col in ("T_active_K", "T_raw_act", "dT_syn_weather"):
                a, b = pd.to_numeric(pre[col], errors="coerce").dropna(), pd.to_numeric(post[col], errors="coerce").dropna()
                off = 273.15 if col != "dT_syn_weather" else 0.0
                rows.append(dict(spacecraft=sc, window=("whole record" if w is None else f"{w} d"),
                                 quantity=col, n_pre=len(a), n_post=len(b),
                                 median_pre=round(a.median() - off, 2) if len(a) else np.nan,
                                 median_post=round(b.median() - off, 2) if len(b) else np.nan,
                                 delta=round(b.median() - a.median(), 2) if len(a) and len(b) else np.nan))
    return pd.DataFrame(rows)


def window_sensor_mix(dfs, dTs):
    """Sensor composition of the pre/post windows of every T5 contrast (f >= 0.70)."""
    rows = []
    for site, onset, w in T5:
        df = dfs[site].merge(dTs[site][["datetime", "spacecraft"]].drop_duplicates("datetime"),
                             on="datetime", how="left")
        t0 = pd.Timestamp(onset)
        y = pd.to_numeric(df["dT_syn_weather"], errors="coerce")
        pre = df[(df.date >= t0 - pd.Timedelta(days=w)) & (df.date < t0) & y.notna()]
        post = df[(df.date >= t0) & (df.date <= t0 + pd.Timedelta(days=w)) & y.notna()]
        fmt = lambda g: " ".join(f"{SC_MAP[k]}:{v}" for k, v in g["spacecraft"].value_counts().sort_index().items())
        rows.append(dict(site=site, onset=onset, window_d=w, pre=fmt(pre), post=fmt(post)))
    return pd.DataFrame(rows)


def within_sensor_event_deltas(dfs, dTs):
    """T5 contrasts recomputed from a single sensor (each sensor with >= 2 scenes
    in both windows), primary estimator and lake surface temperature."""
    rows = []
    for site, onset, w in T5:
        df = dfs[site].merge(dTs[site][["datetime", "spacecraft"]].drop_duplicates("datetime"),
                             on="datetime", how="left")
        t0 = pd.Timestamp(onset)
        for sc in sorted(df["spacecraft"].dropna().unique()):
            g = df[df["spacecraft"] == sc]
            for col in ("dT_syn_weather", "T_active_K"):
                y = pd.to_numeric(g[col], errors="coerce")
                pre = y[(g.date >= t0 - pd.Timedelta(days=w)) & (g.date < t0)].dropna()
                post = y[(g.date >= t0) & (g.date <= t0 + pd.Timedelta(days=w))].dropna()
                if len(pre) >= 2 and len(post) >= 2:
                    rows.append(dict(site=site, onset=onset, window_d=w, sensor=SC_MAP[sc],
                                     estimator=col, n_pre=len(pre), n_post=len(post),
                                     delta_single_sensor=round(post.median() - pre.median(), 2)))
    return pd.DataFrame(rows)


def sensor_regression(dfs, dTs):
    """Quiet-period dT_syn_weather ~ sensor dummies (ref ETM+) + annual harmonic, HC3 SE.
    Also the same with lake-minus-background (raw)."""
    import numpy.linalg as LA
    rows = []
    for site in SITES:
        df = dfs[site].merge(dTs[site][["datetime", "spacecraft", "bg_active_K", "T_raw_act"]]
                             .drop_duplicates("datetime"), on="datetime", how="left")
        df = df[df["_quiet"]].copy()
        df["lake_minus_bg_raw"] = df["T_raw_act"] - df["bg_active_K"]
        doy = df["date"].dt.dayofyear.to_numpy(float)
        for col in ("dT_syn_weather", "lake_minus_bg_raw"):
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            scs = [s for s in sorted(df["spacecraft"].dropna().unique()) if s != "LANDSAT_7"]
            X = np.column_stack([np.ones(len(df)), np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)]
                                + [(df["spacecraft"] == s).to_numpy(float) for s in scs])
            ok = np.isfinite(y)
            X, yy = X[ok], y[ok]
            b = LA.lstsq(X, yy, rcond=None)[0]
            r = yy - X @ b
            XtXi = LA.inv(X.T @ X)
            h = np.einsum("ij,jk,ik->i", X, XtXi, X)
            V = XtXi @ (X.T * (r / (1 - h)) ** 2) @ X @ XtXi
            se = np.sqrt(np.diag(V))
            for k, s in enumerate(scs):
                j = 3 + k
                rows.append(dict(site=site, quantity=col, sensor=SC_MAP[s], reference="ETM+",
                                 n=int(ok.sum()), n_sensor=int(X[:, j].sum()),
                                 offset_K=round(b[j], 2), se_hc3=round(se[j], 2),
                                 ci95=f"{b[j]-1.96*se[j]:+.2f} to {b[j]+1.96*se[j]:+.2f}"))
    return pd.DataFrame(rows)


def l7_drift_check(dfs, dTs):
    """Ruapehu: ETM+ minus TIRS near-coincident (<= 8 d) differences before/after the 2022 L7 orbit lowering."""
    rows = []
    for site in SITES:
        df = dfs[site].merge(dTs[site][["datetime", "spacecraft", "bg_active_K", "T_raw_act"]]
                             .drop_duplicates("datetime"), on="datetime", how="left")
        df["lake_minus_bg_raw"] = df["T_raw_act"] - df["bg_active_K"]
        df["hour"] = pd.to_datetime(df["datetime"], utc=True).dt.hour + pd.to_datetime(df["datetime"], utc=True).dt.minute/60
        for sc, g in df.groupby("spacecraft"):
            for per, gg in (("<2022-04", g[g.date < "2022-04-01"]), (">=2022-04", g[g.date >= "2022-04-01"])):
                if len(gg):
                    rows.append(dict(site=site, spacecraft=SC_MAP[sc], period=per, n=len(gg),
                                     utc_hour_median=round(gg["hour"].median(), 2),
                                     dT_syn_weather_median=round(pd.to_numeric(gg["dT_syn_weather"], errors="coerce").median(), 2),
                                     lake_minus_bg_raw_median=round(gg["lake_minus_bg_raw"].median(), 2)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="unmixing_psf_"))
    print("sanity unmix reproduction (max |dT| K):",
          {s: round(sanity_unmix(s), 6) for s in SITES})

    fref = f_ref_table()
    fref.to_csv(OUT / "f_ref_by_sensor_scenario.csv", index=False)
    print(fref[fref.lake != "taal_lake"].pivot_table(index=["site", "scenario"], columns="spacecraft",
                                                    values="f_ref_new").round(3))

    # ---- baseline reproduction + scenarios -------------------------------
    all_t5, pass_rows, dfs, dTs = [], [], {}, {}
    paper = pd.read_csv(S.RECORDS_DIR / "10_post_eruption" / "per_eruption_delta_paper.csv")
    scen = [("baseline_repro", {}), ("no_unmix", {}), ("lambda_sensor", {})] + \
           [(s, dict(fref_tab=fref)) for s in PSF_SCN[1:]] + \
           [(s + "|same_scenes", dict(fref_tab=fref, hold_screen=True)) for s in ("native_box_scaled", "common_tm_worst")]
    for site in SITES:
        for tag, kw in scen:
            name = tag.split("|")[0]
            sub = tmp / f"{site}_{tag.replace('|', '_')}"
            sub.mkdir(exist_ok=True)
            d = scenario_deltaT(site, name, **kw)
            if tag == "baseline_repro":
                dTs[site] = d
            ev, df, meta = run_s10(site, d, S10.F_MIN, sub)
            if tag == "baseline_repro":
                dfs[site] = df
            all_t5.append(pick_t5(ev, tag, site))
            ok = d["qc_flag"].eq("ok")
            for sc, g in d[ok].groupby("spacecraft"):
                f = pd.to_numeric(g["f_est_act"], errors="coerce")
                pass_rows.append(dict(site=site, scenario=tag, spacecraft=sc, n_ok=len(g),
                                      pass_070=int((f >= 0.70).sum()),
                                      share_070=round(float((f >= 0.70).mean()), 3),
                                      pass_050=int((f >= 0.50).sum()),
                                      f_median=round(float(f.median()), 3)))
            print(site, tag, "done")
    t5 = pd.concat(all_t5, ignore_index=True)
    # check that the baseline scenario reproduces the primary Stage 10 records
    pk = paper.rename(columns={"delta": "paper_delta", "p_placebo_year": "paper_p"})
    chk = t5[t5.scenario == "baseline_repro"].merge(
        pk[["site", "onset", "window_d", "estimator", "paper_delta", "paper_p"]],
        on=["site", "onset", "window_d", "estimator"], how="left")
    chk["abs_diff"] = (chk["delta"] - chk["paper_delta"]).abs()
    chk.to_csv(OUT / "baseline_reproduction.csv", index=False)
    print("baseline repro max |diff|:", chk["abs_diff"].max())

    t5.to_csv(OUT / "event_contrasts_by_scenario_long.csv", index=False)
    wide = t5.pivot_table(index=["site", "onset", "window_d", "estimator"], columns="scenario",
                          values="delta").round(2)
    base = wide["baseline_repro"]
    for c in wide.columns:
        if c != "baseline_repro":
            wide[f"chg_{c}"] = (wide[c] - base).round(2)
    wide.reset_index().to_csv(OUT / "event_contrasts_by_scenario_wide.csv", index=False)
    print(wide)
    rk = t5.pivot_table(index=["site", "onset", "window_d", "estimator"], columns="scenario",
                        values="max_rank", aggfunc="first")
    rk.reset_index().to_csv(OUT / "event_maxrank_by_scenario.csv", index=False)
    print(rk.loc[(slice(None), slice(None), slice(None), "dT_syn_weather"), :])
    pd.DataFrame(pass_rows).to_csv(OUT / "f_screen_pass_by_scenario.csv", index=False)

    # ---- (c) sensor consistency ------------------------------------------
    by_sensor, overlap, pairs = sensor_consistency(dfs, dTs)
    by_sensor.to_csv(OUT / "quiet_by_sensor.csv", index=False)
    overlap.to_csv(OUT / "quiet_sensor_overlap_offsets.csv", index=False)
    pairs.to_csv(OUT / "quiet_near_coincident_pairs.csv", index=False)
    within_sensor_event_deltas(dfs, dTs).to_csv(OUT / "event_contrasts_single_sensor.csv", index=False)
    window_sensor_mix(dfs, dTs).to_csv(OUT / "event_window_sensor_mix.csv", index=False)
    sensor_regression(dfs, dTs).to_csv(OUT / "quiet_sensor_offset_regression.csv", index=False)
    l7_drift_check(dfs, dTs).to_csv(OUT / "sensor_period_overpass_check.csv", index=False)
    sa = santa_ana_within_sensor(dfs["santa_ana"], dTs["santa_ana"])
    sa.to_csv(OUT / "santa_ana_within_sensor.csv", index=False)
    # unmixing correction magnitude by sensor/era (retained scenes, f >= 0.70)
    um = []
    for site in SITES:
        d = dTs[site]
        d = d[d.qc_flag.eq("ok") & (pd.to_numeric(d.f_est_act, errors="coerce") >= 0.70)]
        for (sc, src), g in d.groupby(["spacecraft", "f_source_act"]):
            v = g["unmix_delta_active_K"]
            um.append(dict(site=site, spacecraft=sc, f_source=src, n=len(g),
                           f_median=round(g.f_est_act.median(), 3),
                           share_f_lt1=round(float((g.f_est_act < 0.999).mean()), 3),
                           unmix_K_median=round(v.median(), 2),
                           unmix_K_p05=round(v.quantile(.05), 2), unmix_K_p95=round(v.quantile(.95), 2)))
    pd.DataFrame(um).to_csv(OUT / "unmix_correction_by_sensor_era.csv", index=False)
    shutil.rmtree(tmp, ignore_errors=True)
    print("outputs ->", OUT)


if __name__ == "__main__":
    main()
