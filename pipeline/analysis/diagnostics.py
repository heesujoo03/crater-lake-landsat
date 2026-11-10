# -*- coding: utf-8 -*-
"""Unique-date operating diagnostics, spatially matched mixing correction,
thermal-filter ablation and a symmetric null for the thermometry comparison.

Pipeline modules are imported and not modified; scratch copies of the processed
tables carry every variant, and all outputs go to results/12_diagnostics/. The
matched mixing correction reads cached per-pixel thermal samples from
results/12_diagnostics/samples_buf/.

    python -m pipeline.analysis.diagnostics thermometry
    python -m pipeline.analysis.diagnostics diagnostics
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import s10_post_eruption as P
from pipeline import s3b_synthetic_control as SC
from pipeline import s9b_event_stats as B9
from pipeline.config import settings as S
from pipeline.config.sites import get_site
from pipeline.analysis import placebo_design as PD
from pipeline.analysis import unmixing_psf_f as UF
from pipeline.analysis import validation_energy as VE
from pipeline.s3_delta_t import local_robust_z, planck_radiance, planck_temperature

warnings.filterwarnings("ignore")
OUT = S.RECORDS_DIR / "12_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "samples_buf").mkdir(exist_ok=True)
SITES = ("taal", "santa_ana", "ruapehu")
EST = P.PRIMARY
F_FLOOR = 0.15
SEED = 20261102


# ---------------------------------------------------------------------------
#  1. thermal samples (cached)
# ---------------------------------------------------------------------------
def sample_scene_ids(site):
    """Every scene that survives or is rejected only by the local-z filter."""
    d = pd.read_csv(S.PROCESSED_DIR / f"{site}_deltaT.csv")
    return d[d.qc_flag.isin(["ok", "outlier"])]


def load_samples(site):
    rows = []
    for fp in sorted((OUT / "samples_buf" / site).glob("chunk_*.pkl")):
        rows += pd.read_pickle(fp)
    return {r["scene_id"]: r for r in rows}


# ---------------------------------------------------------------------------
#  2. matched correction
# ---------------------------------------------------------------------------
def f_interpolator(site, sigma_m=58.0, box_m=100.0):
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    from shapely.geometry import shape
    import json
    lake = get_site(site).active.key
    gj = json.loads((UF.AOI_DIR / f"{site}_{lake}.geojson").read_text(encoding="utf-8"))
    c = shape(gj["geometry"]).centroid
    tr = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m",
                              always_xy=True)
    xx, yy, f, _aoi = UF.f_field(site, lake, sigma_m, box_m)
    it = RegularGridInterpolator((yy[:, 0], xx[0, :]), f, bounds_error=False, fill_value=np.nan)

    def fn(lon, lat):
        x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
        return it(np.c_[y, x])
    return fn


def weighted_median(v, w):
    """Area-weighted median, the estimator of the recorded polygon medians."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def cell_weights(site, s):
    """Fraction of each 30 m cell inside the sampling polygon, in the scene UTM."""
    import json
    from pyproj import Transformer
    from shapely.geometry import box, shape
    from shapely.ops import transform
    lake = get_site(site).active.key
    gj = json.loads((UF.AOI_DIR / f"{site}_{lake}.geojson").read_text(encoding="utf-8"))
    tr = Transformer.from_crs("EPSG:4326", s["crs"], always_xy=True)
    G = transform(tr.transform, shape(gj["geometry"]))
    x, y = tr.transform(np.asarray(s["lon"], float), np.asarray(s["lat"], float))
    return np.array([box(a - 15, b - 15, a + 15, b + 15).intersection(G).area / 900.0
                     for a, b in zip(x, y)])


def f_constant(site):
    """Assigned scene fraction before the Sentinel-2 area scaling (geometric f)."""
    lst = pd.read_csv(S.INTERIM_DIR / f"{site}_lst.csv")
    a = lst[lst.lake == get_site(site).active.key]
    return float(pd.to_numeric(a.f_max, errors="coerce").median())


def matched_scene_table(site, sigma_m=58.0):
    """One row per scene: matched-pair corrected temperature and support."""
    d = sample_scene_ids(site).copy()
    smp = load_samples(site)
    fn = f_interpolator(site, sigma_m)
    f0 = f_constant(site)
    lst = pd.read_csv(S.INTERIM_DIR / f"{site}_lst.csv")
    lst = lst[lst.lake == get_site(site).active.key].set_index("scene_id")
    rows = []
    for r in d.itertuples():
        s = smp.get(r.scene_id)
        rec = dict(scene_id=r.scene_id, date=r.date, qc_flag=r.qc_flag)
        ft = pd.to_numeric(r.f_est_act, errors="coerce")
        if s is None or s.get("st") is None or not len(s["st"]):
            rows.append(rec)
            continue
        wt = cell_weights(site, s)
        keep = wt > 0
        if not keep.any():
            rows.append(rec)
            continue
        wt = wt[keep]
        st = np.asarray(s["st"], float)[keep]
        fi0 = fn(np.asarray(s["lon"], float)[keep], np.asarray(s["lat"], float)[keep])
        rt = ft / f0 if np.isfinite(ft) and f0 > 0 else 1.0
        fi = np.minimum(fi0 * rt, 1.0)
        Lb = float(planck_radiance(r.bg_active_K)) if np.isfinite(r.bg_active_K) else np.nan
        Lp = planck_radiance(st)
        ok = np.isfinite(fi) & (fi >= F_FLOOR) & np.isfinite(Lb) & np.isfinite(Lp)
        Lw = np.where(ok, (Lp - (1 - fi) * Lb) / np.where(ok, fi, 1.0), np.nan)
        nonpos = ok & ~(Lw > 0)
        Tw = np.where(ok & ~nonpos, planck_temperature(np.where(Lw > 0, Lw, np.nan)), np.nan)
        hi = np.isfinite(Tw) & (fi >= P.F_MIN)
        raw_med = weighted_median(st, wt)
        good = np.isfinite(Tw)
        T_scene_old = float(r.T_active_K) if np.isfinite(r.T_active_K) else np.nan
        rec.update(
            n_samples=len(st), area_cells=float(wt.sum()), n_valid_pairs=int(good.sum()), n_nonpositive=int(nonpos.sum()),
            n_f70=int(hi.sum()), raw_median_K=raw_med,
            raw_median_rec_K=float(lst.loc[r.scene_id, "lst_water_median_K"]) if r.scene_id in lst.index else np.nan,
            f_scene=ft, f_sample_med=weighted_median(fi[np.isfinite(fi)], wt[np.isfinite(fi)]),
            f_sample_min=float(np.nanmin(fi)),
            f_sample_max=float(np.nanmax(fi)), T_old_K=T_scene_old,
            T_match_K=weighted_median(Tw[good], wt[good]) if good.any() else np.nan,
            T_match_f70_K=weighted_median(Tw[hi], wt[hi]) if hi.sum() >= 3 else np.nan,
            n_cells=s.get("n_cells"), n_qa_clear=s.get("n_qa_clear"), n_st_valid=s.get("n_st_valid"),
            n_clear_st=s.get("n_clear_st"), n_clear_st_water=s.get("n_clear_st_water"))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  3. record variants (scratch processed directory)
# ---------------------------------------------------------------------------
class Scratch:
    def __init__(self, site, deltaT, restore_met=False):
        self.site, self.d, self.restore = site, deltaT, restore_met

    def __enter__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="crit_"))
        src = S.PROCESSED_DIR
        for f in (f"{self.site}_dataset.csv", f"{self.site}_energy_balance.csv"):
            if (src / f).exists():
                shutil.copy(src / f, self.tmp / f)
        self.d.to_csv(self.tmp / f"{self.site}_deltaT.csv", index=False)
        if self.restore:     # scenes the filter removed may lack a dataset row
            ds = pd.read_csv(src / f"{self.site}_dataset.csv")
            met = pd.read_csv(src / f"{self.site}_met.csv")
            miss = met[~met.date.isin(ds.date)].drop_duplicates("date")
            cols = ["date"] + [c for c in B9.MET_COLS if c in met.columns]
            pd.concat([ds, miss[cols]], ignore_index=True).to_csv(
                self.tmp / f"{self.site}_dataset.csv", index=False)
        self.old = S.PROCESSED_DIR
        S.PROCESSED_DIR = self.tmp
        return self

    def __exit__(self, *a):
        S.PROCESSED_DIR = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)


def collapse(df):
    g = df.groupby("date")
    return pd.DataFrame({"date": list(g.groups.keys()), EST: g[EST].median().to_numpy(),
                         "T_active_K": g["T_active_K"].median().to_numpy(),
                         "_quiet": g["_quiet"].max().to_numpy(),
                         "n_dup": g.size().to_numpy()}).sort_values("date").reset_index(drop=True)


def dedupe_fit_input(d):
    """Collapse same-date products before the reference and meteorological fits."""
    d = d.copy()
    d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
    d = d[d["qc_flag"].eq("ok")]
    d = d[pd.to_numeric(d["f_est_act"], errors="coerce") >= P.F_MIN]
    d["_day"] = d["t"].dt.tz_localize(None).dt.normalize()
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    agg = {c: "median" for c in num}
    agg.update({c: "first" for c in d.columns if c not in num and c != "_day"})
    d = d.sort_values("t").groupby("_day", as_index=False).agg(agg)
    return d.drop(columns=["t", "_day"], errors="ignore")


def load_variant(site, deltaT, restore=False):
    with Scratch(site, deltaT, restore):
        df, ke, buf = P.load(site, P.F_MIN)
    df = df.copy().sort_values("date").reset_index(drop=True)
    return df, ke


def base_deltaT(site):
    return pd.read_csv(S.PROCESSED_DIR / f"{site}_deltaT.csv")


def restore_filter(d):
    """qc_flag 'outlier' -> 'ok' only for scenes rejected by the local z alone."""
    d = d.copy()
    z = pd.to_numeric(d.dT_local_z, errors="coerce")
    n = pd.to_numeric(d.T_active_n, errors="coerce")
    m = d.qc_flag.eq("outlier") & (z.abs() > S.DELTA_T_OUTLIER_Z) & d.dT_K.notna() & (n >= 3)
    d.loc[m, "qc_flag"] = "ok"
    return d, d[m]


def apply_matched(site, d, tab, col):
    d = d.copy()
    t = tab.set_index("scene_id")[col]
    new = pd.to_numeric(d.scene_id.map(t), errors="coerce")
    old = pd.to_numeric(d.T_active_K, errors="coerce")
    has = new.notna()
    d["T_active_K"] = np.where(has, new, old)
    d["dT_K"] = pd.to_numeric(d.dT_K, errors="coerce") + (d["T_active_K"] - old)
    d["unmix_delta_active_K"] = pd.to_numeric(d.unmix_delta_active_K, errors="coerce") + (d["T_active_K"] - old)
    return d, ~has


def shares_obs(dn_sorted, c0, cs, w):
    """True where the comparator window [c-w, c+w] shares a retained date with the target."""
    lo = np.maximum(c0 - w, cs - w)
    hi = np.minimum(c0 + w, cs + w)
    a = np.searchsorted(dn_sorted, lo, side="left")
    b = np.searchsorted(dn_sorted, hi, side="right")
    return (hi >= lo) & (b > a)


def operating(yp, dnum, w, obs, fixed_amps, indep=False, rules=("max", "q95"),
              steps=np.arange(0, 60.01, 0.1), drop_year=None):
    """Leave-one-year injection. indep=True drops comparators sharing a retained date."""
    v = np.asarray(yp["v"], float)
    c = np.asarray(yp["c"])
    idx = np.arange(len(v))
    if drop_year is not None:
        idx = idx[yp["yr"] != drop_year]
    dn = np.sort(np.unique(dnum))
    trials = []
    for i in idx:
        comps = idx[idx != i]
        if indep:
            comps = comps[~shares_obs(dn, c[i], c[comps], w)]
        if len(comps) >= P.MIN_YEARS:
            trials.append((i, comps))
    out = dict(n_trials=len(trials), n_targets_all=len(idx),
               comp_min=min((len(t[1]) for t in trials), default=0),
               comp_max=max((len(t[1]) for t in trials), default=0))
    if not trials:
        return out
    for rule in rules:
        thrs = []
        for i, comps in trials:
            null = v[comps]; cc = np.median(null); dev = np.abs(null - cc)
            thrs.append((v[i], cc, dev.max() if rule == "max" else np.quantile(dev, 0.95)))
        T = np.asarray(thrs)
        rate = lambda s: float(np.mean(np.abs(T[:, 0] + s - T[:, 1]) > T[:, 2]))
        out[f"power_obs_{rule}"] = rate(obs)
        out[f"hits_obs_{rule}"] = int(round(rate(obs) * len(T)))
        for nm, a in fixed_amps.items():
            out[f"power_{nm}_{rule}"] = rate(a)
        for sign, tag in ((1, "pos"), (-1, "neg")):
            found = np.nan
            for s in steps:
                if rate(sign * s) >= 0.8:
                    found = round(float(s), 1)
                    break
            out[f"mds80_{rule}_{tag}"] = found
    return out


def alert_table(df, ke, w):
    """Per-anchor alert flags: all-other-years (retro), earlier-only (pro), and
    retro with comparators sharing a retained date removed (indep)."""
    dnum, y = PD.arrays(df)
    dn = np.sort(np.unique(dnum[np.isfinite(y)]))
    eps = PD.anchors(ke, w, dnum, "post")
    C = PD.candidates(dnum, y, [e["ref"] for e in eps], w)
    rows = []
    for i in range(len(C["c"])):
        rec = dict(anchor=int(C["c"][i]), yr=int(C["yr"][i]))
        doy0, yr0 = C["doy"][i], C["yr"][i]
        share = shares_obs(dn, C["c"][i], C["c"], w)
        for tag, m in (("retro", C["yr"] != yr0), ("pro", C["yr"] < yr0),
                       ("indep", (C["yr"] != yr0) & ~share)):
            g, yp, fell = PD.matched(PD._sub(C, m), doy0, C["npre"][i], C["npost"][i], "baseline")
            if fell or len(yp["v"]) < P.MIN_YEARS:
                rec[f"{tag}_ok"] = False
                continue
            cc = np.median(yp["v"]); dev = np.abs(yp["v"] - cc); x = abs(C["v"][i] - cc)
            rec[f"{tag}_ok"] = True
            rec[f"{tag}_n"] = len(yp["v"])
            rec[f"{tag}_max"] = bool(x > dev.max())
            rec[f"{tag}_q95"] = bool(x > np.quantile(dev, 0.95))
        rows.append(rec)
    return pd.DataFrame(rows)


def alert_summary(t):
    out = {}
    for tag in ("retro", "pro", "indep"):
        k = t[t[f"{tag}_ok"].fillna(False).astype(bool)]
        out[f"{tag}_n"] = len(k)
        for rule in ("max", "q95"):
            a = k[f"{tag}_{rule}"].astype(bool)
            out[f"{tag}_{rule}_alerts"] = int(a.sum())
            out[f"{tag}_{rule}_rate"] = float(a.mean()) if len(k) else np.nan
            yr = k.assign(a=a).groupby("yr")["a"].mean()
            out[f"{tag}_{rule}_annual_mean"] = float(yr.mean()) if len(yr) else np.nan
        out[f"{tag}_n_years"] = int(k.yr.nunique())
        if len(k):
            out[f"{tag}_comp_min"] = int(k[f"{tag}_n"].min())
            out[f"{tag}_comp_med"] = float(k[f"{tag}_n"].median())
            out[f"{tag}_comp_max"] = int(k[f"{tag}_n"].max())
    return out


def diagnose(site, df, ke, label, base_amp=None, alerts=True):
    ev, _, store = PD.evaluate(site, df, ke, label, extra_pools=False)
    ev = ev[ev.analysis.eq("post")].copy()
    dnum, _y = PD.arrays(df)
    rows, inj = [], []
    for r in ev.itertuples():
        key = (site, "post", r.window_d, r.onset, label)
        rec = dict(variant=label, site=site, window_d=r.window_d, onset=r.onset,
                   n_pre=r.n_pre, n_post=r.n_post, d_pre=r.n_pre_unique_dates, d_post=r.n_post_unique_dates,
                   tested=r.tested, delta=r.delta)
        if r.tested and key in store:
            yp, g, d = store[key]
            rec.update(ny=len(yp["v"]), n_grid=len(g["v"]), p_rank=r.p_rank, q025=r.q025, q975=r.q975,
                       max_exceedance=r.max_exceedance, fell_back=r.baseline_fell_back_to_season_only,
                       placebo_min=float(np.min(yp["v"])) if len(yp["v"]) else np.nan,
                       placebo_max=float(np.max(yp["v"])) if len(yp["v"]) else np.nan,
                       pool_years=r.pool_years)
            if len(yp["v"]) >= P.MIN_YEARS + 1:
                amps = {}
                if base_amp is not None and (site, r.window_d, r.onset) in base_amp:
                    amps["baseAmp"] = base_amp[(site, r.window_d, r.onset)]
                for flag in (False, True):
                    o = operating(yp, dnum, r.window_d, float(d), amps, indep=flag)
                    pre = "indep_" if flag else ""
                    rec.update({pre + k: v for k, v in o.items()})
                # comparator-year removal range
                mv = []
                for u in np.unique(yp["yr"]):
                    o = operating(yp, dnum, r.window_d, float(d), {}, rules=("max",), drop_year=u)
                    mv.append((o.get("mds80_max_pos", np.nan), o.get("power_obs_max", np.nan)))
                mv = np.asarray(mv, float)
                rec.update(loo_mds_lo=np.nanmin(mv[:, 0]) if np.isfinite(mv[:, 0]).any() else np.nan,
                           loo_mds_hi=np.nanmax(mv[:, 0]) if np.isfinite(mv[:, 0]).any() else np.nan,
                           loo_mds_nan=int(np.isnan(mv[:, 0]).sum()),
                           loo_pow_lo=np.nanmin(mv[:, 1]), loo_pow_hi=np.nanmax(mv[:, 1]))
        rows.append(rec)
    al = []
    tabs = {}
    if alerts:
        for w in (180, 365):
            t = alert_table(df, ke, w)
            tabs[w] = t
            al.append(dict(variant=label, site=site, window_d=w, n_candidates=len(t), **alert_summary(t)))
    return pd.DataFrame(rows), pd.DataFrame(al), tabs


def common_alerts(ta, tb):
    m = ta.merge(tb, on="anchor", suffixes=("_a", "_b"))
    rows = {}
    for tag in ("retro",):
        k = m[m[f"{tag}_ok_a"].fillna(False).astype(bool) & m[f"{tag}_ok_b"].fillna(False).astype(bool)]
        for rule in ("max", "q95"):
            rows[f"common_{tag}_{rule}_n"] = len(k)
            rows[f"common_{tag}_{rule}_a"] = float(k[f"{tag}_{rule}_a"].astype(bool).mean()) if len(k) else np.nan
            rows[f"common_{tag}_{rule}_b"] = float(k[f"{tag}_{rule}_b"].astype(bool).mean()) if len(k) else np.nan
    return rows


def run_diagnostics():
    all_ev, all_al, comm, abl = [], [], [], []
    # check: reproduce the product-based operating values of placebo_design
    pub = pd.read_csv(S.RECORDS_DIR / "11_analysis" / "placebo_design" / "power_injection.csv")
    pub = pub[pub.analysis.eq("post")]
    match_rows = []
    for site in SITES:
        d0 = base_deltaT(site)
        tab = matched_scene_table(site)
        tab.to_csv(OUT / f"matched_scenes_{site}.csv", index=False)

        base, ke = load_variant(site, d0)
        tabs_store = {}
        ev0, al0, t0 = diagnose(site, base, ke, "product")
        base_amp = {(site, r.window_d, r.onset): r.delta for r in ev0.itertuples() if r.tested}
        all_ev.append(ev0); all_al.append(al0); tabs_store["product"] = t0

        # 1. unique-date, fixed model and full refit
        fx = collapse(base)
        e, a, t = diagnose(site, fx, ke, "date_fixed", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["date_fixed"] = t
        with Scratch(site, dedupe_fit_input(d0)):
            rf, ke_, _b = P.load(site, P.F_MIN)
        rf = rf.sort_values("date").reset_index(drop=True)
        e, a, t = diagnose(site, rf, ke, "date_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["date_refit"] = t

        # 3. thermal-filter ablation (full refit)
        dn, rest = restore_filter(d0)
        rest.assign(site=site)[["site", "scene_id", "date", "spacecraft", "T_active_K", "bg_active_K",
                                 "T_control_K", "dT_K", "dT_local_z", "f_est_act"]].to_csv(
            OUT / f"filter_restored_{site}.csv", index=False)
        nf, _ = load_variant(site, dn, restore=True)
        e, a, t = diagnose(site, nf, ke, "nofilter_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["nofilter_refit"] = t
        with Scratch(site, dedupe_fit_input(dn), True):
            nfd, _k, _b = P.load(site, P.F_MIN)
        nfd = nfd.sort_values("date").reset_index(drop=True)
        e, a, t = diagnose(site, nfd, ke, "nofilter_date_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["nofilter_date_refit"] = t
        abl.append(dict(site=site, n_restored_total=len(rest),
                        n_retained_base=len(base), n_retained_nofilter=len(nf)))

        # 2. matched correction
        dm, miss = apply_matched(site, d0, tab, "T_match_K")
        mm, _ = load_variant(site, dm)
        # same scenes, fixed reference and meteorological model
        y_old = base.set_index("datetime")[EST]
        T_old = base.set_index("datetime")["T_active_K"]
        fxm = mm[["datetime", "date", "_quiet", "T_active_K"]].copy()
        fxm[EST] = fxm.datetime.map(y_old) + (fxm.T_active_K - fxm.datetime.map(T_old))
        e, a, t = diagnose(site, fxm, ke, "matched_fixed", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["matched_fixed"] = t
        e, a, t = diagnose(site, mm, ke, "matched_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["matched_refit"] = t
        cmp_ = base[["datetime", "date", "T_active_K", EST]].merge(
            mm[["datetime", "T_active_K", EST]], on="datetime", suffixes=("_old", "_new"))
        cmp_["dT"] = cmp_.T_active_K_new - cmp_.T_active_K_old
        cmp_["dy"] = cmp_[EST + "_new"] - cmp_[EST + "_old"]
        cmp_.to_csv(OUT / f"matched_vs_maxf_{site}.csv", index=False)

        # sample-level f >= 0.70 (changed eligibility) with re-evaluated local z
        de, _ = apply_matched(site, d0, tab, "T_match_f70_K")
        no_pair = d0.scene_id.map(tab.set_index("scene_id")["T_match_f70_K"]).isna()
        de.loc[no_pair & de.qc_flag.eq("ok"), "qc_flag"] = "low_pixels"
        z, _w = local_robust_z(de["datetime"], pd.to_numeric(de["dT_K"], errors="coerce"))
        de.loc[de.qc_flag.eq("ok") & (np.abs(np.nan_to_num(z)) > S.DELTA_T_OUTLIER_Z), "qc_flag"] = "outlier"
        me, _ = load_variant(site, de)
        e, a, t = diagnose(site, me, ke, "matched_f70_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["matched_f70_refit"] = t

        # 1+2 together, full refit
        with Scratch(site, dedupe_fit_input(dm)):
            md, _k, _b = P.load(site, P.F_MIN)
        md = md.sort_values("date").reset_index(drop=True)
        e, a, t = diagnose(site, md, ke, "matched_date_refit", base_amp)
        all_ev.append(e); all_al.append(a); tabs_store["matched_date_refit"] = t

        # PSF width on the matched correction, event contrasts only
        for sg in (40.6, 75.4):
            ts = matched_scene_table(site, sg)
            ds_, _m = apply_matched(site, d0, ts, "T_match_K")
            mx, _ = load_variant(site, ds_)
            e, _a, _t = diagnose(site, mx, ke, f"matched_sigma{sg}", base_amp, alerts=False)
            all_ev.append(e)

        for v, t in tabs_store.items():
            if v == "product":
                continue
            for w in (180, 365):
                comm.append(dict(site=site, variant=v, window_d=w, **common_alerts(t0[w], t[w])))
        print(site, "diagnostics done", flush=True)
        pd.concat(all_ev, ignore_index=True).to_csv(OUT / "event_diagnostics.csv", index=False)
        pd.concat(all_al, ignore_index=True).to_csv(OUT / "alert_rates.csv", index=False)
        pd.DataFrame(comm).to_csv(OUT / "alert_rates_common_targets.csv", index=False)
    pd.DataFrame(abl).to_csv(OUT / "filter_ablation_counts.csv", index=False)
    ev = pd.concat(all_ev, ignore_index=True)
    a0 = ev[ev.variant.eq("product") & ev.tested.eq(True)].merge(pub, on=["site", "window_d", "onset"])
    for r in a0.itertuples():
        match_rows.append(dict(site=r.site, window_d=r.window_d, onset=r.onset,
                               delta=r.delta_x, delta_pub=r.delta_y,
                               mds_pos=r.mds80_max_pos, mds_pos_pub=r.mds80_max_pos_K,
                               mds_neg=r.mds80_max_neg, mds_neg_pub=r.mds80_max_neg_K,
                               power=r.power_obs_max, power_pub=r.power_at_observed_delta_max))
    pd.DataFrame(match_rows).to_csv(OUT / "reproduction_operating.csv", index=False)


# ---------------------------------------------------------------------------
#  5. thermometry: identical fitting for observed and shifted labels
# ---------------------------------------------------------------------------
class RuapehuFit:
    """Replicates s3b._core + s9b.frame for an arbitrary non-heating mask."""

    def __init__(self, f_min=P.F_MIN):
        d, _o, _q, _u, _m = SC._core("ruapehu", [dict(date="1980-01-01", start="1980-01-01",
                                                      end="1980-01-08", label="d", severity="unrest",
                                                      kind="dummy")], 0, f_min)
        self.d = d
        self.t = d["t"]
        self.date = pd.to_datetime(d["t"], utc=True).dt.tz_localize(None).dt.normalize()
        self.dnum = self.date.to_numpy("datetime64[D]").astype("int64")
        self.y = pd.to_numeric(d["T_active_K"], errors="coerce").to_numpy(float)
        self.ctl = pd.to_numeric(d["T_control_K"], errors="coerce").to_numpy(float)
        self.bg = pd.to_numeric(d["bg_active_K"], errors="coerce").to_numpy(float)
        self.dow = SC._doy_terms(d["t"])
        self.pool = list(SC._donor_pool("ruapehu", d["t"]).values())
        ds = pd.read_csv(S.PROCESSED_DIR / "ruapehu_dataset.csv")
        ds["date"] = pd.to_datetime(ds["date"], errors="coerce").dt.normalize()
        keep = ["date"] + [c for c in B9.MET_COLS if c in ds.columns]
        m = pd.DataFrame({"date": self.date.to_numpy()}).merge(ds[keep].drop_duplicates("date"), on="date", how="left")
        met = [c for c in B9.MET_COLS if c in m.columns and pd.to_numeric(m[c], errors="coerce").notna().mean() > 0.7]
        self.X = m[met].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    def relative(self, nonheat, idx=None):
        """y(t) with every label-dependent component fitted on `nonheat`.
        idx (optional) resamples rows for the block bootstrap."""
        ii = np.arange(len(self.y)) if idx is None else idx
        y, ctl, bg, dow, X0 = self.y[ii], self.ctl[ii], self.bg[ii], self.dow[ii], self.X[ii].copy()
        q = nonheat[ii]
        fit = q & np.isfinite(y) & np.isfinite(ctl) & np.isfinite(bg)
        Xc = np.c_[np.ones(len(y)), dow]
        bc, *_ = np.linalg.lstsq(Xc[fit], y[fit], rcond=None)
        clim = Xc @ bc
        donors = [ctl, bg, clim] + [p[ii] for p in self.pool]
        D = np.column_stack(donors)
        keep, seen = [], []
        for i in range(D.shape[1]):
            col = D[:, i]
            if not np.isfinite(col[fit]).all():
                continue
            if any(np.allclose(col[fit], D[:, j][fit], atol=1e-9) for j in seen):
                continue
            seen.append(i); keep.append(i)
        D = D[:, keep]
        w = SC._simplex_fit(y[fit], D[fit])
        syn = y - D @ w
        ff = q & np.isfinite(syn)
        for j in range(X0.shape[1]):
            med = np.nanmedian(X0[ff, j]) if ff.any() else np.nanmedian(X0[:, j])
            X0[~np.isfinite(X0[:, j]), j] = med if np.isfinite(med) else 0.0
        Xd = np.c_[np.ones(len(y)), X0]
        good = ff & np.isfinite(Xd).all(axis=1)
        beta, *_ = np.linalg.lstsq(Xd[good], syn[good], rcond=None)
        return syn - Xd @ beta, ii


def in_windows(d, wins):
    m = np.zeros(len(d), bool)
    for a, b in wins:
        m |= (d >= a) & (d <= b)
    return m


def shifted_mask(d, wins, s, dmin, span):
    pos = ((d - dmin) - s) % span
    m = np.zeros(len(d), bool)
    for a, b in wins:
        r0, r1 = (a - dmin) % span, (b - dmin) % span
        m |= ((pos >= r0) & (pos <= r1)) if r0 <= r1 else ((pos >= r0) | (pos <= r1))
    return m


def contrast(y, m, min_b=3, min_a=5):
    ok = np.isfinite(y)
    b, a = y[m & ok], y[~m & ok]
    if len(b) < min_b or len(a) < min_a:
        return np.nan
    return float(np.median(b) - np.median(a))


def thermometry_case(R, wins, label, subset=None, n_boot=2000):
    """Observed, exact-enumeration null (all circular shifts) and block bootstrap,
    each under fixed y and under refitting. subset: rows evaluated AND fitted."""
    keep = np.ones(len(R.y), bool) if subset is None else subset
    idx = np.flatnonzero(keep)
    d = R.dnum[idx]
    dmin, span = int(d.min()), int(d.max() - d.min()) + 1
    wins = [(a, b) for a, b in wins if b >= d.min() and a <= d.max()]

    def fit_for(mask_sub):
        nh = np.zeros(len(R.y), bool)
        nh[idx] = ~mask_sub
        yy, _ = R.relative(nh, idx)
        return yy

    m0 = in_windows(d, wins)
    y0 = fit_for(m0)
    obs = contrast(y0, m0)
    rows = []
    t0 = time.time()
    for s in range(span):
        m = shifted_mask(d, wins, s, dmin, span)
        fx = contrast(y0, m)
        try:
            ys = fit_for(m)
            rf = contrast(ys, m)
        except Exception:                                           # noqa: BLE001
            rf = np.nan
        rows.append((s, int(m.sum()), int((~m).sum()), fx, rf))
    L = pd.DataFrame(rows, columns=["shift_d", "n_heating", "n_control", "stat_fixed_y", "stat_refit"])
    L.to_csv(OUT / f"thermometry_shift_ledger_{label}.csv", index=False)
    nz = L[L.shift_d != 0]
    pv = lambda c: float((np.sum(np.abs(nz[c].dropna()) >= abs(obs)) + 1) / (nz[c].notna().sum() + 1))
    pv1 = lambda c: float((np.sum(nz[c].dropna() >= obs) + 1) / (nz[c].notna().sum() + 1))
    # bootstrap
    blk = (d - d.min()) // 90
    ids, inv = np.unique(blk, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ids))]
    rng = np.random.default_rng(SEED)
    bf, br = [], []
    for _ in range(n_boot):
        pick = rng.integers(len(ids), size=len(ids))
        sel = np.concatenate([members[p] for p in pick])
        bf.append(contrast(y0[sel], m0[sel]))
        nh = np.zeros(len(R.y), bool); nh[idx] = ~m0
        try:
            yb, _ = R.relative(nh, idx[sel])
            br.append(contrast(yb, m0[sel]))
        except Exception:                                           # noqa: BLE001
            br.append(np.nan)
    per = []
    a_ref = np.median(y0[~m0 & np.isfinite(y0)])
    for a, b in wins:
        k = (d >= a) & (d <= b) & np.isfinite(y0)
        per.append(dict(start=str(np.datetime64(int(a), "D")), end=str(np.datetime64(int(b), "D")),
                        n_scenes=int(k.sum()),
                        contrast=float(np.median(y0[k]) - a_ref) if k.any() else np.nan))
    P_ = pd.DataFrame(per)
    P_.to_csv(OUT / f"thermometry_episodes_{label}.csv", index=False)
    w = P_[P_.n_scenes > 0]
    return dict(case=label, n_scenes=len(d), span_d=span, n_episodes=len(wins), n_ep_with_scenes=len(w),
                n_positive=int((w.contrast > 0).sum()), n_heating=int(m0.sum()), n_control=int((~m0).sum()),
                obs_K=obs, n_valid_shifts_fixed=int(nz.stat_fixed_y.notna().sum()),
                n_valid_shifts_refit=int(nz.stat_refit.notna().sum()),
                p_fixed=pv("stat_fixed_y"), p_refit=pv("stat_refit"),
                p1_fixed=pv1("stat_fixed_y"), p1_refit=pv1("stat_refit"),
                null_refit_q025=float(nz.stat_refit.quantile(.025)), null_refit_q975=float(nz.stat_refit.quantile(.975)),
                null_refit_sd=float(nz.stat_refit.std()), null_fixed_sd=float(nz.stat_fixed_y.std()),
                ci_fixed_lo=float(np.nanpercentile(bf, 2.5)), ci_fixed_hi=float(np.nanpercentile(bf, 97.5)),
                ci_refit_lo=float(np.nanpercentile(br, 2.5)), ci_refit_hi=float(np.nanpercentile(br, 97.5)),
                runtime_s=round(time.time() - t0, 1))


def run_thermometry():
    R = RuapehuFit()
    ev_clean = VE.heating_events_from(VE.ground_daily("clean"))
    ev_tele = VE.heating_events_from(VE.ground_daily("tele"))
    towin = lambda ev: [(int(np.datetime64(e["start"], "D").astype("int64")),
                         int(np.datetime64(e["end"], "D").astype("int64"))) for e in ev]
    # label validity: dates with a defined 60-day rise in the daily thermometer grid
    ins = VE.ground_daily("clean")
    ins.index = ins.index.tz_localize(None)
    grid = ins.resample("D").mean().interpolate(limit=30)
    rise = grid - grid.shift(60)
    lab = rise.reindex(R.date.to_numpy()).notna().to_numpy()
    rows = [thermometry_case(R, towin(ev_clean), "thermometer_all"),
            thermometry_case(R, towin(ev_clean), "thermometer_labelled", subset=lab)]
    era = R.dnum >= int(np.datetime64(VE.TELE_START, "D").astype("int64"))
    ev2 = [e for e in ev_tele if pd.Timestamp(e["end"]) >= pd.Timestamp(VE.TELE_START)]
    rows.append(thermometry_case(R, towin(ev2), "telemetry_era", subset=era))
    pd.DataFrame(rows).to_csv(OUT / "thermometry_symmetric.csv", index=False)
    # consistency check against the pipeline frame (s9b_event_stats.frame)
    df = B9.frame("ruapehu", ev_clean, 0, f_min=P.F_MIN)
    y0, _ = R.relative(~in_windows(R.dnum, towin(ev_clean)))
    print("max |y(frame) - y(refit)| =", float(np.nanmax(np.abs(pd.to_numeric(df[EST]).to_numpy() - y0))))
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("thermometry", "all"):
        run_thermometry()
    if what in ("diagnostics", "all"):
        run_diagnostics()
