# -*- coding: utf-8 -*-
"""Short-lag variability term, quiet-period variance and the observed timescale tau_obs.

Re-uses the Stage 10d functions (pipeline.s10_detectability) without modifying
them, checks that the re-implementation reproduces the primary series, then runs:
  (2) sigma_pair under common lag / same-sensor conditions (+ Santa Ana <1 d)
  (3) sensitivity of sigma_quiet and tau_obs to the short-lag term
  (4) time-block bootstrap of tau_obs (dependence preserving)
  (5) supplementary structure-function figure
  (6) geometry scaling with each site as the reference
  (7) Taal diagnosis (quiet segments, pairs per bin, SSE profile)

Outputs: results/11_analysis/variability_timescale/.

    python -m pipeline.analysis.variability_timescale
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import s10_detectability as D
from pipeline import s10_post_eruption as P
from pipeline.config.sites import get_site

OUT = Path("results/11_analysis/variability_timescale")
OUT.mkdir(parents=True, exist_ok=True)
NAME = {"taal": "Taal", "santa_ana": "Santa Ana", "ruapehu": "Ruapehu"}
COLOR = {"taal": "#2a78d6", "ruapehu": "#eb6834", "santa_ana": "#c2415a"}
GRID = np.geomspace(0.5, 600.0, 4000)          # same grid as Stage 10d
EDGES = np.unique(np.round(np.geomspace(1, D.SF_MAX_LAG_D, 22)).astype(int))
SENS = {"LANDSAT_5": "L5", "LANDSAT_7": "L7", "LANDSAT_8": "L8", "LANDSAT_9": "L9"}


# ---------------------------------------------------------------- data
def series(site):
    """Exactly D._series, plus the spacecraft and scene id of each row."""
    d, col = D._series(site)
    df, _, _ = P.load(site, P.F_MIN)
    raw = pd.read_csv(f"results/03_deltaT/{site}/deltaT.csv",
                      usecols=["datetime", "scene_id", "spacecraft"])
    m = df[["datetime", "date", col]].dropna().merge(raw, on="datetime", how="left")
    m["date"] = pd.to_datetime(m["date"])
    m = m.rename(columns={col: "y"})
    d = d.merge(m[["date", "y", "datetime", "scene_id", "spacecraft"]],
                on=["date", "y"], how="left")
    assert d["spacecraft"].notna().all()
    d["sensor"] = d["spacecraft"].map(SENS)
    return d, col


def quiet_segments(site, d, seg_min=D.SEG_MIN_SCENES, before=None):
    """Same rule as D._quiet_segments; optional seg_min / end-date override."""
    ev = [pd.Timestamp(e["start"]) for e in get_site(site).known_events
          if e.get("verified")]
    near = np.zeros(len(d), bool)
    for e in ev:
        near |= (np.abs((d["date"] - e).dt.days) <= P.PRIMARY_W).to_numpy()
    if before is not None:
        near |= (d["date"] >= pd.Timestamp(before)).to_numpy()
    segs, cur = [], []
    for i, bad in enumerate(near):
        if bad:
            if cur:
                segs.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    all_segs = [d.iloc[ix].copy() for ix in segs]
    kept = []
    for g in all_segs:
        if len(g) >= seg_min:
            g["y"] = g["y"] - g["y"].mean()
            kept.append(g)
    return kept, all_segs


# ---------------------------------------------------------------- pairs
def pairs_consecutive(d, lo, hi, same_sensor=False, sensor=None):
    """Stage-10d rule: consecutive scenes in the date-sorted series."""
    gap = d["date"].diff().dt.days.to_numpy()[1:]
    dy = np.diff(d["y"].to_numpy(float))
    s = d["sensor"].to_numpy()
    m = (gap >= lo) & (gap <= hi)
    if same_sensor:
        m &= s[1:] == s[:-1]
    if sensor is not None:
        m &= (s[1:] == sensor) & (s[:-1] == sensor)
    return dy[m]


def pairs_all(d, lo, hi, same_sensor=False, sensor=None):
    """All scene pairs with lo <= gap <= hi (not only consecutive)."""
    t = d["date"].to_numpy("datetime64[D]").astype("int64")
    y = d["y"].to_numpy(float); s = d["sensor"].to_numpy()
    i, j = np.triu_indices(len(d), 1)
    g = t[j] - t[i]
    m = (g >= lo) & (g <= hi)
    if same_sensor:
        m &= s[i] == s[j]
    if sensor is not None:
        m &= (s[i] == sensor) & (s[j] == sensor)
    return y[j][m] - y[i][m]


def sig(dy, nmin=5):
    return (float(np.std(dy, ddof=1) / np.sqrt(2)), len(dy)) if len(dy) >= nmin \
        else (np.nan, len(dy))


def sf_pairs(segs):
    lag, dif, t0 = [], [], []
    for g in segs:
        v = g["y"].to_numpy(float)
        t = g["date"].to_numpy("datetime64[D]").astype("int64")
        for i in range(len(v)):
            dt = t[i + 1:] - t[i]
            m = (dt >= 1) & (dt <= D.SF_MAX_LAG_D)
            if m.any():
                lag.append(dt[m]); dif.append(v[i + 1:][m] - v[i])
                t0.append(np.full(m.sum(), t[i]))
    if not lag:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(lag).astype(float), np.concatenate(dif).astype(float),
            np.concatenate(t0))


def binned(lag, dif, w=None):
    w = np.ones(len(lag)) if w is None else w
    rows = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (lag >= a) & (lag < b) & (w > 0)
        n = float(w[m].sum())
        if n >= 5:
            # weighted median lag / weighted mean squared difference
            ll, ww = lag[m], w[m]
            o = np.argsort(ll); cw = np.cumsum(ww[o])
            xmed = float(ll[o][np.searchsorted(cw, cw[-1] / 2.0)])
            rows.append((a, b, xmed, float(np.sum(ww * dif[m] ** 2) / n), n))
    return pd.DataFrame(rows, columns=["lag_lo_d", "lag_hi_d", "lag_median_d",
                                       "D_K2", "n_pairs"])


def fit(tb, s2m, nugget_free=False):
    """Stage-10d fit: plateau fixed to mean of the last three bins,
    intercept 2*s2m, pair-count-weighted SSE over GRID."""
    if len(tb) < 5:
        return dict(tau=np.nan, sill=np.nan, s2m=s2m, sse=None)
    xs, ds, ns = tb.lag_median_d.values, tb.D_K2.values, tb.n_pairs.values
    plateau = float(np.mean(ds[-3:]))
    if nugget_free:
        best = (np.inf, None, None)
        for c0 in np.linspace(0, plateau, 201):
            e = 1 - np.exp(-xs[None, :] / GRID[:, None])
            sse = np.sum(ns * (ds - (c0 + (plateau - c0) * e)) ** 2, axis=1)
            k = int(np.argmin(sse))
            if sse[k] < best[0]:
                best = (sse[k], GRID[k], c0)
        return dict(tau=float(best[1]), sill=plateau - best[2], s2m=best[2] / 2,
                    sse=None)
    sill = max(plateau - 2 * s2m, 1e-6)
    e = 1 - np.exp(-xs[None, :] / GRID[:, None])
    sse = np.sum(ns * (ds - (2 * s2m + sill * e)) ** 2, axis=1)
    return dict(tau=float(GRID[int(np.argmin(sse))]), sill=sill, s2m=s2m, sse=sse)


def lagbin_boot(tb, s2m, n=300, seed=11):
    """Lag-bin bootstrap of the Stage 10d scheme (sill fixed)."""
    xs, ds, ns = tb.lag_median_d.values, tb.D_K2.values, tb.n_pairs.values
    sill = max(float(np.mean(ds[-3:])) - 2 * s2m, 1e-6)
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        p = rng.integers(0, len(xs), len(xs))
        e = 1 - np.exp(-xs[p][None, :] / GRID[:, None])
        sse = np.sum(ns[p] * (ds[p] - (2 * s2m + sill * e)) ** 2, axis=1)
        out.append(GRID[int(np.argmin(sse))])
    return np.array(out)


def block_boot(lag, dif, t0, segs, s2m, block_d, n=1000, seed=7):
    """Time-block bootstrap. Each pair is attached to the calendar block
    (length block_d, restarting in every quiet segment) of its earlier scene;
    blocks are resampled with replacement and all their pairs enter with the
    block multiplicity. Pairs sharing a scene therefore stay together, the
    binned D, the plateau and the fit are all recomputed per replicate."""
    bid = np.full(len(lag), -1)
    k = 0
    for g in segs:
        t = g["date"].to_numpy("datetime64[D]").astype("int64")
        m = (t0 >= t.min()) & (t0 <= t.max())
        b = (t0[m] - t.min()) // block_d
        bid[m] = b + k
        k += int(b.max()) + 1 if m.any() else 0
    ub = np.unique(bid[bid >= 0])
    rng = np.random.default_rng(seed)
    taus, sq = [], []
    for _ in range(n):
        pick = rng.choice(ub, len(ub), replace=True)
        cnt = np.bincount(pick, minlength=bid.max() + 1)
        w = cnt[bid].astype(float)
        tb = binned(lag, dif, w)
        f = fit(tb, s2m)
        taus.append(f["tau"]); sq.append(np.sqrt(f["sill"] / 2) if np.isfinite(f["sill"]) else np.nan)
    return np.array(taus), np.array(sq), len(ub)


def q(a, p):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


# ---------------------------------------------------------------- main
def main():
    det = pd.read_csv("results/10_detectability/detectability_by_site.csv").set_index("site")
    repro, common, sens_rows, boot_rows, bins_all, seg_rows, geo_rows = [], [], [], [], [], [], []
    cache = {}
    for s in D.SITES:
        d, col = series(s)
        # ---- (1) reproduce baseline
        sm, npair, win, by, rep, nsame = D.noise_from_close_pairs(d[["date", "y"]])
        ts = D.observed_timescale(s, d[["date", "y"]], sm)
        segs, all_segs = quiet_segments(s, d)
        lag, dif, t0 = sf_pairs(segs)
        tb = binned(lag, dif)
        f0 = fit(tb, sm ** 2)
        lb = lagbin_boot(tb, sm ** 2)
        r = det.loc[s]
        repro.append(dict(site=s, sigma_pair_K=round(sm, 2), pub=r.sigma_meas_K,
                          n_pairs=npair, pub_n=r.n_close_pairs, window_d=win,
                          tau=round(f0["tau"], 1), pub_tau=r.tau_obs_d,
                          tau16=round(q(lb, 16), 1), tau84=round(q(lb, 84), 1),
                          pub_lo=r.tau_obs_lo_d, pub_hi=r.tau_obs_hi_d,
                          sigma_quiet=round(np.sqrt(f0["sill"] / 2), 2),
                          pub_sq=r.sigma_real_quiet_K, n_sf_pairs=len(lag),
                          pub_nsf=r.n_sf_pairs, same_day_rep=rep, n_same_day=nsame,
                          lagbin_tau_2p5=round(q(lb, 2.5), 1),
                          lagbin_tau_97p5=round(q(lb, 97.5), 1)))
        # ---- quiet segment table
        for g in all_segs:
            seg_rows.append(dict(site=s, start=g.date.min().date(), end=g.date.max().date(),
                                 n_scenes=len(g), used=len(g) >= D.SEG_MIN_SCENES,
                                 span_d=(g.date.max() - g.date.min()).days,
                                 sd_centred_K=round(float(g.y.std(ddof=1)), 2) if len(g) > 1 else np.nan))
        # ---- (2) common conditions
        for mode, fn in (("consecutive", pairs_consecutive), ("all_pairs", pairs_all)):
            for (lo, hi) in ((1, 3), (1, 8), (1, 16), (4, 8), (9, 16)):
                for ss in (False, True):
                    v, n = sig(fn(d, lo, hi, same_sensor=ss))
                    common.append(dict(site=s, pairing=mode, lag_lo_d=lo, lag_hi_d=hi,
                                       sensor="same-sensor" if ss else "any",
                                       sigma_pair_K=round(v, 2), n_pairs=n))
                for sn in ("L5", "L7", "L8", "L9"):
                    v, n = sig(fn(d, lo, hi, sensor=sn))
                    common.append(dict(site=s, pairing=mode, lag_lo_d=lo, lag_hi_d=hi,
                                       sensor=sn, sigma_pair_K=round(v, 2), n_pairs=n))
            # quiet segments only (centring does not change differences within a segment)
            qd = pd.concat(segs)
            for (lo, hi) in ((1, 8), (1, 16)):
                dy = np.concatenate([fn(g, lo, hi) for g in segs])
                v, n = sig(dy)
                common.append(dict(site=s, pairing=mode + "_quiet_only", lag_lo_d=lo,
                                   lag_hi_d=hi, sensor="any", sigma_pair_K=round(v, 2), n_pairs=n))
        # same-day (<1 d) repeatability
        gap = d["date"].diff().dt.days.to_numpy()[1:]
        dy = np.diff(d["y"].to_numpy(float))
        sd_m = gap == 0
        tsec = pd.to_datetime(d["datetime"]).astype("int64").to_numpy() / 1e9
        dsec = np.diff(tsec)[sd_m]
        sc = d["scene_id"].to_numpy()
        same_path = np.array([a.split("_")[2][:3] == b.split("_")[2][:3]
                              for a, b in zip(sc[:-1][sd_m], sc[1:][sd_m])]) if sd_m.any() else np.array([])
        s1 = d["sensor"].to_numpy()
        same_sen = (s1[:-1] == s1[1:])[sd_m]
        v, n = sig(dy[sd_m])
        cache[s] = dict(d=d, segs=segs, lag=lag, dif=dif, t0=t0, tb=tb, f0=f0, sm=sm,
                        same_day=dict(sigma=v, n=n, median_sep_s=float(np.median(np.abs(dsec))) if n else np.nan,
                                      frac_same_path=float(same_path.mean()) if n else np.nan,
                                      frac_same_sensor=float(same_sen.mean()) if n else np.nan,
                                      mean_abs_diff=float(np.mean(np.abs(dy[sd_m]))) if n else np.nan,
                                      n_scenes=len(d), n_unique_dates=int(d.date.nunique())))
        tbx = tb.copy(); tbx.insert(0, "site", s)
        tbx["used_for_plateau"] = False; tbx.loc[tbx.index[-3:], "used_for_plateau"] = True
        tbx["model_D_K2"] = 2 * sm ** 2 + f0["sill"] * (1 - np.exp(-tbx.lag_median_d / f0["tau"]))
        bins_all.append(tbx)

    repro = pd.DataFrame(repro); repro.to_csv(OUT / "baseline_reproduction.csv", index=False)
    common = pd.DataFrame(common); common.to_csv(OUT / "sigma_pair_common_conditions.csv", index=False)
    pd.DataFrame(seg_rows).to_csv(OUT / "quiet_segments.csv", index=False)
    pd.concat(bins_all).round(3).to_csv(OUT / "structure_function_bins.csv", index=False)
    sd = pd.DataFrame([dict(site=k, **v["same_day"]) for k, v in cache.items()]).round(3)
    sd.to_csv(OUT / "near_simultaneous_pairs.csv", index=False)

    # ---- (3) sensitivity to the short-lag term
    def cget(s, mode, lo, hi, sensor="any"):
        x = common[(common.site == s) & (common.pairing == mode) & (common.lag_lo_d == lo)
                   & (common.lag_hi_d == hi) & (common.sensor == sensor)]
        return float(x.sigma_pair_K.iloc[0])
    for s in D.SITES:
        c = cache[s]
        qv = float(np.var(np.concatenate([g.y.values for g in c["segs"]]), ddof=1))
        opts = [("published sigma_pair", c["sm"]),
                ("zero short-lag term", 0.0),
                ("common 1-8 d, consecutive", cget(s, "consecutive", 1, 8)),
                ("common 1-16 d, consecutive", cget(s, "consecutive", 1, 16)),
                ("common 1-16 d, same sensor, all pairs", cget(s, "all_pairs", 1, 16, "same-sensor")),
                ("common 1-16 d, quiet segments, all pairs", cget(s, "all_pairs_quiet_only", 1, 16)),
                ("Santa Ana near-simultaneous 0.56 K", cache["santa_ana"]["same_day"]["sigma"]),
                ("free intercept (fitted)", None)]
        for name, sv in opts:
            if sv is None:
                f = fit(c["tb"], 0, nugget_free=True); sv_eff = np.sqrt(f["s2m"])
            else:
                if not np.isfinite(sv):
                    continue
                f = fit(c["tb"], sv ** 2); sv_eff = sv
            lb = lagbin_boot(c["tb"], sv_eff ** 2)
            bb, _, nb = block_boot(c["lag"], c["dif"], c["t0"], c["segs"], sv_eff ** 2, 365, n=500)
            sens_rows.append(dict(
                site=s, short_lag_term=name, sigma_short_K=round(sv_eff, 2),
                sigma_quiet_SF_K=round(np.sqrt(f["sill"] / 2), 2),
                sigma_quiet_vardecomp_K=round(np.sqrt(max(qv - sv_eff ** 2, 0)), 2),
                tau_obs_d=round(f["tau"], 1),
                tau_lagbin_16=round(q(lb, 16), 1), tau_lagbin_84=round(q(lb, 84), 1),
                tau_block365_16=round(q(bb, 16), 1), tau_block365_84=round(q(bb, 84), 1)))
    sens = pd.DataFrame(sens_rows); sens.to_csv(OUT / "short_lag_sensitivity.csv", index=False)

    # ---- (4) dependence-preserving bootstrap
    for s in D.SITES:
        c = cache[s]
        for B in (90, 180, 365, 730):
            bb, sq, nb = block_boot(c["lag"], c["dif"], c["t0"], c["segs"], c["sm"] ** 2, B, n=1000)
            fin = np.isfinite(bb)
            boot_rows.append(dict(
                site=s, block_d=B, n_blocks=nb, n_rep=len(bb), tau_obs_d=round(c["f0"]["tau"], 1),
                tau_16=round(q(bb, 16), 1), tau_50=round(q(bb, 50), 1), tau_84=round(q(bb, 84), 1),
                tau_2p5=round(q(bb, 2.5), 1), tau_97p5=round(q(bb, 97.5), 1),
                frac_tau_below_8d=round(float(np.mean(bb[fin] < 8)), 3),
                frac_tau_above_300d=round(float(np.mean(bb[fin] > 300)), 3),
                sigma_quiet_16=round(q(sq, 16), 2), sigma_quiet_84=round(q(sq, 84), 2)))
        lb = lagbin_boot(c["tb"], c["sm"] ** 2)
        boot_rows.append(dict(site=s, block_d="lag-bin (published scheme)", n_blocks=len(c["tb"]),
                              n_rep=300, tau_obs_d=round(c["f0"]["tau"], 1),
                              tau_16=round(q(lb, 16), 1), tau_50=round(q(lb, 50), 1),
                              tau_84=round(q(lb, 84), 1), tau_2p5=round(q(lb, 2.5), 1),
                              tau_97p5=round(q(lb, 97.5), 1)))
    boot = pd.DataFrame(boot_rows); boot.to_csv(OUT / "tau_block_bootstrap.csv", index=False)

    # ---- (6) geometry scaling with each reference
    g = det[["n_tir_pixels", "f_median"]].copy()
    variants = {"published (window rule)": det.sigma_meas_K,
                "common 1-8 d consecutive": pd.Series({s: cget(s, "consecutive", 1, 8) for s in D.SITES}),
                "common 1-16 d consecutive": pd.Series({s: cget(s, "consecutive", 1, 16) for s in D.SITES}),
                "common 1-16 d same-sensor all pairs": pd.Series({s: cget(s, "all_pairs", 1, 16, "same-sensor") for s in D.SITES})}
    for vname, sv in variants.items():
        for ref in D.SITES:
            for s in D.SITES:
                pred = sv[ref] * np.sqrt(g.n_tir_pixels[ref] / g.n_tir_pixels[s]) * g.f_median[ref] / g.f_median[s]
                geo_rows.append(dict(sigma_input=vname, reference=ref, site=s,
                                     sigma_obs_K=round(sv[s], 2), sigma_pred_K=round(pred, 2),
                                     ratio_obs_pred=round(sv[s] / pred, 2)))
        k = sv * g.f_median * np.sqrt(g.n_tir_pixels)
        for s in D.SITES:
            geo_rows.append(dict(sigma_input=vname, reference="invariant k = sigma*f*sqrt(N)",
                                 site=s, sigma_obs_K=round(sv[s], 2), sigma_pred_K=round(k[s], 2),
                                 ratio_obs_pred=round(k[s] / k.mean(), 2)))
    geo = pd.DataFrame(geo_rows); geo.to_csv(OUT / "geometry_scaling_references.csv", index=False)

    # ---- (7) Taal diagnosis
    c = cache["taal"]; d = c["d"]
    taal = {}
    taal["scenes_by_period"] = {
        "1990-2019-07-15 (pre-window)": int((d.date < "2019-07-16").sum()),
        "2019-07-16..2020-01-11 (pre-onset window)": int(((d.date >= "2019-07-16") & (d.date < "2020-01-12")).sum()),
        "2020-01-12..2020-07-10 (post-onset window)": int(((d.date >= "2020-01-12") & (d.date <= "2020-07-10")).sum()),
        "after 2020-07-10": int((d.date > "2020-07-10").sum())}
    taal["first_post_2020"] = str(d.loc[d.date >= "2020-01-12", "date"].min().date())
    taal["last_pre_2020"] = str(d.loc[d.date < "2020-01-12", "date"].max().date())
    yr = d.groupby(d.date.dt.year).size()
    taal["scenes_per_year_quiet_median"] = float(yr[yr.index <= 2018].median())
    # sensitivity: relax seg_min to let post-2020 short segments in
    alt = []
    for seg_min in (15, 10, 5, 3):
        segs, _ = quiet_segments("taal", d, seg_min=seg_min)
        lag, dif, t0 = sf_pairs(segs)
        tb = binned(lag, dif)
        f = fit(tb, c["sm"] ** 2)
        lb = lagbin_boot(tb, c["sm"] ** 2) if len(tb) >= 5 else np.array([np.nan])
        bb, _, nb = block_boot(lag, dif, t0, segs, c["sm"] ** 2, 365, n=500) if len(tb) >= 5 else (np.array([np.nan]), None, 0)
        alt.append(dict(case=f"seg_min={seg_min}", n_segments=len(segs), n_scenes=int(sum(len(g) for g in segs)),
                        n_pairs=len(lag), n_bins=len(tb), tau=round(f["tau"], 1),
                        lagbin_16_84=f"{q(lb,16):.1f}-{q(lb,84):.1f}",
                        block365_16_84=f"{q(bb,16):.1f}-{q(bb,84):.1f}",
                        sigma_quiet=round(np.sqrt(f["sill"] / 2), 2)))
    # pre-2020 only, and an L7/L8-era subset with denser sampling (2000-2019)
    for label, before, after in (("pre-2020 only (== baseline)", "2019-07-16", None),
                                 ("2000-2019 only (L7/L8 era)", "2019-07-16", "2000-01-01"),
                                 ("1990-1999 only", "2000-01-01", None)):
        dd = d if after is None else d[d.date >= after].reset_index(drop=True)
        segs, _ = quiet_segments("taal", dd, before=before)
        lag, dif, t0 = sf_pairs(segs); tb = binned(lag, dif)
        f = fit(tb, c["sm"] ** 2)
        lb = lagbin_boot(tb, c["sm"] ** 2) if len(tb) >= 5 else np.array([np.nan])
        alt.append(dict(case=label, n_segments=len(segs), n_scenes=int(sum(len(g) for g in segs)),
                        n_pairs=len(lag), n_bins=len(tb), tau=round(f["tau"], 1),
                        lagbin_16_84=f"{q(lb,16):.1f}-{q(lb,84):.1f}", block365_16_84="",
                        sigma_quiet=round(np.sqrt(f["sill"] / 2), 2) if np.isfinite(f["sill"]) else np.nan))
    pd.DataFrame(alt).to_csv(OUT / "taal_diagnosis_variants.csv", index=False)
    # SSE profile across tau (flatness)
    prof = []
    for s in D.SITES:
        sse = cache[s]["f0"]["sse"]
        for tt in (1, 2, 4, 8, 16, 30, 60, 120, 240, 480):
            k = int(np.argmin(np.abs(GRID - tt)))
            prof.append(dict(site=s, tau_d=tt, sse_rel_min=round(float(sse[k] / sse.min()), 3)))
    pd.DataFrame(prof).to_csv(OUT / "sse_profile.csv", index=False)
    (OUT / "taal_diagnosis.json").write_text(json.dumps(taal, indent=2, default=str))

    make_figure(cache)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(repro.T); print(sd.T); print(pd.read_csv(OUT / "quiet_segments.csv"))
    print(common.to_string()); print(sens.to_string()); print(boot.to_string())
    print(geo.to_string()); print(pd.DataFrame(alt).to_string()); print(json.dumps(taal, indent=1))
    print(pd.DataFrame(prof).pivot(index="tau_d", columns="site", values="sse_rel_min"))
    print(pd.concat(bins_all).round(2).to_string())


# ---------------------------------------------------------------- figure
def make_figure(cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grey = "#1a1a1a", "#555555", "#9a9a9a"
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.6),
                           gridspec_kw=dict(height_ratios=[1, 2.1], hspace=0.42, wspace=0.26))
    fig.subplots_adjust(bottom=0.16)
    for k, s in enumerate(D.SITES):
        c = cache[s]; d = c["d"]; col = COLOR[s]
        # timeline
        a = ax[0, k]
        ev = [pd.Timestamp(e["start"]) for e in get_site(s).known_events if e.get("verified")]
        for e in ev:
            a.axvspan(e - pd.Timedelta(days=P.PRIMARY_W), e + pd.Timedelta(days=P.PRIMARY_W),
                      color="#d9d9d9", lw=0, zorder=0)
            a.axvline(e, color=grey, lw=0.8, zorder=1)
        qi = pd.concat(c["segs"]).index
        m = d.index.isin(qi)
        a.plot(d.date[~m], d.y[~m], "o", ms=2.6, color=grey, mec="none", label="Excluded / short segment")
        a.plot(d.date[m], d.y[m], "o", ms=2.6, color=col, mec="none", label="Quiet segment (used)")
        for g in c["segs"]:
            a.plot([g.date.min(), g.date.max()], [1.04, 1.04], transform=a.get_xaxis_transform(),
                   color=col, lw=3, solid_capstyle="butt", clip_on=False)
        a.set_title(f"({chr(97 + k)}) {NAME[s]}: quiet segments", loc="left", fontsize=9.5, color=ink, pad=11)
        if s == "taal":
            a.axvspan(pd.Timestamp("2020-01-12"), pd.Timestamp("2020-03-31"), color="#7a7a7a", lw=0, zorder=1)
            a.annotate("lake absent\nJan–Mar 2020", xy=(pd.Timestamp("2020-02-20"), 0.80),
                       xycoords=("data", "axes fraction"), xytext=(pd.Timestamp("2017-06-01"), 0.55),
                       textcoords=("data", "axes fraction"), ha="right", va="top", fontsize=6.5, color=ink2,
                       arrowprops=dict(arrowstyle="-", lw=0.6, color=ink2))
        a.set_ylabel("Relative temperature (K)", fontsize=8.5)
        a.tick_params(labelsize=8)
        a.set_xlim(pd.Timestamp("1989-06-01"), pd.Timestamp("2027-01-01"))
        if k == 0:
            a.legend(fontsize=7, frameon=False, loc="upper left", markerscale=2)
        # structure function
        b = ax[1, k]; tb = c["tb"]; f = c["f0"]; sm = c["sm"]
        b.axvspan(1, D.SF_MAX_LAG_D, color="#f3f3f3", lw=0, zorder=0)
        b.axvspan(tb.lag_lo_d.iloc[-3], tb.lag_hi_d.iloc[-1], color="#e4e4e4", lw=0, zorder=0)
        b.plot(tb.lag_median_d, tb.D_K2, "o", ms=6.5, color=col, mec="white", mew=0.8, zorder=3,
               label="Empirical D (lag bin)")
        #  pair counts in a row along the bottom of the panel (one per bin, staggered)
        for j, (x, n) in enumerate(zip(tb.lag_median_d, tb.n_pairs)):
            b.text(x, 0.025 + 0.045 * (j % 2), f"{int(n)}", transform=b.get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=6, color=ink2)
        xx = np.geomspace(0.8, 600, 300)
        b.plot(xx, 2 * sm ** 2 + f["sill"] * (1 - np.exp(-xx / f["tau"])), "-", color=ink, lw=1.4,
               zorder=2, label="Fitted model")
        b.axhline(2 * sm ** 2, color=ink2, lw=0.9, ls=":", label=r"Intercept $2\sigma_{pair}^2$")
        b.axhline(2 * sm ** 2 + f["sill"], color=ink2, lw=0.9, ls="--",
                  label=r"Plateau (mean of last three bins)")
        b.set_xscale("log")
        b.set_xlim(0.8, 600)
        b.set_ylim(-0.16 * max(tb.D_K2.max(), 2 * sm ** 2 + f["sill"]), max(tb.D_K2.max(), 2 * sm ** 2 + f["sill"]) * 1.45)
        b.set_xlabel("Lag (days)", fontsize=8.5)
        if k == 0:
            b.set_ylabel(r"Structure function D (K$^2$)", fontsize=8.5)
        b.tick_params(labelsize=8)
        tau_txt = (f"$\\tau_{{obs}}$ = {f['tau']:.0f} d" if s != "taal"
                   else f"$\\tau_{{obs}}$ not constrained (best fit {f['tau']:.1f} d)")
        b.set_title(f"({chr(100 + k)}) {NAME[s]}: structure function", loc="left", fontsize=9.5, color=ink)
        b.text(0.02, 0.97, f"{tau_txt}\n$\\sigma_{{pair}}$ = {sm:.2f} K, "
               f"$\\sigma_{{quiet}}$ = {np.sqrt(f['sill']/2):.2f} K\n"
               f"{len(c['lag'])} pairs, {len(tb)} bins (≥5 pairs), lags 1–{D.SF_MAX_LAG_D} d",
               transform=b.transAxes, va="top", fontsize=7.5, color=ink)
    h, l = ax[1, 2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.035))
    from pipeline.analysis.figcheck import check
    iss = check(fig, "figS1_structure_function")
    if iss:
        print("   ! figS1_structure_function:", len(iss), "overlap issue(s)")
    fig.savefig(OUT / "figS_structure_function.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figS_structure_function.pdf", bbox_inches="tight")
    plt.close(fig)




# ---------------------------------------------------------------- extras
def _sf_generic(segs, max_lag, edges, robust=False):
    lag, dif, _ = [], [], None
    for g in segs:
        v = g["y"].to_numpy(float)
        t = g["date"].to_numpy("datetime64[D]").astype("int64")
        for i in range(len(v)):
            dt = t[i + 1:] - t[i]
            m = (dt >= 1) & (dt <= max_lag)
            if m.any():
                lag.append(dt[m]); dif.append(v[i + 1:][m] - v[i])
    lag = np.concatenate(lag).astype(float); dif = np.concatenate(dif)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lag >= a) & (lag < b)
        if m.sum() >= 5:
            x = dif[m]
            Dv = (1.4826 * np.median(np.abs(x - np.median(x)))) ** 2 if robust else float(np.mean(x ** 2))
            rows.append((a, b, float(np.median(lag[m])), float(Dv), int(m.sum())))
    return pd.DataFrame(rows, columns=["lag_lo_d", "lag_hi_d", "lag_median_d", "D_K2", "n_pairs"]), lag, dif


def extras():
    global EDGES
    rows, longlag, collapse = [], [], []
    base_edges = EDGES.copy()
    for s in D.SITES:
        d, col = series(s)
        sm = D.noise_from_close_pairs(d[["date", "y"]])[0]
        segs, _ = quiet_segments(s, d)
        qv = float(np.var(np.concatenate([g.y.values for g in segs]), ddof=1))
        # (a) max-lag sensitivity (bins: same log spacing, 22 edges from 1 to max_lag)
        for ml in (200, 400, 800, 1600, 3650):
            e = np.unique(np.round(np.geomspace(1, ml, 22)).astype(int))
            for rb in (False, True):
                tb, lag, dif = _sf_generic(segs, ml, e, robust=rb)
                EDGES = e
                f = fit(tb, sm ** 2)
                EDGES = base_edges
                rows.append(dict(site=s, max_lag_d=ml, D_estimator="robust (MAD)" if rb else "mean square",
                                 n_bins=len(tb), plateau_K2=round(float(np.mean(tb.D_K2.values[-3:])), 2),
                                 two_var_quiet_K2=round(2 * qv, 2), tau_obs_d=round(f["tau"], 1),
                                 sigma_quiet_SF_K=round(np.sqrt(f["sill"] / 2), 2)))
        # (b) D at long lags (beyond the fitted range)
        e = np.array([1, 16, 64, 200, 400, 800, 1600, 3200, 11000])
        tb, _, _ = _sf_generic(segs, 11000, e)
        for r in tb.itertuples():
            longlag.append(dict(site=s, lag_lo_d=r.lag_lo_d, lag_hi_d=r.lag_hi_d, D_K2=round(r.D_K2, 2),
                                n_pairs=r.n_pairs, two_var_quiet_K2=round(2 * qv, 2)))
        # (c) collapse same-date scenes (duplicate WRS-row products) to their mean
        dc = d.groupby("date", as_index=False).agg(y=("y", "mean"), sensor=("sensor", "first"))
        smc, npc, winc, byc, _, _ = D.noise_from_close_pairs(dc[["date", "y"]])
        segc, _ = quiet_segments(s, dc)
        lag, dif, t0 = sf_pairs(segc); tb = binned(lag, dif); f = fit(tb, smc ** 2)
        lb = lagbin_boot(tb, smc ** 2)
        bb, _, _ = block_boot(lag, dif, t0, segc, smc ** 2, 365, n=500)
        collapse.append(dict(site=s, n_scenes=len(d), n_dates=len(dc), sigma_pair_K=round(smc, 2),
                             n_pairs=npc, window_d=winc, sigma_8d=byc[8], sigma_16d=byc[16],
                             n_sf_pairs=len(lag), tau_obs_d=round(f["tau"], 1),
                             lagbin_16_84=f"{q(lb,16):.1f}-{q(lb,84):.1f}",
                             block365_16_84=f"{q(bb,16):.1f}-{q(bb,84):.1f}",
                             sigma_quiet_SF_K=round(np.sqrt(f["sill"] / 2), 2)))
    a = pd.DataFrame(rows); a.to_csv(OUT / "tau_maxlag_and_robust_D.csv", index=False)
    b = pd.DataFrame(longlag); b.to_csv(OUT / "structure_function_long_lags.csv", index=False)
    c = pd.DataFrame(collapse); c.to_csv(OUT / "same_date_collapsed.csv", index=False)
    pd.set_option("display.width", 250)
    print(a.to_string()); print(b.to_string()); print(c.to_string())
    # Taal 16-d bin: which pairs dominate
    d, _ = series("taal"); segs, _ = quiet_segments("taal", d)
    g = segs[0]; t = g.date.values; v = g.y.values; out = []
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            dt = (t[j] - t[i]).astype("timedelta64[D]").astype(int)
            if 13 <= dt < 17:
                out.append((str(g.date.iloc[i].date()), str(g.date.iloc[j].date()), g.sensor.iloc[i],
                            g.sensor.iloc[j], round(v[j] - v[i], 2)))
    o = pd.DataFrame(out, columns=["t1", "t2", "s1", "s2", "dy"])
    o["sq"] = o.dy ** 2
    print(o.sort_values("sq", ascending=False).head(8).to_string())
    print("share of top-2 pairs in 16 d bin sum of squares:", round(o.sq.nlargest(2).sum() / o.sq.sum(), 2),
          "D without top-2:", round(o.sq.nsmallest(len(o) - 2).mean(), 2))
    o.to_csv(OUT / "taal_16d_bin_pairs.csv", index=False)




def jackknife():
    """Leave-one-scene-out refit (quiet segments re-centred each time)."""
    out = []
    for s in D.SITES:
        d, _ = series(s)
        sm = D.noise_from_close_pairs(d[["date", "y"]])[0]
        segs0, _ = quiet_segments(s, d)
        qidx = pd.concat(segs0).index
        for i in qidx:
            dd = d.drop(index=i).reset_index(drop=True)
            segs, _ = quiet_segments(s, dd)
            lag, dif, _ = sf_pairs(segs)
            f = fit(binned(lag, dif), sm ** 2)
            out.append(dict(site=s, dropped_date=str(d.date[i].date()), sensor=d.sensor[i],
                            y_K=round(d.y[i], 2), tau_obs_d=round(f["tau"], 1),
                            sigma_quiet_K=round(np.sqrt(f["sill"] / 2), 2)))
    j = pd.DataFrame(out); j.to_csv(OUT / "leave_one_scene_out.csv", index=False)
    for s, g in j.groupby("site"):
        print(s, "n", len(g), "tau min/median/max", g.tau_obs_d.min(), g.tau_obs_d.median(), g.tau_obs_d.max(),
              "q05/q95", g.tau_obs_d.quantile([.05, .95]).round(1).tolist())
        print(g.sort_values("tau_obs_d", ascending=False).head(4).to_string())


if __name__ == "__main__":
    import sys
    if "--figure-only" in sys.argv:
        c = {}
        for st in D.SITES:
            d, _ = series(st); sm = D.noise_from_close_pairs(d[["date", "y"]])[0]
            segs, _ = quiet_segments(st, d); lag, dif, t0 = sf_pairs(segs); tb = binned(lag, dif)
            c[st] = dict(d=d, segs=segs, lag=lag, tb=tb, f0=fit(tb, sm ** 2), sm=sm)
        make_figure(c)
    else:
        main(); extras(); jackknife()
