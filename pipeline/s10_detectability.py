# -*- coding: utf-8 -*-
"""s10_detectability: what change can be detected at each lake, independent of eruptions.

Computes, from lake geometry and observing conditions only:

1. Measurement noise sigma_meas from the differences of close scene pairs. The
   lake is assumed not to change within a few days, so the pair difference is
   measurement error: sigma = std(diff) / sqrt(2). No model is assumed.

2. The sources of that noise:
   (a) pixel count: averaging over N thermal pixels reduces noise by sqrt(N);
   (b) background ring: solving the unmixing L = f*L_lake + (1-f)*L_bg for
       L_lake carries the background error with weight (1-f)/f (zero at f = 1).
   Prediction: sigma_meas ≈ sigma_ref * sqrt(N_ref / N) / f  (reference site).

3. Real lake variability sigma_real = sqrt(sigma_tot^2 - sigma_meas^2), set by
   depth through the thermal time constant tau = rho c h / K (h = mean depth,
   K = surface heat-transfer coefficient). Area cancels because volume and
   surface area scale together; shallow lakes change state within weeks, deep
   lakes retain it for longer.

Together these imply that the comparison window should be matched to lake depth.

Outputs (s10_detectability records): detectability_by_site, threshold_bridge,
pre_eruption_state.

    python -m pipeline.s10_detectability
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common.report import StageCheck
from pipeline.config import settings as S
from pipeline.config.sites import get_site
from pipeline import s10_post_eruption as P

RHO_W, C_W = 1000.0, 4186.0
#  Range of the combined surface heat-transfer coefficient (evaporation, sensible
#  heat, net radiation) for a crater lake near 40 °C; both ends are reported.
K_LO, K_HI = 60.0, 120.0
#  Candidate maximum separations of close scene pairs. Shorter separations make
#  the "same lake state" assumption safer but give fewer pairs; the shortest window
#  with at least 5 pairs is used and recorded (Taal has no 3-day pairs because of
#  its revisit geometry). All three are reported: sigma rising with separation
#  indicates real change entering, a flat curve indicates noise-dominated pairs.
PAIR_WINDOWS_D = (3, 8, 16)
LANDSAT_TIR_PIX_M2 = 1e4  # nominal 100 m thermal pixel

SITES = ("taal", "santa_ana", "ruapehu")
REF_SITE = "ruapehu"      # reference for the noise prediction: many pixels and f = 1
#  minimum prior scenes for the pre-onset percentile
MIN_PRIOR_SCENES = 20


def _series(site_key: str):
    """Relative-temperature series after the water-fraction screen."""
    df, ke, _ = P.load(site_key, P.F_MIN)
    if df is None:
        return None, None
    col = P.PRIMARY if df[P.PRIMARY].notna().sum() >= 20 else P.FALLBACK
    d = df[["date", col]].dropna().rename(columns={col: "y"})
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True), col


def noise_from_close_pairs(d: pd.DataFrame):
    """Measurement noise from differences of close scene pairs.

    The lake is assumed unchanged over a few days, so the difference is
    measurement error; sigma = std(diff) / sqrt(2) for independent scene errors.
    Returns (sigma used, number of pairs, window used, sigma per window,
    same-day repeatability, number of same-day pairs).
    """
    gap = d["date"].diff().dt.days.to_numpy()[1:]
    dy = np.diff(d["y"].to_numpy(float))
    #  Exclude same-day pairs (gap 0). Where Landsat paths overlap, adjacent paths
    #  image the lake twice on one day; that difference measures repeatability of
    #  near-simultaneous observations and would underestimate sigma_meas. It is
    #  reported separately as same-day repeatability.
    same_day = gap == 0
    repeat = (round(float(np.std(dy[same_day], ddof=1) / np.sqrt(2)), 2)
              if same_day.sum() >= 5 else np.nan)
    by = {}
    used = (np.nan, 0, np.nan)
    for lim in PAIR_WINDOWS_D:
        m = (gap >= 1) & (gap <= lim)
        by[lim] = (round(float(np.std(dy[m], ddof=1) / np.sqrt(2)), 2)
                   if m.sum() >= 5 else np.nan)
        if m.sum() >= 5 and not np.isfinite(used[0]):
            used = (float(np.std(dy[m], ddof=1) / np.sqrt(2)), int(m.sum()), lim)
    return used[0], used[1], used[2], by, repeat, int(same_day.sum())


#  minimum scenes per segment for the structure function, and maximum fitted lag (days)
SEG_MIN_SCENES = 15
SF_MAX_LAG_D = 400


def _quiet_segments(site_key: str, d: pd.DataFrame):
    """Split the series into contiguous segments away from eruptions.

    Step changes in lake level (Taal 2020, Santa Ana 2005) would inflate the
    long-lag squared differences of a whole-record structure function and imply a
    spuriously long memory. The series is cut at the eruption windows (±180 days),
    only pairs within one segment are used, and each segment's mean is removed.
    """
    ev = [pd.Timestamp(e["start"]) for e in get_site(site_key).known_events
          if e.get("verified")]
    t = d["date"]
    near = np.zeros(len(d), bool)
    for e in ev:
        near |= (np.abs((t - e).dt.days) <= P.PRIMARY_W).to_numpy()
    #  group the scenes outside eruption windows into contiguous segments
    seg, cur, out = [], [], None
    for i, bad in enumerate(near):
        if bad:
            if cur:
                seg.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        seg.append(cur)
    for idx in seg:
        if len(idx) < SEG_MIN_SCENES:
            continue
        g = d.iloc[idx].copy()
        g["y"] = g["y"] - g["y"].mean()      # remove the segment mean
        out = g if out is None else out
        yield g


def observed_timescale(site_key: str, d: pd.DataFrame, sigma_meas: float):
    """Observed persistence of the lake thermal state from the structure function.

    tau = rho c h / K is computed from depth, not observed. The observed
    persistence comes from the mean squared difference by lag

        D(Δt) = ⟨[y(t+Δt) − y(t)]²⟩,

    which tends to 2σ_meas² as Δt → 0 and to 2σ_real² for lags longer than the
    memory. After removing the measurement part, D_real is fitted with the
    exponential-memory model

        D_real(Δt) = 2σ_real² (1 − e^(−Δt/τ_obs))

    to obtain τ_obs, which is compared with the depth-based tau.
    """
    lag, dif = [], []
    for g in _quiet_segments(site_key, d):
        v = g["y"].to_numpy(float)
        t = g["date"].to_numpy("datetime64[D]").astype("int64")
        for i in range(len(v)):
            dt = t[i + 1:] - t[i]
            m = (dt >= 1) & (dt <= SF_MAX_LAG_D)
            if m.any():
                lag.append(dt[m]); dif.append(v[i + 1:][m] - v[i])
    if not lag:
        return dict(tau_obs_d=np.nan, tau_obs_lo_d=np.nan, tau_obs_hi_d=np.nan,
                    n_sf_pairs=0, sigma_real_quiet_K=np.nan)
    lag = np.concatenate(lag).astype(float)
    dif = np.concatenate(dif).astype(float)
    #  log-spaced lag bins; D per bin (bins with >= 5 pairs)
    edges = np.unique(np.round(np.geomspace(1, SF_MAX_LAG_D, 22)).astype(int))
    xs, ds, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lag >= a) & (lag < b)
        if m.sum() >= 5:
            xs.append(float(np.median(lag[m])))
            ds.append(float(np.mean(dif[m] ** 2)))
            ns.append(int(m.sum()))
    if len(xs) < 5:
        return dict(tau_obs_d=np.nan, tau_obs_lo_d=np.nan, tau_obs_hi_d=np.nan,
                    n_sf_pairs=int(len(lag)), sigma_real_quiet_K=np.nan)
    xs, ds, ns = np.array(xs), np.array(ds), np.array(ns)
    s2m = (sigma_meas ** 2 if np.isfinite(sigma_meas) else 0.0)
    #  sill: mean of the three longest-lag bins minus the noise part
    sill = max(float(np.mean(ds[-3:])) - 2 * s2m, 1e-6)

    def model(tt, tau):
        return 2 * s2m + sill * (1.0 - np.exp(-tt / tau))

    grid = np.geomspace(0.5, 600.0, 4000)
    #  pair-count-weighted sum of squared residuals
    sse = np.array([float(np.sum(ns * (ds - model(xs, g)) ** 2)) for g in grid])
    tau_hat = float(grid[int(np.argmin(sse))])
    #  bootstrap over bins for the range
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(300):
        pick = rng.integers(0, len(xs), len(xs))
        s = np.array([float(np.sum(ns[pick] * (ds[pick] - model(xs[pick], g)) ** 2))
                      for g in grid])
        boots.append(float(grid[int(np.argmin(s))]))
    lo, hi = np.percentile(boots, [16, 84])
    return dict(tau_obs_d=round(tau_hat, 1), tau_obs_lo_d=round(float(lo), 1),
                tau_obs_hi_d=round(float(hi), 1), n_sf_pairs=int(len(lag)),
                sigma_real_quiet_K=round(float(np.sqrt(sill / 2.0)), 2))


def geometry(site_key: str):
    """Lake geometry: area, number of thermal pixels and mean depth.

    Mean depth is volume / area, with the published volume from s10_magma.GEOM
    (`mean_depth_m` in sites.py is close to the maximum depth and is not used):
        Taal    4.2e7 m³ / 1.2 km²  = 35 m   (Perttu et al., 2023)
        Ruapehu 9.0e6 m³ / 0.19 km² = 47 m   (Christenson, 1994)
    """
    from pipeline.s10_magma import GEOM
    a = get_site(site_key).active
    area = float(a.area_km2_ref) * 1e6
    n_pix = area / LANDSAT_TIR_PIX_M2
    vol = GEOM.get(site_key, {}).get("volume_m3", np.nan)
    h = (vol / area) if (vol and np.isfinite(vol) and area > 0) else np.nan
    return area, n_pix, h


#  ── Depth bounds where a single mean depth cannot be fixed ────────────────
#  Santa Ana: the published volume (0.47 × 10⁶ m³, max 27 m; Bernard et al. 2004)
#  was surveyed in 2000-2002 when the lake was ~200 m across (≈0.031 km²); after
#  the 2005 eruption the area is 0.055 km². Depending on the area used the mean
#  depth is 8.5-15 m, which is used as a range (depth_is_bound = True).
DEPTH_BOUND_M = {"santa_ana": (8.5, 15.0)}


def _tau_fields(site_key, h):
    """Thermal time constant tau = rho c h / K; a range where depth is only bounded."""
    def tau(hh, K):
        return round(RHO_W * C_W * hh / K / 86400.0, 1)
    #  sites with explicit depth bounds use the range even if a volume exists
    if np.isfinite(h) and site_key not in DEPTH_BOUND_M:
        return dict(depth_is_bound=False,
                    tau_days_lo=tau(h, K_HI), tau_days_hi=tau(h, K_LO))
    lo_h, hi_h = DEPTH_BOUND_M.get(site_key, (np.nan, np.nan))
    if not np.isfinite(lo_h):
        return dict(depth_is_bound=False, tau_days_lo=np.nan, tau_days_hi=np.nan)
    #  shallowest/fastest-cooling to deepest/slowest-cooling
    return dict(depth_is_bound=True,
                tau_days_lo=tau(lo_h, K_HI), tau_days_hi=tau(hi_h, K_LO))


def run():
    rows, ref = [], {}
    for s in SITES:
        d, col = _series(s)
        if d is None or len(d) < 20:
            continue
        sig_meas, n_pair, pair_win, sig_by_win, repeat, n_same = noise_from_close_pairs(d)
        sig_tot = float(np.std(d["y"].to_numpy(float), ddof=1))
        sig_real = (float(np.sqrt(max(sig_tot ** 2 - sig_meas ** 2, 0.0)))
                    if np.isfinite(sig_meas) else np.nan)
        area, n_pix, h = geometry(s)

        #  water fraction and background ring: background error carried by unmixing
        raw = pd.read_csv(S.records_dir(3, s) / "deltaT.csv")
        raw = raw[raw["qc_flag"].eq("ok")]
        f = pd.to_numeric(raw["f_est_act"], errors="coerce")
        raw, f = raw[f >= P.F_MIN], f[f >= P.F_MIN]
        bgc = next((c for c in ("bg_active_K", "bg_mean_K", "T_bg_act_K")
                    if c in raw.columns and raw[c].notna().any()), None)
        bg = pd.to_numeric(raw[bgc], errors="coerce") if bgc else pd.Series(dtype=float)
        f_med = float(f.median())
        leak_w = (1.0 - f_med) / f_med
        bg_sd = float(bg.std()) if len(bg.dropna()) > 5 else np.nan

        rows.append(dict(
            site=s, estimator=col, n_scenes=len(d),
            lake_area_m2=round(area, 0), n_tir_pixels=round(n_pix, 1),
            mean_depth_m=(round(h, 1) if np.isfinite(h) else np.nan),
            depth_bound_m=("-".join(str(x) for x in DEPTH_BOUND_M[s])
                           if s in DEPTH_BOUND_M else ""),
            f_median=round(f_med, 3),
            background_sd_K=(round(bg_sd, 2) if np.isfinite(bg_sd) else np.nan),
            unmix_leak_weight=round(leak_w, 3),
            background_into_lake_K=(round(bg_sd * leak_w, 2)
                                    if np.isfinite(bg_sd) else np.nan),
            n_close_pairs=n_pair, pair_window_d=pair_win,
            n_same_day_pairs=n_same, same_day_repeatability_K=repeat,
            sigma_meas_3d_K=sig_by_win.get(3, np.nan),
            sigma_meas_8d_K=sig_by_win.get(8, np.nan),
            sigma_meas_16d_K=sig_by_win.get(16, np.nan),
            sigma_meas_K=(round(sig_meas, 2) if np.isfinite(sig_meas) else np.nan),
            sigma_total_K=round(sig_tot, 2),
            sigma_real_K=(round(sig_real, 2) if np.isfinite(sig_real) else np.nan),
            meas_variance_share=(round(sig_meas ** 2 / sig_tot ** 2, 3)
                                 if np.isfinite(sig_meas) else np.nan),
            **_tau_fields(s, h),
            **observed_timescale(s, d, sig_meas)))
        if s == REF_SITE:
            ref = rows[-1]

    df = pd.DataFrame(rows)
    #  predict noise from the reference site using pixel count and f only
    if ref:
        df["sigma_meas_predicted_K"] = [
            (round(ref["sigma_meas_K"] * np.sqrt(ref["n_tir_pixels"] / r.n_tir_pixels)
                   / r.f_median, 2) if np.isfinite(ref["sigma_meas_K"]) else np.nan)
            for r in df.itertuples()]
    return df


#  variance of a window median is π/2 times that of the mean (Gaussian case)
MEDIAN_COST = float(np.sqrt(np.pi / 2))


def threshold_bridge(det: pd.DataFrame):
    """Compare the predicted detection threshold with the placebo range per onset.

    For independent scenes the standard deviation of a window-median difference is

        σ_Δ ≈ √(π/2) · √(σ_meas² + σ_quiet²) · √(1/n_pre + 1/n_post),

    with a 95 % width of 3.92 σ_Δ. This is set beside the 2.5–97.5 % width of the
    matched placebo distribution, using the detectability columns and the scene
    counts of the test.

    Diagnostic: the placebo range after also removing placebo days within 3W after
    a real onset. Placebo days are already more than 2W from real onsets, but a
    response lasting longer than W can decay into the next year's placebo window.
    The test itself is unchanged; such placebos only widen the null.
    """
    ev = pd.read_csv(S.RECORDS_DIR / "10_post_eruption" / "per_eruption_delta_paper.csv")
    det = det.set_index("site")
    out = []
    for s in SITES:
        d, col = _series(s)
        if d is None or s not in det.index:
            continue
        r = det.loc[s]
        sig_scene = float(np.hypot(r["sigma_meas_K"], r["sigma_real_quiet_K"]))
        dnum = d["date"].to_numpy("datetime64[D]").astype("int64")
        y = d["y"].to_numpy(float)
        e = ev[(ev["site"] == s) & (ev["estimator"] == col) & ev["delta"].notna()
               & ev["window_d"].isin([180, 365])]
        for w in sorted(e["window_d"].unique()):
            #  candidates from the same real onset list as the test (merged within W)
            real = [int(np.datetime64(o["t0"].date(), "D").astype("int64"))
                    for o in P.onsets(get_site(s).known_events, int(w))]
            real = [t for t in real if dnum.min() <= t - w and t <= dnum.max()]
            cands = P.placebo_candidates(dnum, y, real, int(w))
            for x in e[e["window_d"] == w].itertuples():
                t0 = pd.Timestamp(x.onset)
                pool, _, _ = P._matched_pool(cands, t0.dayofyear, x.n_pre, x.n_post)
                sd = MEDIAN_COST * sig_scene * np.sqrt(1 / x.n_pre + 1 / x.n_post)
                q_lo, q_hi = np.percentile(pool, [2.5, 97.5])
                #  diagnostic pool excluding placebos within 3W after a real onset
                #  (same matching); day numbers rebuilt from year and day of year
                cand_n = ((cands[2].astype("int64") - 1970).astype("datetime64[Y]")
                          .astype("datetime64[D]").astype("int64") + cands[1] - 1)
                far = np.ones(len(cand_n), bool)
                for t in real:
                    far &= ~((cand_n - t > 0) & (cand_n - t <= 3 * w))
                sub = tuple(c[far] for c in cands)
                pool2, _, _ = P._matched_pool(sub, t0.dayofyear, x.n_pre, x.n_post)
                q2 = (np.percentile(pool2, [2.5, 97.5]) if len(pool2) >= 10
                      else (np.nan, np.nan))
                out.append(dict(
                    site=s, onset=x.onset, window_d=int(w), n_pre=int(x.n_pre),
                    n_post=int(x.n_post), sigma_scene_quiet_K=round(sig_scene, 2),
                    sigma_delta_pred_K=round(float(sd), 2),
                    width_pred_K=round(float(3.92 * sd), 1),
                    placebo_q025=round(float(q_lo), 2), placebo_q975=round(float(q_hi), 2),
                    width_placebo_K=round(float(q_hi - q_lo), 1),
                    width_ratio=round(float((q_hi - q_lo) / (3.92 * sd)), 2),
                    n_placebo=int(len(pool)),
                    placebo_q025_excl3w=(round(float(q2[0]), 2) if np.isfinite(q2[0]) else np.nan),
                    placebo_q975_excl3w=(round(float(q2[1]), 2) if np.isfinite(q2[1]) else np.nan),
                    width_placebo_excl3w_K=(round(float(q2[1] - q2[0]), 1)
                                            if np.isfinite(q2[0]) else np.nan),
                    n_placebo_excl3w=int(len(pool2))))
    return pd.DataFrame(out)


def pre_eruption_state():
    """Percentile of the 180-day pre-onset lake state within the lake's own record.

    Uses the level rather than the change (Δ_pre), which is more robust when the
    window holds only a few scenes. A sealed system would be colder than usual
    before an eruption.

    The reference distribution is the record before the onset only. Using the
    whole record would be circular: when an eruption raises the lake level
    permanently (Taal 2020, Santa Ana 2005), any pre-onset scene ranks low by
    construction. Only data available on the onset date are used; the
    whole-record percentile is reported for comparison.
    """
    out = []
    for s in SITES:
        d, col = _series(s)
        if d is None:
            continue
        allv = d["y"].to_numpy(float)
        for e in get_site(s).known_events:
            if not e.get("verified"):
                continue
            t0 = pd.Timestamp(e["start"])
            pre = d[(d.date >= t0 - pd.Timedelta(days=P.PRIMARY_W)) & (d.date < t0)]
            if len(pre) < P.MIN_SCENES:
                continue
            med = float(pre["y"].median())
            prior = d.loc[d.date < t0, "y"].to_numpy(float)
            out.append(dict(
                site=s, onset=str(t0.date()), estimator=col,
                n_pre=len(pre), pre_median_K=round(med, 2),
                n_prior=len(prior),
                percentile_in_prior_record=(
                    round(float((prior < med).mean() * 100), 1)
                    if len(prior) >= MIN_PRIOR_SCENES else np.nan),
                percentile_in_own_record=round(
                    float((allv < med).mean() * 100), 1)))
    return pd.DataFrame(out)


def main():
    chk = StageCheck(stage="10d", title="Detectability: noise, depth and window length",
                     desc="Detectable change at each lake from geometry and "
                          "observing conditions only, independent of eruptions.")
    df = run()
    chk.table(df, "detectability_by_site",
              desc="Per-site measurement noise (close scene pairs), its sources "
                   "(pixel count, background leakage), real variability and "
                   "thermal time constant")
    br = threshold_bridge(df)
    chk.table(br, "threshold_bridge",
              desc="Predicted 95 % width of Δ against the placebo range width "
                   "(with a diagnostic excluding placebos within 3W after an onset)")
    st = pre_eruption_state()
    chk.table(st, "pre_eruption_state",
              desc="Percentile of the 180-day pre-onset lake state within its own record")
    if len(df):
        w = df.set_index("site")
        for s in df.site:
            r = w.loc[s]
            chk.note(f"{s}: TIR pixels {r.n_tir_pixels} · f {r.f_median} · "
                     f"measurement noise {r.sigma_meas_K} K (predicted "
                     f"{r.get('sigma_meas_predicted_K', float('nan'))}) · "
                     f"real variability {r.sigma_real_K} K · τ {r.tau_days_lo}~"
                     f"{r.tau_days_hi} d"
                     + ("  (depth is a bound)" if r.depth_is_bound else ""))
    chk.finish()


if __name__ == "__main__":
    main()
