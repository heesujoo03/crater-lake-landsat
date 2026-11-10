# -*- coding: utf-8 -*-
"""Stage 10: change in crater-lake relative temperature after each eruption onset.

Event-study design
------------------
1. The unit is one eruption onset. Onsets starting within W days of the previous
   one are merged with it (the first start date is the reference day), so that
   the pre-onset window of one onset does not overlap the post-onset window of
   the previous one.
2. Per-onset change Δ_e = median(y ∈ [t0, t0+W]) − median(y ∈ [t0−W, t0)), where
   y is the relative temperature: the crater lake minus a synthetic control,
   with the meteorologically explained part removed (`dT_syn_weather`, the
   primary estimator).
3. The site statistic is the median of the per-onset Δ_e.
4. Null distribution from same-season placebo onsets: for each real onset,
   placebo reference days from other years within ±30 days of the same day of
   year give the same Δ; the median over onsets is resampled 20 000 times.
   Placebo days are kept away from every real onset.
   - Season matching puts the annual cycle of the control term, which can enter
     a before/after difference, into the null distribution.
   - Each onset is one sample, so the scenes within an onset window are not
     counted as independent.
   An unmatched placebo (random days over the whole record) is reported for
   comparison.
5. Response curves: for each onset, relative temperature against days since
   onset, relative to the median of the preceding year (superposed epoch).
6. Heat budget: the same design applied to the net surface energy exchange
   (Q_total, MW).

Per-onset detection rule: in the 180- or 365-day window with the primary
estimator, the change exceeds every same-season placebo year (>= 5 comparison
years, >= 2 scenes in each window). Per-site summaries are written to
`site_selection.csv`.

Usage
-----
    python -m pipeline.s10_post_eruption
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline import s9b_event_stats as B9
from pipeline.common.report import StageCheck
from pipeline.config.sites import get_site

#  study lakes with eruptions inside the Landsat record
ERUPTION_SITES = ("taal", "ruapehu", "santa_ana")
WINDOWS = (30, 60, 90, 180, 365)
PRIMARY_W = 180
PRIMARY = "dT_syn_weather"
FALLBACK = "dT_syn"          # used where too few quiet scenes exist to fit the weather regression
ESTIMATORS = [("dT_syn_weather", "Synthetic control, weather-normalised"),
              ("dT_syn", "Synthetic control"),
              ("dT_simple", "Paired difference (lake - control)"),
              ("T_active_K", "Lake surface temperature"),
              ("Q_total_MW", "Net surface energy exchange (MW)")]
SEASON_HALF_WIDTH_D = 30
MIN_YEARS = 5           # minimum number of placebo years for a decision
#  Use only scenes in which the lake fills >= 70 % of the thermal pixel footprint
#  (water fraction f). At lower f the unmixing overcorrects and inflates the lake
#  temperature. Unscreened results are written as *_unscreened.csv (sensitivity).
F_MIN = 0.70
#  Minimum scenes in each of the pre- and post-onset windows. A one-scene window
#  median is that single scene, whose variance is too large for a reliable test.
MIN_SCENES = 2
#  Relaxed water-fraction screen, applied identically to all sites as a
#  sensitivity run (*_relaxed outputs).
F_RELAXED = 0.50
N_DRAW = 20000
SEED = 20260911
EPOCH_BINS = [(-365, -181), (-180, -91), (-90, -1), (0, 30), (31, 90), (91, 180),
              (181, 365), (366, 730), (731, 1095)]


# ---------------------------------------------------------------------------
#  1. Data
# ---------------------------------------------------------------------------
def load(site_key: str, f_min: float | None = F_MIN):
    """Relative-temperature series for one site.

    The synthetic control and weather regression are fitted on quiet scenes
    outside ±180 days of each catalogue event (the Stage 9b definition), so all
    tables share the same residuals. If too few quiet scenes remain, the buffer is
    reduced step by step; the buffer used is returned.
    """
    ke = list(get_site(site_key).known_events or [])
    for buf in (180, 90, 60, 30, 0):
        df = B9.frame(site_key, ke, buf, f_min=f_min)
        if df is not None and len(df) >= 20:
            return df, ke, buf
    return None, ke, None


def onsets(events, w_days: int):
    """Onset dates; onsets within w_days of the previous one are merged into it."""
    ts = sorted(pd.Timestamp(e.get("start") or e.get("date")) for e in events)
    out = []
    for t in ts:
        if out and (t - out[-1]["t0"]).days <= w_days:
            out[-1]["n_merged"] += 1
            continue
        out.append(dict(t0=t, n_merged=1))
    return out


def _window_median(dnum, y, a, b):
    m = (dnum >= a) & (dnum <= b) & np.isfinite(y)
    return (float(np.median(y[m])) if m.any() else np.nan), int(m.sum())


def before_after(dnum, y, t0: int, w: int):
    pre, npre = _window_median(dnum, y, t0 - w, t0 - 1)
    post, npost = _window_median(dnum, y, t0, t0 + w)
    return post - pre, npre, npost


# ---------------------------------------------------------------------------
#  2. Placebo null distribution
# ---------------------------------------------------------------------------
def placebo_candidates(dnum, y, real_t0, w: int):
    """All placebo reference days with their Δ, day of year, year and scene counts.

    Only days more than 2W from every real onset are used. With a W separation the
    pre-onset window of a placebo day could overlap a real post-onset window and
    put the real response (sign-reversed) into the null; with 2W the whole placebo
    window lies outside the real windows.
    """
    lo, hi = int(dnum.min()) + w, int(dnum.max()) - w
    cand = np.arange(lo, hi + 1, 3)                      # 3-day grid
    far = np.ones(len(cand), bool)
    for t in real_t0:
        far &= np.abs(cand - t) > 2 * w
    cand = cand[far]
    vals = np.full(len(cand), np.nan)
    npre = np.zeros(len(cand), int)
    npost = np.zeros(len(cand), int)
    for i, c in enumerate(cand):
        d, a, b = before_after(dnum, y, int(c), w)
        npre[i], npost[i] = a, b
        if a >= MIN_SCENES and b >= MIN_SCENES:
            vals[i] = d
    ts = pd.to_datetime(cand.astype("datetime64[D]"))
    ok = np.isfinite(vals)
    return (vals[ok], ts.dayofyear.to_numpy()[ok], ts.year.to_numpy()[ok],
            npre[ok], npost[ok])


def _matched_pool(cands, doy0, n_pre0, n_post0):
    """Placebo distribution matched to one onset.

    (a) Season: day of year within ±30 days.
    (b) Scene count: the same post-onset count when it is 1-2, otherwise within ±2.
        Window medians from fewer scenes have larger variance; without matching, a
        sparse observation would be compared with denser placebos and favour
        detection.
    Returns (full grid pool, one-per-year pool, number of years).
    """
    vals, doy, yr, npre, npost = cands
    dd = np.abs(doy - doy0)
    m = np.minimum(dd, 365 - dd) <= SEASON_HALF_WIDTH_D
    m &= (npost == n_post0) if n_post0 <= 2 else (np.abs(npost - n_post0) <= 2)
    m &= (npre >= 1) if n_pre0 <= 2 else (np.abs(npre - n_pre0) <= max(2, n_pre0 // 2))
    if m.sum() < 5:                      # fall back to season matching only
        dd = np.abs(doy - doy0)
        m = np.minimum(dd, 365 - dd) <= SEASON_HALF_WIDTH_D
    v, yy, d2 = vals[m], yr[m], np.minimum(np.abs(doy[m] - doy0),
                                           365 - np.abs(doy[m] - doy0))
    #  one value per year (the closest in day of year): windows within one year
    #  share most scenes and are not independent
    per_year = []
    for u in np.unique(yy):
        k = np.flatnonzero(yy == u)
        per_year.append(v[k[np.argmin(d2[k])]])
    return v, np.asarray(per_year, float), len(per_year)


def _p_two_sided(pool, stat):
    if len(pool) < 5 or not np.isfinite(stat):
        return np.nan
    c = float(np.median(pool))
    return float((np.sum(np.abs(pool - c) >= abs(stat - c)) + 1) / (len(pool) + 1))


def placebo_test(dnum, y, eps, w: int, seasonal: bool, rng, cands=None):
    """Two-sided placebo test of the median per-onset Δ.

    With seasonal=True each real onset draws only placebo days within ±30 days of
    its day of year."""
    real = [e for e in eps if np.isfinite(e["delta"])]
    if not real:
        return np.nan, np.nan, 0
    cands = cands if cands is not None else \
        placebo_candidates(dnum, y, [e["t0n"] for e in eps], w)
    vals, doy = cands[0], cands[1]
    pools = []
    for e in real:
        if seasonal:
            p = _matched_pool(cands, pd.Timestamp(e["t0"]).dayofyear,
                              e["n_pre"], e["n_post"])[0]
        else:
            p = vals
        if len(p) < 5:
            return np.nan, np.nan, 0
        pools.append(p)
    stat = float(np.median([e["delta"] for e in real]))
    draws = np.column_stack([p[rng.integers(len(p), size=N_DRAW)] for p in pools])
    null = np.median(draws, axis=1)
    c = float(np.median(null))
    pv = (np.sum(np.abs(null - c) >= abs(stat - c)) + 1) / (N_DRAW + 1)
    return stat, float(pv), int(min(len(p) for p in pools))


# ---------------------------------------------------------------------------
#  3. One site
# ---------------------------------------------------------------------------
def one_scene_diagnostic(f_min: float = 0.50):
    """Per-onset test results with MIN_SCENES = 1 (one scene per window allowed).

    Diagnostic for the two-scene requirement: onsets removed by that requirement
    in the main analysis are reported here with their placebo rank.
    """
    global MIN_SCENES
    keep, MIN_SCENES = MIN_SCENES, 1
    rows = []
    try:
        for site_key in ERUPTION_SITES:
            df, ke, _ = load(site_key, f_min)
            if df is None:
                continue
            dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
            t_first, t_last = int(dnum.min()), int(dnum.max())
            est = PRIMARY if df[PRIMARY].notna().sum() >= 20 else FALLBACK
            y = pd.to_numeric(df[est], errors="coerce").to_numpy(float)
            if np.isfinite(y).sum() < 20:
                continue
            for w in (PRIMARY_W, 365):
                eps = []
                for o in onsets(ke, w):
                    t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
                    if t0n - w >= t_first and t0n <= t_last:
                        eps.append(dict(t0=o["t0"], t0n=t0n))
                if not eps:
                    continue
                cands = placebo_candidates(dnum, y, [e["t0n"] for e in eps], w)
                for e in eps:
                    d, npre, npost = before_after(dnum, y, e["t0n"], w)
                    if not (npre >= 1 and npost >= 1) or not np.isfinite(d):
                        continue
                    pool, pool_yr, n_year = _matched_pool(
                        cands, e["t0"].dayofyear, npre, npost)
                    rows.append(dict(
                        site=site_key, window_d=w, estimator=est,
                        onset=str(e["t0"].date()), f_min=f_min,
                        n_pre=npre, n_post=npost, delta=round(float(d), 3),
                        placebo_n_years=n_year,
                        p_placebo_year=_p_two_sided(pool_yr, d),
                        n_exceeding_years=int(np.sum(
                            np.abs(pool_yr - np.median(pool_yr)) >=
                            abs(d - np.median(pool_yr)))) if n_year else 0))
    finally:
        MIN_SCENES = keep
    return pd.DataFrame(rows)


def run_site(site_key: str, f_min: float | None = F_MIN):
    df, ke, buf = load(site_key, f_min)
    if df is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dict(site=site_key)
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    t_first, t_last = int(dnum.min()), int(dnum.max())
    rng = np.random.default_rng(SEED)
    ev_rows, site_rows = [], []
    est_used = PRIMARY if df[PRIMARY].notna().sum() >= 20 else FALLBACK

    for w in WINDOWS:
        eps = []
        for o in onsets(ke, w):
            t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
            #  skip onsets whose windows fall outside the record
            if t0n - w < t_first or t0n > t_last:
                continue
            eps.append(dict(t0=o["t0"], t0n=t0n, n_merged=o["n_merged"]))
        for col, label in ESTIMATORS:
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            if np.isfinite(y).sum() < 20:
                continue
            ee = []
            cands = placebo_candidates(dnum, y, [e["t0n"] for e in eps], w)
            for e in eps:
                d, npre, npost = before_after(dnum, y, e["t0n"], w)
                ok = bool(npre >= MIN_SCENES and npost >= MIN_SCENES)
                ee.append(dict(e, delta=(d if ok else np.nan), n_pre=npre,
                               n_post=npost))
                #  per-onset p: extremeness of Δ within the same-season placebo
                #  distribution (used by select())
                p_one, pool_n, q_lo, q_hi = np.nan, 0, np.nan, np.nan
                p_year, n_year = np.nan, 0
                if ok:
                    pool, pool_yr, n_year = _matched_pool(
                        cands, e["t0"].dayofyear, npre, npost)
                    pool_n = int(len(pool))
                    if pool_n >= 10:
                        p_one = _p_two_sided(pool, d)
                        q_lo, q_hi = np.percentile(pool, [2.5, 97.5])
                    #  primary p from the one-per-year pool; neighbouring grid windows
                    #  share most scenes, so the grid pool understates p
                    p_year = _p_two_sided(pool_yr, d)
                ev_rows.append(dict(site=site_key, window_d=w, estimator=col,
                                    onset=str(e["t0"].date()),
                                    n_merged=e["n_merged"], n_pre=npre,
                                    n_post=npost,
                                    delta=(round(d, 3) if ok else np.nan),
                                    placebo_n=pool_n, placebo_n_years=n_year,
                                    p_placebo_year=(round(p_year, 5)
                                                    if np.isfinite(p_year) else np.nan),
                                    placebo_q025=(round(q_lo, 3) if np.isfinite(q_lo) else np.nan),
                                    placebo_q975=(round(q_hi, 3) if np.isfinite(q_hi) else np.nan),
                                    p_placebo_season=(round(p_one, 5) if np.isfinite(p_one) else np.nan)))
            good = [e for e in ee if np.isfinite(e["delta"])]
            stat_s, p_s, npool_s = placebo_test(dnum, y, ee, w, True, rng, cands)
            stat_u, p_u, _ = placebo_test(dnum, y, ee, w, False, rng, cands)
            deltas = np.array([e["delta"] for e in good])
            site_rows.append(dict(
                site=site_key, window_d=w, estimator=col, estimator_label=label,
                is_primary=(col == est_used), fit_buffer_d=buf,
                n_eruptions_in_record=len(eps), n_eruptions_tested=len(good),
                n_scenes_pre=int(sum(e["n_pre"] for e in good)),
                n_scenes_post=int(sum(e["n_post"] for e in good)),
                median_delta=(round(stat_s, 3) if np.isfinite(stat_s) else np.nan),
                n_positive=int((deltas > 0).sum()), n_negative=int((deltas < 0).sum()),
                p_placebo_season=(round(p_s, 5) if np.isfinite(p_s) else np.nan),
                p_placebo_any=(round(p_u, 5) if np.isfinite(p_u) else np.nan),
                n_placebo_min=npool_s))

    ev = pd.DataFrame(ev_rows)
    st = pd.DataFrame(site_rows)

    #  response curves: primary estimator, relative to the preceding-year median
    y = pd.to_numeric(df[est_used], errors="coerce").to_numpy(float)
    ep_rows = []
    for o in onsets(ke, PRIMARY_W):
        t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
        if t0n - 365 < t_first or t0n > t_last:
            continue
        base, nb = _window_median(dnum, y, t0n - 365, t0n - 1)
        if not nb:
            continue
        for a, b in EPOCH_BINS:
            v, n = _window_median(dnum, y, t0n + a, t0n + b)
            ep_rows.append(dict(site=site_key, onset=str(o["t0"].date()),
                                bin_lo_d=a, bin_hi_d=b, n_scenes=n,
                                rel_K=(round(v - base, 3) if n else np.nan)))
        #  individual scenes (n_scenes = -1) for plotting
        m = (dnum >= t0n - 365) & (dnum <= t0n + 1095) & np.isfinite(y)
        for di, yi in zip(dnum[m] - t0n, y[m]):
            ep_rows.append(dict(site=site_key, onset=str(o["t0"].date()),
                                bin_lo_d=int(di), bin_hi_d=int(di), n_scenes=-1,
                                rel_K=round(float(yi - base), 3)))
    epo = pd.DataFrame(ep_rows)
    meta = dict(site=site_key, estimator_used=est_used, fit_buffer_d=buf, f_min=f_min,
                n_scenes=int(len(df)), record=(str(df["date"].min().date()),
                                               str(df["date"].max().date())))
    return st, ev, epo, meta


# ---------------------------------------------------------------------------
#  4. Per-site detection summary
# ---------------------------------------------------------------------------
def select(st: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Per-site summary of detected onsets (primary estimator, 180- or 365-day window).

    A site is flagged when at least one onset exceeds every same-season placebo
    year. The decision is per onset rather than per site median, because responses
    can differ between onsets and pooling would average them out; the site median
    and its placebo p are reported alongside.
    """
    pr = st[st["is_primary"] & st["window_d"].isin([PRIMARY_W, 365])].copy()
    pr = pr.sort_values("window_d").drop_duplicates("site", keep="first")
    rows = []
    for _, r in pr.iterrows():
        e = ev[(ev["site"] == r["site"]) & ev["window_d"].isin([PRIMARY_W, 365])
               & (ev["estimator"] == r["estimator"]) & ev["delta"].notna()]
        #  Detection: the observed change is more extreme than every year of the
        #  one-per-year placebo pool (rank 1, i.e. the minimum attainable p), with
        #  at least MIN_YEARS years.
        hit = e[(e["placebo_n_years"] >= MIN_YEARS)
                & (np.abs(e["p_placebo_year"]
                          - 1.0 / (e["placebo_n_years"] + 1)) < 2e-4)]
        #  sign agreement across other windows (informative only)
        other = st[(st["site"] == r["site"]) & st["is_primary"]
                   & (st["window_d"] != PRIMARY_W) & st["median_delta"].notna()]
        agree = int((np.sign(other["median_delta"]) == np.sign(r["median_delta"])).sum())
        rows.append(dict(site=r["site"], estimator=r["estimator"],
                         n_eruptions_tested=r["n_eruptions_tested"],
                         n_eruptions_detected=int(len(hit)),
                         detected_onsets="|".join(
                             f"{o} {int(w)} d ({d:+.2f} K, p={p:.3f}, {int(n)} yr)"
                             for o, w, d, p, n in
                             zip(hit["onset"], hit["window_d"], hit["delta"],
                                 hit["p_placebo_year"], hit["placebo_n_years"])),
                         #  multiplicity: expected number of rank-1 onsets by chance
                         expected_by_chance=round(float(sum(
                             1.0 / (n + 1) for n in
                             e.loc[e["placebo_n_years"] >= MIN_YEARS,
                                   "placebo_n_years"])), 2),
                         median_delta_K=r["median_delta"],
                         n_positive=r["n_positive"], n_negative=r["n_negative"],
                         p_site_placebo_season=r["p_placebo_season"],
                         p_site_placebo_any=r["p_placebo_any"],
                         windows_same_sign=f"{agree}/{len(other)}",
                         selected=bool(len(hit) >= 1)))
    for k in ERUPTION_SITES:
        if k not in set(pr["site"]):
            rows.append(dict(site=k, estimator=PRIMARY, n_eruptions_tested=0,
                             n_eruptions_detected=0, detected_onsets="",
                             median_delta_K=np.nan, n_positive=0, n_negative=0,
                             p_site_placebo_season=np.nan, p_site_placebo_any=np.nan,
                             windows_same_sign="0/0", selected=False))
    out = pd.DataFrame(rows)
    return out.sort_values(["selected", "n_eruptions_detected",
                            "p_site_placebo_season"],
                           ascending=[False, False, True])


def insitu_check(site_key: str = "ruapehu") -> pd.DataFrame:
    """Before/after change of the in-situ lake temperature (Ruapehu) for each onset.

    Compares the satellite before/after change with the same windows of the lake
    thermometer record. Before 2009 the in-situ record consists of 1-2 samples per
    month, so window sample counts are small; they are reported.
    """
    from pipeline.insitu_validate import _insitu_daily
    ins = _insitu_daily(site_key)
    if ins.empty:
        return pd.DataFrame()
    ins.index = ins.index.tz_localize(None)
    df, ke, _ = load(site_key, F_MIN)
    rows = []
    for w in WINDOWS:
        for o in onsets(ke, w):
            t = o["t0"]
            pre = ins[(ins.index >= t - pd.Timedelta(days=w)) & (ins.index < t)]
            post = ins[(ins.index >= t) & (ins.index <= t + pd.Timedelta(days=w))]
            sat = {}
            for col in (PRIMARY, "T_active_K"):
                y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
                dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
                t0n = int(np.datetime64(t.date(), "D").astype("int64"))
                d, a, b = before_after(dnum, y, t0n, w)
                sat[col] = (round(d, 3) if a and b else np.nan)
                sat["n_sat_pre"], sat["n_sat_post"] = a, b
            rows.append(dict(site=site_key, window_d=w, onset=str(t.date()),
                             n_insitu_pre=len(pre), n_insitu_post=len(post),
                             insitu_pre_C=(round(pre.median(), 2) if len(pre) else np.nan),
                             insitu_post_C=(round(post.median(), 2) if len(post) else np.nan),
                             insitu_delta_K=(round(post.median() - pre.median(), 3)
                                             if len(pre) and len(post) else np.nan),
                             sat_delta_rel_K=sat[PRIMARY],
                             sat_delta_lst_K=sat["T_active_K"],
                             n_sat_pre=sat["n_sat_pre"], n_sat_post=sat["n_sat_post"]))
    return pd.DataFrame(rows)


def insitu_heating_check(site_key: str = "ruapehu") -> pd.DataFrame:
    """Relative temperature during the thermometer-defined heating episodes (Ruapehu).

    Applies the Stage 9b statistics (circular window-shift p, 90-day block
    bootstrap) to the relative temperature with and without the water-fraction
    screen (f >= F_MIN), so the validated quantity matches the analysed one.
    """
    ev = B9.insitu_heating_events(site_key)
    rows = []
    for tag, fm in (("screened", F_MIN), ("unscreened", None)):
        df = B9.frame(site_key, ev, 0, f_min=fm)
        if df is None:
            continue
        y = pd.to_numeric(df[PRIMARY], errors="coerce").to_numpy(float)
        dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
        wins = [(int(np.datetime64(e["start"], "D").astype("int64")),
                 int(np.datetime64(e["end"], "D").astype("int64"))) for e in ev]
        unrest = B9._labels_from_windows(dnum, wins)
        quiet = df["_quiet"].to_numpy(bool) & ~unrest
        ok = np.isfinite(y)
        a, b = y[quiet & ok], y[unrest & ok]
        _, p_shift, _ = B9.circular_shift_p(df["date"], y, wins)
        lo, hi = B9.block_bootstrap_ci(df["date"], y, unrest)
        #  per-episode median: how many are positive
        per = []
        for s0, s1 in wins:
            m = (dnum >= s0) & (dnum <= s1) & ok
            if m.any():
                per.append(float(np.median(y[m])) - float(np.median(a)))
        rows.append(dict(site=site_key, screen=tag, f_min=fm, n_scenes=int(ok.sum()),
                         n_episodes=len(ev), n_episodes_with_scenes=len(per),
                         n_episodes_positive=int(sum(v > 0 for v in per)),
                         diff_K=round(float(np.median(b) - np.median(a)), 3),
                         ci_low=lo, ci_high=hi, p_window_shift=round(p_shift, 5),
                         n_quiet=int(len(a)), n_heating=int(len(b))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=list(ERUPTION_SITES))
    a = ap.parse_args()
    chk = StageCheck(stage="10", title="Before/after relative temperature against placebo onsets",
                     desc="Compares the W days before and after each onset and builds "
                          "the distribution of the same statistic from same-season "
                          "placebo onsets.")
    S_, E_, P_, M_ = [], [], [], []
    for s in a.sites:
        st, ev, epo, meta = run_site(s, F_MIN)
        M_.append(meta)
        if len(st):
            S_.append(st); E_.append(ev); P_.append(epo)
        print(f"  {s}: {meta}")
    st = pd.concat(S_, ignore_index=True)
    chk.table(st, "post_eruption_contrast",
              desc=f"Site x window x estimator: median per-onset Δ and placebo p (f≥{F_MIN})")
    chk.table(pd.concat(E_, ignore_index=True), "per_eruption_delta",
              desc=f"Before/after change for each onset (f≥{F_MIN})")
    chk.table(pd.concat(P_, ignore_index=True), "epoch_response",
              desc="Superposed-epoch response curves (preceding-year median = 0)")
    #  sensitivity: no water-fraction screen
    Su, Eu = [], []
    for s in a.sites:
        stu, evu, _, _ = run_site(s, None)
        if len(stu):
            Su.append(stu); Eu.append(evu)
    chk.table(pd.concat(Su, ignore_index=True), "post_eruption_contrast_unscreened",
              desc="Sensitivity: all scenes, no water-fraction screen")
    chk.table(pd.concat(Eu, ignore_index=True), "per_eruption_delta_unscreened",
              desc="Sensitivity: all scenes, no water-fraction screen")
    ev = pd.concat(E_, ignore_index=True)
    sel = select(st, ev)
    chk.table(sel, "site_selection", desc=f"Per-site detection summary (f≥{F_MIN})")
    chk.json_(M_, "site_meta", desc="Per-site estimator and fit buffer")

    #  ── relaxed water-fraction screen, identical for all sites ──────────
    Sr, Er, Pr, Mr = [], [], [], []
    for s_ in a.sites:
        str_, evr, epr, mr = run_site(s_, F_RELAXED)
        Mr.append(mr)
        if len(str_):
            Sr.append(str_); Er.append(evr); Pr.append(epr)
    st_r = pd.concat(Sr, ignore_index=True)
    ev_r = pd.concat(Er, ignore_index=True)
    ep_r = pd.concat(Pr, ignore_index=True)
    chk.table(st_r, "post_eruption_contrast_relaxed", desc=f"Relaxed screen f≥{F_RELAXED}")
    chk.table(ev_r, "per_eruption_delta_relaxed", desc=f"Relaxed screen f≥{F_RELAXED}")
    chk.table(ep_r, "epoch_response_relaxed", desc=f"Relaxed screen f≥{F_RELAXED}")
    sel_r = select(st_r, ev_r)
    chk.table(sel_r, "site_selection_relaxed", desc=f"Per-site detection summary, relaxed screen f≥{F_RELAXED}")
    chk.json_(Mr, "site_meta_relaxed", desc="Per-site estimator, relaxed screen")

    #  ── combined set: primary-screen values, relaxed values only for sites
    #     detected under the relaxed screen alone ──
    prim = set(sel.loc[sel["selected"], "site"])
    added = [k for k in sel_r.loc[sel_r["selected"], "site"] if k not in prim]
    ev_p = pd.concat([ev[ev["site"].isin(prim)].assign(f_min=F_MIN),
                      ev_r[ev_r["site"].isin(added)].assign(f_min=F_RELAXED)],
                     ignore_index=True)
    ep_all = pd.concat(P_, ignore_index=True)
    ep_p = pd.concat([ep_all[ep_all["site"].isin(prim)],
                      ep_r[ep_r["site"].isin(added)]], ignore_index=True)
    st_p = pd.concat([st[st["site"].isin(prim)].assign(f_min=F_MIN),
                      st_r[st_r["site"].isin(added)].assign(f_min=F_RELAXED)],
                     ignore_index=True)
    meta_p = ([dict(m, f_min=F_MIN, entered="primary") for m in M_ if m.get("site") in prim]
              + [dict(m, f_min=F_RELAXED, entered="relaxed") for m in Mr
                 if m.get("site") in added])
    chk.table(ev_p, "per_eruption_delta_paper", desc="Combined set: per onset")
    chk.table(ep_p, "epoch_response_paper", desc="Combined set: response curves")
    chk.table(st_p, "post_eruption_contrast_paper", desc="Combined set: per site")
    chk.json_(meta_p, "site_meta_paper", desc="Combined set: per-site screen and estimator")
    sel_all = sel.merge(sel_r[["site", "n_eruptions_tested", "n_eruptions_detected",
                               "detected_onsets", "selected"]],
                        on="site", how="outer", suffixes=("", "_relaxed"))
    sel_all["entered"] = np.where(sel_all["site"].isin(prim), "primary",
                                  np.where(sel_all["site"].isin(added), "relaxed", "no"))
    chk.table(sel_all, "site_selection_both", desc="Primary and relaxed screens side by side")
    osd = one_scene_diagnostic()
    if len(osd):
        chk.table(osd, "one_scene_diagnostic",
                  desc="Results if one scene per window were allowed; diagnostic "
                       "for the two-scene requirement")
    chk.note(f"Detected, primary screen: {sorted(prim)} · added by relaxed screen: {added}")
    ic = insitu_check("ruapehu")
    if len(ic):
        chk.table(ic, "insitu_before_after",
                  desc="Ruapehu: before/after change from thermometer and satellite")
    #  Taal change point: a single step fitted without the event list (Stage 9j
    #  function applied to the screened scenes)
    from pipeline import s9j_within_volcano as WJ
    cp = WJ.taal_changepoint(f_min=F_MIN)
    if cp:
        chk.table(pd.DataFrame([dict(quantity=k, **v) for k, v in cp.items()]),
                  "taal_changepoint_screened", desc=f"Taal change point (f≥{F_MIN})")
    ih = insitu_heating_check("ruapehu")
    if len(ih):
        chk.table(ih, "insitu_heating_validation",
                  desc="Ruapehu thermometer heating episodes: screened and unscreened scenes")
    chk.param(windows=list(WINDOWS), primary_window_d=PRIMARY_W, primary=PRIMARY,
              fallback=FALLBACK, season_half_width_d=SEASON_HALF_WIDTH_D,
              n_draw=N_DRAW, seed=SEED, f_min=F_MIN, f_relaxed=F_RELAXED,
              min_scenes=MIN_SCENES, min_years=MIN_YEARS,
              selection_rule="180 or 365 d window, primary estimator, >=2 scenes in "
                             "each window: at least one eruption whose before-after "
                             "change exceeds every same-season placebo year "
                             "(>=5 comparable years)")
    chk.gate("SITES_SELECTED", int(sel["selected"].sum()) >= 1,
             f"{int(sel['selected'].sum())} sites with a detected onset: "
             f"{', '.join(sel.loc[sel['selected'], 'site'])}", blocking=False)
    chk.finish()


if __name__ == "__main__":
    main()
