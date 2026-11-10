# -*- coding: utf-8 -*-
"""Stage 10p: pre-onset change of crater-lake relative temperature.

`s10_post_eruption` measures the change after each onset. This module measures
the change over the windows preceding the onset t0:

    Δ_pre = median( y in [t0−W, t0) ) − median( y in [t0−2W, t0−W) )

which is `s10_post_eruption.before_after` evaluated at the reference day t0 − W.
Only the window definition is new; placebo candidates, season and scene-count
matching, the year-placebo rank p and the minimum number of comparison years all
reuse the `s10_post_eruption` functions, so pre- and post-onset tests share one
design.

Caveats
  · The tail of an earlier onset can fall inside the pre-onset windows. Onsets
    within W are merged as in the post-eruption analysis, but if an earlier onset
    lies within 2W, Δ_pre partly measures its relaxation; `prev_gap_d` records
    the gap to the previous onset.
  · The values are retrospective and are not a forecast; forecasting would
    require a false-alarm rate, which this module does not estimate.

Outputs: per_eruption_pre and pre_vs_post tables in the Stage 10p record folder.

    python -m pipeline.s10_pre_eruption
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common.report import StageCheck
from pipeline.config import settings as S
from pipeline import s10_post_eruption as P

#  identical settings to the post-eruption analysis so the two tests are comparable
WINDOWS = P.WINDOWS
PRIMARY_W = P.PRIMARY_W
MIN_SCENES = P.MIN_SCENES
MIN_YEARS = P.MIN_YEARS
F_MIN = P.F_MIN
ESTIMATORS = P.ESTIMATORS
PRIMARY, FALLBACK = P.PRIMARY, P.FALLBACK
SITES = P.ERUPTION_SITES


def run_site(site_key: str, f_min: float | None = F_MIN):
    """Δ_pre and its placebo rank p for every onset of one site."""
    df, ke, _buf = P.load(site_key, f_min)
    if df is None:
        return pd.DataFrame()
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    t_first, t_last = int(dnum.min()), int(dnum.max())
    rows = []

    for w in WINDOWS:
        eps, prev = [], None
        for o in P.onsets(ke, w):
            t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
            #  Shift the reference day to t0 − W so that before_after compares
            #  [t0−W, t0) with [t0−2W, t0−W), i.e. Δ_pre.
            ref = t0n - w
            #  both windows must lie within the observed record
            if ref - w < t_first or ref > t_last:
                prev = t0n
                continue
            eps.append(dict(t0=o["t0"], t0n=t0n, ref=ref, n_merged=o["n_merged"],
                            prev_gap_d=(t0n - prev) if prev is not None else np.nan))
            prev = t0n

        for col, label in ESTIMATORS:
            if col not in df.columns:
                continue
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            if np.isfinite(y).sum() < 20:
                continue
            #  placebo candidates kept 2W away from the shifted reference days
            cands = P.placebo_candidates(dnum, y, [e["ref"] for e in eps], w)
            for e in eps:
                d, npre, npost = P.before_after(dnum, y, e["ref"], w)
                ok = bool(npre >= MIN_SCENES and npost >= MIN_SCENES)
                p_one, pool_n, q_lo, q_hi = np.nan, 0, np.nan, np.nan
                p_year, n_year = np.nan, 0
                if ok:
                    pool, pool_yr, n_year = P._matched_pool(
                        cands, pd.Timestamp(e["ref"], unit="D").dayofyear,
                        npre, npost)
                    pool_n = int(len(pool))
                    if pool_n >= 10:
                        p_one = P._p_two_sided(pool, d)
                        q_lo, q_hi = np.percentile(pool, [2.5, 97.5])
                    p_year = P._p_two_sided(pool_yr, d)
                rows.append(dict(
                    site=site_key, window_d=w, estimator=col,
                    onset=str(e["t0"].date()), n_merged=e["n_merged"],
                    prev_gap_d=e["prev_gap_d"],
                    #  early window = [t0−2W, t0−W), late window = [t0−W, t0)
                    n_early=npre, n_late=npost,
                    delta_pre=(round(d, 3) if ok else np.nan),
                    placebo_n=pool_n, placebo_n_years=n_year,
                    p_placebo_year=(round(p_year, 5) if np.isfinite(p_year) else np.nan),
                    placebo_q025=(round(q_lo, 3) if np.isfinite(q_lo) else np.nan),
                    placebo_q975=(round(q_hi, 3) if np.isfinite(q_hi) else np.nan),
                    p_placebo_season=(round(p_one, 5) if np.isfinite(p_one) else np.nan),
                    f_min=(f_min if f_min is not None else np.nan)))
    return pd.DataFrame(rows)


def resolved(r) -> bool:
    """Same rule as the post-eruption analysis: exceeds every placebo year, >= MIN_YEARS years."""
    n = r.get("placebo_n_years", 0)
    p = r.get("p_placebo_year", np.nan)
    return bool(np.isfinite(p) and n >= MIN_YEARS
                and abs(p - 1.0 / (n + 1)) < 2e-4)


def combine(pre: pd.DataFrame, sites) -> pd.DataFrame:
    """Pre-onset (Δ_pre) and post-onset (Δ_post) change side by side, with a sign-reversal flag."""
    post = pd.read_csv(S.records_dir("10") / "per_eruption_delta_paper.csv")
    out = []
    for s in sites:
        a = pre[(pre.site == s) & (pre.estimator.isin([PRIMARY, FALLBACK]))]
        b = post[(post.site == s) & (post.estimator.isin([PRIMARY, FALLBACK]))]
        for w in (PRIMARY_W, 365):
            aw = a[a.window_d == w].set_index("onset")
            bw = b[b.window_d == w].set_index("onset")
            for o in sorted(set(aw.index) | set(bw.index)):
                ra = aw.loc[o] if o in aw.index else None
                rb = bw.loc[o] if o in bw.index else None
                if ra is not None and isinstance(ra, pd.DataFrame):
                    ra = ra.iloc[0]
                if rb is not None and isinstance(rb, pd.DataFrame):
                    rb = rb.iloc[0]
                dpre = float(ra["delta_pre"]) if ra is not None else np.nan
                dpost = float(rb["delta"]) if rb is not None else np.nan
                out.append(dict(
                    site=s, onset=o, window_d=w,
                    delta_pre=dpre, delta_post=dpost,
                    pre_resolved=(resolved(ra) if ra is not None else False),
                    post_resolved=(
                        bool(np.isfinite(rb["p_placebo_year"])
                             and rb["placebo_n_years"] >= MIN_YEARS
                             and abs(rb["p_placebo_year"]
                                     - 1.0 / (rb["placebo_n_years"] + 1)) < 2e-4)
                        if rb is not None else False),
                    p_pre=(ra["p_placebo_year"] if ra is not None else np.nan),
                    p_post=(rb["p_placebo_year"] if rb is not None else np.nan),
                    prev_gap_d=(ra["prev_gap_d"] if ra is not None else np.nan),
                    signs_opposite=(bool(np.isfinite(dpre) and np.isfinite(dpost)
                                         and dpre * dpost < 0))))
    return pd.DataFrame(out)


def main():
    chk = StageCheck(stage="10p", title="Pre-onset change test",
                     desc="Compares the two windows preceding each onset with the "
                          "post-eruption test design, reference day shifted to t0−W.")
    frames = [run_site(s) for s in SITES]
    pre = pd.concat([f for f in frames if len(f)], ignore_index=True)
    chk.table(pre, "per_eruption_pre",
              desc="Per-onset Δ_pre = median[t0−W,t0) − median[t0−2W,t0−W)")
    cmb = combine(pre, SITES)
    chk.table(cmb, "pre_vs_post",
              desc="Pre-onset and post-onset change of the same onset, with sign reversal")

    res = pre[pre.estimator.eq(PRIMARY) & pre.apply(resolved, axis=1)]
    chk.note(f"{len(res)} onsets with a resolved pre-onset change "
             f"(exceeds every placebo year, comparison years ≥ {MIN_YEARS})")
    chk.finish()


if __name__ == "__main__":
    main()
