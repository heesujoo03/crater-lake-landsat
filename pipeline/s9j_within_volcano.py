# -*- coding: utf-8 -*-
"""Single-step change point of the Taal lake temperature series.

`taal_changepoint` estimates one step in the relative-temperature series (and in
the lake surface temperature) without using the eruption list, gives a
block-bootstrap confidence interval for its date (90-day blocks), reports the
slope after the step, and re-estimates the step controlling for season and
spacecraft. s10_post_eruption applies it to the water-fraction-screened scenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import s9b_event_stats as EV
from pipeline.config import settings as S

SEED = 20260831


def _definition(site_key: str, name: str):
    for D in EV.definitions(site_key):
        if D["name"] == name:
            return D
    return None


# ---------------------------------------------------------------------------
#  Taal — change-point estimation
# ---------------------------------------------------------------------------
def _step_changepoint(t: np.ndarray, y: np.ndarray, edge: int = 12):
    """Split index minimising the residual sum of squares of a single-step model.

    Returns: (split index, mean before, mean after, fraction of SSE explained)
    """
    n = len(y)
    if n < 2 * edge + 2:
        return None
    cs = np.cumsum(y)
    cs2 = np.cumsum(y ** 2)
    tot = cs[-1]
    best, best_sse = None, np.inf
    for k in range(edge, n - edge):
        n1, n2 = k, n - k
        s1, s2 = cs[k - 1], tot - cs[k - 1]
        sse = (cs2[-1] - s1 ** 2 / n1 - s2 ** 2 / n2)
        if sse < best_sse:
            best_sse, best = sse, k
    k = best
    m1 = float(y[:k].mean())
    m2 = float(y[k:].mean())
    sse0 = float(((y - y.mean()) ** 2).sum())
    return k, m1, m2, float(1 - best_sse / sse0) if sse0 > 0 else np.nan


def taal_changepoint(site_key="taal", definition="eruption_pm180d",
                     f_min=None) -> dict:
    D = _definition(site_key, definition)
    if D is None:
        return {}
    df = EV.frame(site_key, D["events"], D["buffer"], f_min=f_min)
    if df is None:
        return {}
    out = {}
    for col in ("dT_syn_weather", "dT_syn", "T_active_K"):
        v = df[["date", col]].dropna().sort_values("date").reset_index(drop=True)
        if len(v) < 40:
            continue
        y = pd.to_numeric(v[col], errors="coerce").to_numpy(float)
        t = v["date"].to_numpy("datetime64[D]")
        got = _step_changepoint(t, y)
        if got is None:
            continue
        k, m1, m2, r2 = got
        #  Confidence interval: block bootstrap (90-day blocks), re-estimating the change point each time.
        rng = np.random.default_rng(SEED)
        blk = (v["date"].to_numpy("datetime64[D]").astype("int64")
               - int(t[0].astype("int64"))) // 90
        ids, inv = np.unique(blk, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(ids))]
        taus = []
        for _ in range(500):
            pick = rng.integers(len(ids), size=len(ids))
            idx = np.sort(np.concatenate([members[p] for p in pick]))
            g = _step_changepoint(t[idx], y[idx])
            if g is not None:
                taus.append(t[idx][g[0]])
        taus = np.array(taus, dtype="datetime64[D]")
        out[col] = dict(
            n=len(v),
            changepoint=str(pd.Timestamp(t[k]).date()),
            ci_low=str(pd.Timestamp(np.percentile(
                taus.astype("int64"), 2.5).astype("datetime64[D]")).date())
            if len(taus) else "",
            ci_high=str(pd.Timestamp(np.percentile(
                taus.astype("int64"), 97.5).astype("datetime64[D]")).date())
            if len(taus) else "",
            level_before=round(m1, 3), level_after=round(m2, 3),
            step=round(m2 - m1, 3),
            median_before=round(float(np.median(y[:k])), 3),
            median_after=round(float(np.median(y[k:])), 3),
            variance_explained=round(r2, 3) if np.isfinite(r2) else np.nan,
            n_before=int(k), n_after=int(len(y) - k))
        #  Slope after the change point: a step or a continuing trend?
        tt = (t[k:].astype("int64") - int(t[k].astype("int64"))) / 365.25
        if len(tt) >= 8:
            b = np.polyfit(tt, y[k:], 1)
            out[col]["slope_after_K_per_yr"] = round(float(b[0]), 3)
    #  ── Does the step persist after controlling for season and spacecraft? ──
    p = S.PROCESSED_DIR / f"{site_key}_deltaT.csv"
    if p.exists() and "dT_syn_weather" in out:
        dd = pd.read_csv(p)
        dd["date"] = pd.to_datetime(dd["datetime"], utc=True, format="mixed") \
            .dt.tz_localize(None).dt.normalize()
        keep = [c for c in ("date", "spacecraft") if c in dd.columns]
        if len(keep) == 2:
            v = df[["date", "dT_syn_weather"]].dropna().merge(
                dd[keep].drop_duplicates("date"), on="date", how="left")
            cp = pd.Timestamp(out["dT_syn_weather"]["changepoint"])
            v["after"] = (v["date"] >= cp).astype(float)
            v["month"] = v["date"].dt.month
            X = [np.ones(len(v)), v["after"].to_numpy(float),
                 np.sin(2 * np.pi * v["month"] / 12),
                 np.cos(2 * np.pi * v["month"] / 12)]
            names = ["const", "after", "sin", "cos"]
            for sc in sorted(v["spacecraft"].dropna().unique())[1:]:
                X.append((v["spacecraft"] == sc).to_numpy(float))
                names.append(f"sc:{sc}")
            Xm = np.column_stack(X)
            yv = v["dT_syn_weather"].to_numpy(float)
            good = np.isfinite(yv) & np.isfinite(Xm).all(axis=1)
            if good.sum() > len(names) + 10:
                beta, *_ = np.linalg.lstsq(Xm[good], yv[good], rcond=None)
                res = yv[good] - Xm[good] @ beta
                se = np.sqrt(np.sum(res ** 2) / (good.sum() - len(names))
                             * np.diag(np.linalg.pinv(Xm[good].T @ Xm[good])))
                out["adjusted_step"] = dict(
                    step_K=round(float(beta[1]), 3),
                    se_K=round(float(se[1]), 3),
                    ci_low=round(float(beta[1] - 1.96 * se[1]), 3),
                    ci_high=round(float(beta[1] + 1.96 * se[1]), 3),
                    terms="|".join(names), n=int(good.sum()))
    return out


