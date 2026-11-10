# -*- coding: utf-8 -*-
"""Placebo design, observation independence, detection power and false-alarm rate.

Pipeline modules are imported and not modified; outputs are written only to
results/11_analysis/placebo_design/.

    python -m pipeline.analysis.placebo_design
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import s10_post_eruption as P
from pipeline.config import settings as S

OUT = S.RECORDS_DIR / "11_analysis" / "placebo_design"
OUT.mkdir(parents=True, exist_ok=True)
SITES = ("taal", "santa_ana", "ruapehu")
POST_W = (180, 365)
PRE_W = (90, 180, 365)
EST = P.PRIMARY
REC_POST = S.RECORDS_DIR / "10_post_eruption" / "per_eruption_delta_paper.csv"
REC_PRE = S.RECORDS_DIR / "10_pre_eruption" / "per_eruption_pre.csv"


# ---------------------------------------------------------------------------
#  data
# ---------------------------------------------------------------------------
def _scene_ids(site):
    d = pd.read_csv(S.PROCESSED_DIR / f"{site}_deltaT.csv")
    d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
    return d[["t", "scene_id", "spacecraft"]]


def load_series(site, mode="asis"):
    """mode: 'asis' (primary series), 'collapse' (median per date after fitting),
    'refit' (collapse per date *before* NNLS + weather fit)."""
    if mode == "refit":
        tmp = Path(tempfile.mkdtemp(prefix="pd_refit_"))
        src = S.PROCESSED_DIR
        for f in (f"{site}_dataset.csv", f"{site}_energy_balance.csv"):
            if (src / f).exists():
                shutil.copy(src / f, tmp / f)
        d = pd.read_csv(src / f"{site}_deltaT.csv")
        d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
        d = d[d["qc_flag"].eq("ok")]
        d = d[pd.to_numeric(d["f_est_act"], errors="coerce") >= P.F_MIN]
        d["_day"] = d["t"].dt.tz_localize(None).dt.normalize()
        num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        agg = {c: "median" for c in num}
        agg.update({c: "first" for c in d.columns
                    if c not in num and c not in ("_day",)})
        d = d.sort_values("t").groupby("_day", as_index=False).agg(agg)
        d = d.drop(columns=["t", "_day"], errors="ignore")
        d.to_csv(tmp / f"{site}_deltaT.csv", index=False)
        old = S.PROCESSED_DIR
        S.PROCESSED_DIR = tmp
        try:
            df, ke, buf = P.load(site, P.F_MIN)
        finally:
            S.PROCESSED_DIR = old
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        df, ke, buf = P.load(site, P.F_MIN)
    df = df.copy()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    if mode == "asis":
        df = df.merge(_scene_ids(site), on="t", how="left")
    if mode == "collapse":
        g = df.groupby("date")
        df = pd.DataFrame({
            "date": list(g.groups.keys()),
            EST: g[EST].median().to_numpy(),
            "_quiet": g["_quiet"].max().to_numpy(),
            "n_dup": g.size().to_numpy()})
    df = df.sort_values("date").reset_index(drop=True)
    return df, ke


def arrays(df):
    dnum = df["date"].to_numpy("datetime64[D]").astype("int64")
    y = pd.to_numeric(df[EST], errors="coerce").to_numpy(float)
    return dnum, y


def d2s(n):
    return str(np.datetime64(int(n), "D")) if np.isfinite(n) else ""


# ---------------------------------------------------------------------------
#  placebo machinery (copies of s10_post_eruption logic, returning dates too)
# ---------------------------------------------------------------------------
def candidates(dnum, y, real, w):
    lo, hi = int(dnum.min()) + w, int(dnum.max()) - w
    cand = np.arange(lo, hi + 1, 3)
    far = np.ones(len(cand), bool)
    for t in real:
        far &= np.abs(cand - t) > 2 * w
    cand = cand[far]
    vals = np.full(len(cand), np.nan)
    npre = np.zeros(len(cand), int)
    npost = np.zeros(len(cand), int)
    for i, c in enumerate(cand):
        d, a, b = P.before_after(dnum, y, int(c), w)
        npre[i], npost[i] = a, b
        if a >= P.MIN_SCENES and b >= P.MIN_SCENES:
            vals[i] = d
    ok = np.isfinite(vals)
    ts = pd.to_datetime(cand.astype("datetime64[D]"))
    return dict(c=cand[ok], v=vals[ok], doy=ts.dayofyear.to_numpy()[ok],
                yr=ts.year.to_numpy()[ok], npre=npre[ok], npost=npost[ok])


def _sub(C, m):
    return {k: v[m] for k, v in C.items()}


def matched(C, doy0, n_pre0, n_post0, rule="baseline"):
    """Returns (grid pool dict, per-year pool dict, fell_back flag)."""
    dd = np.abs(C["doy"] - doy0)
    season = np.minimum(dd, 365 - dd) <= P.SEASON_HALF_WIDTH_D
    m = season.copy()
    fell = False
    if rule == "baseline":
        m &= (C["npost"] == n_post0) if n_post0 <= 2 else (np.abs(C["npost"] - n_post0) <= 2)
        m &= (C["npre"] >= 1) if n_pre0 <= 2 else (
            np.abs(C["npre"] - n_pre0) <= max(2, n_pre0 // 2))
        if m.sum() < 5:
            m, fell = season.copy(), True
    elif rule == "exact_both":
        m &= (C["npre"] == n_pre0) & (C["npost"] == n_post0)
    elif rule == "tol1_both":
        m &= (np.abs(C["npre"] - n_pre0) <= 1) & (np.abs(C["npost"] - n_post0) <= 1)
    elif rule == "tol2_both":
        m &= (np.abs(C["npre"] - n_pre0) <= 2) & (np.abs(C["npost"] - n_post0) <= 2)
    elif rule == "season_only":
        pass
    g = _sub(C, m)
    d2 = np.minimum(np.abs(g["doy"] - doy0), 365 - np.abs(g["doy"] - doy0))
    keep = []
    for u in np.unique(g["yr"]):
        k = np.flatnonzero(g["yr"] == u)
        keep.append(k[np.argmin(d2[k])])
    yrp = _sub(g, np.array(keep, int)) if keep else _sub(g, np.zeros(len(g["c"]), bool))
    return g, yrp, fell


def rank_p(pool, d):
    return P._p_two_sided(np.asarray(pool, float), d)


def is_max(p, n):
    return bool(np.isfinite(p) and n >= P.MIN_YEARS and abs(p - 1.0 / (n + 1)) < 2e-4)


def q_range(pool):
    if len(pool) >= 10:
        return tuple(np.percentile(pool, [2.5, 97.5]))
    return (np.nan, np.nan)


# ---------------------------------------------------------------------------
#  event rows
# ---------------------------------------------------------------------------
def anchors(ke, w, dnum, kind):
    t_first, t_last = int(dnum.min()), int(dnum.max())
    out, prev = [], None
    for o in P.onsets(ke, w):
        t0n = int(np.datetime64(o["t0"].date(), "D").astype("int64"))
        ref = t0n if kind == "post" else t0n - w
        if ref - w < t_first or ref > t_last:
            prev = t0n
            continue
        out.append(dict(t0=o["t0"], t0n=t0n, ref=ref, n_merged=o["n_merged"],
                        prev_gap_d=(t0n - prev) if prev is not None else np.nan))
        prev = t0n
    return out


def all_onsets(ke):
    return [int(np.datetime64(pd.Timestamp(e["start"]).date(), "D").astype("int64"))
            for e in ke]


def window_desc(dnum, idx_dates, a, b):
    """max unobserved interval inside [a, b] (edges included) and between scenes."""
    u = np.unique(idx_dates)
    if len(u) == 0:
        return np.nan, np.nan
    pts = np.r_[a, u, b]
    inner = np.diff(u).max() if len(u) > 1 else np.nan
    return int(np.diff(pts).max()), (int(inner) if np.isfinite(inner) else np.nan)


def contamination(pool_c, kind, w, onsets_all):
    """per-year placebo anchors whose windows overlap an actual post-onset W window."""
    n = 0
    for c in pool_c:
        lo = c - (2 * w if kind == "pre" else w)
        hi = c - 1 if kind == "pre" else c + w
        if any((lo <= t + w) and (hi >= t) for t in onsets_all):
            n += 1
    return n


def evaluate(site, df, ke, mode, extra_pools=True):
    dnum, y = arrays(df)
    ok = np.isfinite(y)
    rows, loo_rows, pools_store = [], [], {}
    ons_all = all_onsets(ke)
    for kind, wins in (("post", POST_W), ("pre", PRE_W)):
        for w in wins:
            eps = anchors(ke, w, dnum, kind)
            if not eps:
                continue
            C = candidates(dnum, y, [e["ref"] for e in eps], w)
            for e in eps:
                ref = e["ref"]
                d, npre, npost = P.before_after(dnum, y, ref, w)
                tested = npre >= P.MIN_SCENES and npost >= P.MIN_SCENES
                mpre = (dnum >= ref - w) & (dnum <= ref - 1) & ok
                mpost = (dnum >= ref) & (dnum <= ref + w) & ok
                gp_pre, gi_pre = window_desc(dnum, dnum[mpre], ref - w, ref - 1)
                gp_post, gi_post = window_desc(dnum, dnum[mpost], ref, ref + w)
                before = dnum[(dnum < ref) & ok]
                after = dnum[(dnum >= ref) & ok]
                r = dict(site=site, analysis=kind, window_d=w, onset=str(e["t0"].date()),
                         anchor_date=d2s(ref), n_merged=e["n_merged"], variant=mode,
                         n_pre=int(npre), n_post=int(npost),
                         n_pre_unique_dates=int(len(np.unique(dnum[mpre]))),
                         n_post_unique_dates=int(len(np.unique(dnum[mpost]))),
                         last_obs_before=d2s(before.max()) if len(before) else "",
                         first_obs_after=d2s(after.min()) if len(after) else "",
                         days_last_obs_to_anchor=int(ref - before.max()) if len(before) else np.nan,
                         days_anchor_to_first_obs=int(after.min() - ref) if len(after) else np.nan,
                         max_gap_pre_incl_edges_d=gp_pre, max_gap_pre_between_obs_d=gi_pre,
                         max_gap_post_incl_edges_d=gp_post, max_gap_post_between_obs_d=gi_post,
                         pre_dates="|".join(d2s(x) for x in dnum[mpre]),
                         post_dates="|".join(d2s(x) for x in dnum[mpost]),
                         tested=bool(tested),
                         delta=(round(float(d), 3) if tested else np.nan))
                if tested:
                    doy0 = pd.Timestamp(int(ref), unit="D").dayofyear
                    rules = (("baseline", "exact_both", "tol1_both", "tol2_both")
                             if extra_pools else ("baseline",))
                    for rule in rules:
                        g, yp, fell = matched(C, doy0, npre, npost, rule)
                        p = rank_p(yp["v"], d)
                        n = len(yp["v"])
                        lo, hi = q_range(g["v"])
                        pre = "" if rule == "baseline" else rule + "_"
                        r.update({pre + "placebo_n_grid": int(len(g["v"])),
                                  pre + "placebo_n_years": n,
                                  pre + "p_rank": round(p, 5) if np.isfinite(p) else np.nan,
                                  pre + "q025": round(lo, 2) if np.isfinite(lo) else np.nan,
                                  pre + "q975": round(hi, 2) if np.isfinite(hi) else np.nan,
                                  pre + "max_exceedance": is_max(p, n)})
                        if rule == "baseline":
                            pu = ((np.sum(np.abs(yp["v"]) >= abs(d)) + 1) / (n + 1)
                                  if n >= 5 else np.nan)
                            r["p_rank_uncentred"] = round(float(pu), 5) if np.isfinite(pu) else np.nan
                            r["max_exceedance_uncentred"] = bool(n >= P.MIN_YEARS and np.all(np.abs(yp["v"]) < abs(d)))
                            r["baseline_fell_back_to_season_only"] = fell
                            r["min_attainable_p"] = round(1 / (n + 1), 4) if n else np.nan
                            r["n_pool_years_overlapping_actual_postonset"] = contamination(
                                yp["c"], kind, w, ons_all)
                            r["pool_years"] = "|".join(str(x) for x in yp["yr"])
                            pools_store[(site, kind, w, r["onset"], mode)] = (yp, g, d)
                    # 3W exclusion after any actual onset (anchors as in the primary analysis)
                    for tag, tl in (("excl3w_anchor", [e2["t0n"] for e2 in eps]),
                                    ("excl3w_allonsets", ons_all)):
                        far = np.ones(len(C["c"]), bool)
                        for t in tl:
                            far &= ~((C["c"] - t > 0) & (C["c"] - t <= 3 * w))
                        g, yp, _ = matched(_sub(C, far), doy0, npre, npost, "baseline")
                        p = rank_p(yp["v"], d)
                        n = len(yp["v"])
                        lo, hi = q_range(g["v"])
                        r.update({f"{tag}_placebo_n_grid": int(len(g["v"])),
                                  f"{tag}_placebo_n_years": n,
                                  f"{tag}_p_rank": round(p, 5) if np.isfinite(p) else np.nan,
                                  f"{tag}_q025": round(lo, 2) if np.isfinite(lo) else np.nan,
                                  f"{tag}_q975": round(hi, 2) if np.isfinite(hi) else np.nan,
                                  f"{tag}_max_exceedance": is_max(p, n)})
                    # leave-one-out (scene and date), pool fixed
                    if mode == "asis":
                        yp, g, _ = pools_store[(site, kind, w, r["onset"], mode)]
                        idx = np.flatnonzero(mpre | mpost)
                        dl = []
                        for i in idx:
                            yy = y.copy(); yy[i] = np.nan
                            dd_, a_, b_ = P.before_after(dnum, yy, ref, w)
                            ok2 = a_ >= P.MIN_SCENES and b_ >= P.MIN_SCENES
                            pp = rank_p(yp["v"], dd_) if ok2 else np.nan
                            dl.append((dd_, ok2, is_max(pp, len(yp["v"])) if ok2 else False))
                            loo_rows.append(dict(site=site, analysis=kind, window_d=w,
                                                 onset=r["onset"], dropped=d2s(dnum[i]),
                                                 side="pre" if dnum[i] < ref else "post",
                                                 delta=round(float(dd_), 3), still_tested=ok2,
                                                 p_rank_fixed_pool=pp))
                        dv = np.array([x[0] for x in dl if x[1]])
                        r.update(loo_delta_min=round(float(dv.min()), 2) if len(dv) else np.nan,
                                 loo_delta_max=round(float(dv.max()), 2) if len(dv) else np.nan,
                                 loo_n_variants=len(dl),
                                 loo_n_untestable=int(sum(not x[1] for x in dl)),
                                 loo_n_keep_max_exceedance=int(sum(x[2] for x in dl)))
                        # leave-one-date-out
                        dl2, keep2, un2 = [], 0, 0
                        for u in np.unique(dnum[idx]):
                            yy = y.copy(); yy[dnum == u] = np.nan
                            dd_, a_, b_ = P.before_after(dnum, yy, ref, w)
                            if a_ >= P.MIN_SCENES and b_ >= P.MIN_SCENES:
                                dl2.append(dd_)
                                keep2 += is_max(rank_p(yp["v"], dd_), len(yp["v"]))
                            else:
                                un2 += 1
                        r.update(lodo_delta_min=round(float(min(dl2)), 2) if dl2 else np.nan,
                                 lodo_delta_max=round(float(max(dl2)), 2) if dl2 else np.nan,
                                 lodo_n_variants=len(np.unique(dnum[idx])),
                                 lodo_n_untestable=un2, lodo_n_keep_max_exceedance=int(keep2))
                rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(loo_rows), pools_store


# ---------------------------------------------------------------------------
#  power and false alarm (illustrative)
# ---------------------------------------------------------------------------
def power_row(yp_v, steps=np.arange(0, 60.01, 0.1), obs=None):
    """Leave-one-year-out injection: add step s to each placebo year's post-window
    (Δ -> Δ + s since a uniform step shifts the post median by s) and test it
    against the remaining years. Returns smallest |s| for 80 % power, both signs,
    for the max-placebo rule and a 95th-percentile rule."""
    v = np.asarray(yp_v, float)
    n = len(v)
    if n < P.MIN_YEARS + 1:
        return {}
    out = {}
    if obs is not None:
        for rule in ("max", "q95"):
            hit = 0
            for i in range(n):
                null = np.delete(v, i); c = np.median(null); dev = np.abs(null - c)
                thr = dev.max() if rule == "max" else np.quantile(dev, 0.95)
                hit += abs(v[i] + obs - c) > thr
            out[f"power_at_observed_delta_{rule}"] = round(hit / n, 3)
    for sign in (+1, -1):
        for rule in ("max", "q95"):
            found = np.nan
            for s in steps:
                hit = 0
                for i in range(n):
                    null = np.delete(v, i)
                    c = np.median(null)
                    dev = np.abs(null - c)
                    thr = dev.max() if rule == "max" else np.quantile(dev, 0.95)
                    hit += abs(v[i] + sign * s - c) > thr
                if hit / n >= 0.8:
                    found = round(float(s), 1)
                    break
            out[f"mds80_{rule}_{'pos' if sign > 0 else 'neg'}_K"] = found
    return out


def false_alarm(site, df, ke, w, clean=False):
    """Prospective rule over all non-eruptive candidate windows (post-type, 3-d grid):
    alert if |Δ_c − med| exceeds the 95th percentile (or the maximum) of the
    same-season, count-matched per-year placebo values from *earlier* years only.
    Also a retrospective variant using all other years."""
    dnum, y = arrays(df)
    eps = anchors(ke, w, dnum, "post")
    C = candidates(dnum, y, [e["ref"] for e in eps], w)
    if clean:   # drop candidates within 3W after any catalogue onset
        far = np.ones(len(C["c"]), bool)
        for t in all_onsets(ke):
            far &= ~((C["c"] - t > 0) & (C["c"] - t <= 3 * w))
        C = _sub(C, far)
    res = {k: [0, 0] for k in ("pro_q95", "pro_max", "retro_q95", "retro_max")}
    nominal = {"pro": [], "retro": []}
    by_year = {k: {} for k in res}
    for i in range(len(C["c"])):
        doy0, yr0 = C["doy"][i], C["yr"][i]
        for tag, m in (("pro", C["yr"] < yr0), ("retro", C["yr"] != yr0)):
            g, yp, fell = matched(_sub(C, m), doy0, C["npre"][i], C["npost"][i], "baseline")
            if fell or len(yp["v"]) < P.MIN_YEARS:
                continue
            c = np.median(yp["v"])
            dev = np.abs(yp["v"] - c)
            x = abs(C["v"][i] - c)
            nominal[tag].append(1.0 / (len(yp["v"]) + 1))
            for rule, thr in (("q95", np.quantile(dev, 0.95)), ("max", dev.max())):
                k = f"{tag}_{rule}"
                res[k][0] += 1
                res[k][1] += int(x > thr)
                by_year[k].setdefault(yr0, []).append(int(x > thr))
    out = dict(site=site, window_d=w, candidate_set=("excl_3W_after_onsets" if clean else "all_2W_rule"),
               n_candidate_windows=int(len(C["c"])))
    for tag in ("pro", "retro"):
        out[f"{tag}_max_nominal_rate_exchangeable"] = (
            round(float(np.mean(nominal[tag])), 4) if nominal[tag] else np.nan)
        out[f"{tag}_median_pool_years"] = (
            float(np.median(1.0 / np.asarray(nominal[tag]) - 1)) if nominal[tag] else np.nan)
    for k, (n, a) in res.items():
        out[f"{k}_n_evaluated"] = n
        out[f"{k}_alert_rate"] = round(a / n, 4) if n else np.nan
        yrs = by_year[k]
        out[f"{k}_n_years"] = len(yrs)
        out[f"{k}_mean_annual_alert_fraction"] = (
            round(float(np.mean([np.mean(v) for v in yrs.values()])), 4) if yrs else np.nan)
    return out


# ---------------------------------------------------------------------------
#  inventory / duplicates
# ---------------------------------------------------------------------------
def dup_inventory(site, df):
    g = df.groupby("date")
    sizes = g.size()
    dupdates = sizes[sizes > 1]
    q = df.groupby("date")["_quiet"].max()
    kinds = []
    for dte in dupdates.index:
        s = df[df["date"] == dte]
        sid = s["scene_id"].astype(str).tolist()
        pr = [x.split("_")[2] if x.count("_") >= 3 else "?" for x in sid]
        paths = {p[:3] for p in pr}
        sc = set(s["spacecraft"].astype(str))
        kinds.append("same path, adjacent rows" if len(paths) == 1 and len(sc) == 1
                     else ("different spacecraft" if len(sc) > 1 else "different paths"))
    k = pd.Series(kinds).value_counts().to_dict() if kinds else {}
    y = pd.to_numeric(df[EST], errors="coerce")
    diffs = df.assign(y=y).groupby("date")["y"].agg(lambda s: s.max() - s.min())[dupdates.index]
    return dict(site=site, n_scenes=int(len(df)), n_unique_dates=int(sizes.size),
                n_dates_with_duplicates=int(len(dupdates)),
                n_excess_scenes=int((dupdates - 1).sum()),
                max_scenes_per_date=int(sizes.max()),
                duplicate_kind=json.dumps(k),
                n_quiet_fit_scenes=int(df["_quiet"].sum()),
                n_quiet_fit_unique_dates=int(q.sum()),
                median_abs_sameday_range_K=round(float(diffs.median()), 2) if len(diffs) else np.nan,
                first_valid_obs=str(df.loc[y.notna(), "date"].min().date()),
                last_valid_obs=str(df.loc[y.notna(), "date"].max().date()),
                n_valid=int(y.notna().sum()))


# ---------------------------------------------------------------------------
def main():
    ev_all, loo_all, dup_rows, pow_rows, fa_rows = [], [], [], [], []
    for site in SITES:
        for mode in ("asis", "collapse", "refit"):
            df, ke = load_series(site, mode)
            if mode == "asis":
                dup_rows.append(dup_inventory(site, df))
                for w in POST_W:
                    fa_rows.append(false_alarm(site, df, ke, w))
                    fa_rows.append(false_alarm(site, df, ke, w, clean=True))
            ev, loo, store = evaluate(site, df, ke, mode, extra_pools=(mode == "asis"))
            ev_all.append(ev)
            if len(loo):
                loo_all.append(loo)
            if mode == "asis":
                for (s_, kind, w, onset, _), (yp, g, d) in store.items():
                    pr = power_row(yp["v"], obs=float(d))
                    if pr:
                        pow_rows.append(dict(site=s_, analysis=kind, window_d=w, onset=onset,
                                             delta=round(float(d), 2),
                                             placebo_n_years=len(yp["v"]),
                                             min_attainable_p=round(1 / (len(yp["v"]) + 1), 4),
                                             **pr))
            print(site, mode, "done", flush=True)
    ev = pd.concat(ev_all, ignore_index=True)
    ev.to_csv(OUT / "event_windows_all_variants.csv", index=False)
    pd.concat(loo_all, ignore_index=True).to_csv(OUT / "leave_one_out_detail.csv", index=False)
    dup = pd.DataFrame(dup_rows)
    dup.to_csv(OUT / "duplicate_inventory.csv", index=False)
    pd.DataFrame(pow_rows).to_csv(OUT / "power_injection.csv", index=False)
    pd.DataFrame(fa_rows).to_csv(OUT / "false_alarm_prospective.csv", index=False)

    # check that the re-implementation reproduces the primary s10 records
    rp = pd.read_csv(REC_POST); rp = rp[rp.estimator.eq(EST) & rp.window_d.isin(POST_W)]
    rq = pd.read_csv(REC_PRE); rq = rq[rq.estimator.eq(EST) & rq.window_d.isin(PRE_W)
                                      & rq.site.isin(SITES)]
    a = ev[ev.variant.eq("asis")]
    chk = []
    for r in rp.itertuples():
        m = a[(a.analysis == "post") & (a.site == r.site) & (a.window_d == r.window_d)
              & (a.onset == r.onset)]
        x = m.iloc[0] if len(m) else None
        chk.append(dict(analysis="post", site=r.site, window_d=r.window_d, onset=r.onset,
                        rec_delta=r.delta, new_delta=x["delta"] if x is not None else np.nan,
                        rec_p=r.p_placebo_year, new_p=x.get("p_rank") if x is not None else np.nan,
                        rec_n_years=r.placebo_n_years,
                        new_n_years=x.get("placebo_n_years") if x is not None else np.nan,
                        rec_q025=r.placebo_q025, new_q025=x.get("q025") if x is not None else np.nan,
                        rec_npre=r.n_pre, new_npre=x["n_pre"] if x is not None else np.nan,
                        rec_npost=r.n_post, new_npost=x["n_post"] if x is not None else np.nan))
    for r in rq.itertuples():
        m = a[(a.analysis == "pre") & (a.site == r.site) & (a.window_d == r.window_d)
              & (a.onset == r.onset)]
        x = m.iloc[0] if len(m) else None
        chk.append(dict(analysis="pre", site=r.site, window_d=r.window_d, onset=r.onset,
                        rec_delta=r.delta_pre, new_delta=x["delta"] if x is not None else np.nan,
                        rec_p=r.p_placebo_year, new_p=x.get("p_rank") if x is not None else np.nan,
                        rec_n_years=r.placebo_n_years,
                        new_n_years=x.get("placebo_n_years") if x is not None else np.nan,
                        rec_q025=r.placebo_q025, new_q025=x.get("q025") if x is not None else np.nan,
                        rec_npre=r.n_early, new_npre=x["n_pre"] if x is not None else np.nan,
                        rec_npost=r.n_late, new_npost=x["n_post"] if x is not None else np.nan))
    pd.DataFrame(chk).to_csv(OUT / "reproduction_check.csv", index=False)

    # observation structure of every reported window
    sup_cols = ["site", "analysis", "window_d", "onset", "anchor_date", "n_merged",
                "n_pre", "n_post", "n_pre_unique_dates", "n_post_unique_dates",
                "last_obs_before", "first_obs_after", "days_last_obs_to_anchor",
                "days_anchor_to_first_obs", "max_gap_pre_incl_edges_d",
                "max_gap_pre_between_obs_d", "max_gap_post_incl_edges_d",
                "max_gap_post_between_obs_d", "tested", "delta", "placebo_n_years",
                "p_rank", "min_attainable_p", "q025", "q975", "max_exceedance",
                "pre_dates", "post_dates"]
    sup = a[sup_cols].copy()
    sup.to_csv(OUT / "supp_table_window_observations.csv", index=False)

    # pre-onset window data for all window lengths
    f9 = a[a.analysis.eq("pre")][["site", "window_d", "onset", "n_pre", "n_post",
                                  "n_pre_unique_dates", "n_post_unique_dates",
                                  "tested", "delta", "placebo_n_years", "p_rank",
                                  "q025", "q975", "max_exceedance",
                                  "excl3w_allonsets_p_rank",
                                  "excl3w_allonsets_placebo_n_years",
                                  "excl3w_allonsets_max_exceedance"]]
    f9 = f9.rename(columns={"n_pre": "n_early_window", "n_post": "n_late_window"})
    f9.to_csv(OUT / "fig9b_pre_onset_all_windows.csv", index=False)

    # re-analysis summary (collapse / refit / joint matching / 3W exclusion)
    keys = ["site", "analysis", "window_d", "onset"]
    base = a[a.tested][keys + ["delta", "placebo_n_years", "p_rank", "max_exceedance",
                                "p_rank_uncentred", "max_exceedance_uncentred",
                                "q025", "q975"]]
    for v in ("collapse", "refit"):
        x = ev[ev.variant.eq(v)][keys + ["n_pre", "n_post", "tested", "delta",
                                          "placebo_n_years", "p_rank", "q025", "q975",
                                          "max_exceedance"]]
        x = x.rename(columns={c: f"{v}_{c}" for c in x.columns if c not in keys})
        base = base.merge(x, on=keys, how="left")
    extra = a[a.tested][keys + [c for c in a.columns if c.startswith(("exact_both", "tol1_both",
                                                                        "tol2_both", "excl3w"))]
                        + ["loo_delta_min", "loo_delta_max", "loo_n_keep_max_exceedance",
                           "loo_n_untestable", "loo_n_variants", "lodo_delta_min",
                           "lodo_delta_max", "lodo_n_keep_max_exceedance",
                           "lodo_n_untestable", "lodo_n_variants",
                           "n_pool_years_overlapping_actual_postonset",
                           "baseline_fell_back_to_season_only"]]
    base.merge(extra, on=keys, how="left").to_csv(OUT / "reanalysis_summary.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
