# -*- coding: utf-8 -*-
"""Is Taal Lake (the caldera lake) an acceptable reference for the
Main Crater Lake around the 12 January 2020 eruption, given the NE-SW dike
emplaced beneath the SW part of the caldera (Bato et al., 2021)?

Uses cached clear-sky water ST of open-water circles in all sectors of Taal
Lake, one row per Landsat 5-9 C2 L2 scene (point_scene_st_raw.csv, same
QA_PIXEL clear mask and same-scene water mask as the pipeline), and tests
whether the 2020 before-after change of the caldera lake is spatially organised
(SW/dike sector, distance from the Main Crater) or scene-wide. Also refits the
primary relative-temperature model with the caldera-lake donor replaced by each
alternative point.

Pipeline modules and existing records are not modified. Inputs and outputs:
    results/11_analysis/taal_reference_spatial/

    python -m pipeline.analysis.taal_reference_spatial --stage all
    (stages: analyse, figure)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pipeline.config import settings as S

OUT = S.RECORDS_DIR / "11_analysis" / "taal_reference_spatial"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_SCENES = OUT / "point_scene_st_raw.csv"
CACHE_POINTS = OUT / "points.csv"
CACHE_MAP = OUT / "_lake_water_grid.npz"

MC_LAT, MC_LON = 14.01035, 120.99826          # Main Crater Lake centre
PLACEBO_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023]
CLIM = ("2013-01-01", "2019-12-31")
AN_START, AN_END = "2013-01-01", "2024-01-31"
CLEAR_MIN = 0.80                               # clear pixels / observed pixels in the circle (pipeline convention; SLC-off gaps are not counted)
MIN_WATER_PX = 100
ST_RANGE = (270.0, 330.0)                     # absurd-value guard only (pipeline keeps all QA-clear values)
MIN_OBS_FRAC = 0.50                           # circle must be at least half inside observed (non-fill) pixels
MIN_SCENES = 2

# ---------------------------------------------------------------------------
#  sampling circles. Selected from a 0.005-deg grid of JRC-GSW distance to the
#  nearest non-water pixel (occurrence >= 50 %); criterion: centre distance to
#  shore/island >= r + 1000 m. W sector has no location meeting it (the channel
#  between Volcano Island and the W shore is < 2 km wide) -> relaxed point.
# ---------------------------------------------------------------------------
POINTS = [
    # id, lat, lon, r_m, sector, role
    ("N",      14.0600, 120.9900, 1000, "N",  "open water"),
    ("REF",    None,    None,     1500, "NE", "current caldera-lake reference (published AOI polygon)"),
    ("REF_reg", 14.0450, 121.0300, 1500, "NE", "registered caldera-lake coordinate (r = 1500 m circle)"),
    ("E",      13.9900, 121.0500, 1000, "E",  "open water"),
    ("SE",     13.9700, 121.0400, 1000, "SE", "open water"),
    ("S",      13.9450, 121.0000, 1000, "S",  "open water"),
    ("S_far",  13.9150, 121.0070,  750, "S",  "open water (far)"),
    ("SW_a",   13.9480, 120.9700,  750, "SW", "open water, over inferred 2020 dike area"),
    ("SW_b",   13.9630, 120.9660,  750, "SW", "open water, over inferred 2020 dike area"),
    ("W",      14.0000, 120.9600,  500, "W",  "narrow W channel (shore criterion relaxed)"),
    ("NW",     14.0600, 120.9750,  750, "NW", "open water"),
    ("c01",    13.99502, 121.01314, 1500, "SE", "current donor c01 (r = 1500 m, includes island land; water pixels only)"),
]


# ---------------------------------------------------------------------------
#  analysis
# ---------------------------------------------------------------------------
CORE = ["N", "REF", "E", "SE", "S", "S_far", "SW_a", "SW_b", "NW"]   # meet the shore criterion
SW = ["SW_a", "SW_b"]
ALLP = [p[0] for p in POINTS]
ONSETS = {y: pd.Timestamp(f"{y}-01-12") for y in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2022, 2023]}
LEMERY = (13.883, 120.915)          # SW end of the schematic dike axis (town of Lemery)


def dist_to_dike_axis(lat, lon):
    """Distance (km) to the segment Main Crater -> Lemery, a schematic NE-SW axis only."""
    def xy(la, lo):
        return ((lo - MC_LON) * 111.32 * np.cos(np.radians(MC_LAT)), (la - MC_LAT) * 110.57)
    p, a, b = np.array(xy(lat, lon)), np.array(xy(MC_LAT, MC_LON)), np.array(xy(*LEMERY))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def load_scenes():
    d = pd.read_csv(CACHE_SCENES)
    d["t"] = pd.to_datetime(d["t"])
    d["date"] = d["t"].dt.normalize()
    d["obs_frac"] = d["obs_sum"] / d["tot_sum"]
    d["clear_frac"] = d["clr_sum"] / d["obs_sum"].where(d["obs_sum"] > 0)
    d["clear_frac_of_circle"] = d["clr_sum"] / d["tot_sum"]
    d["sensor"] = d["scene_id"].str[:4]
    d["pathrow"] = d["scene_id"].str[10:16]
    return d


def point_series(d, clear_min=CLEAR_MIN):
    """date x point matrix of clear-sky median water ST (K), plus land/ERA5 columns."""
    w = d[d["point"].isin(ALLP)].copy()
    ok = ((w["clear_frac"] >= clear_min) & (w["obs_frac"] >= MIN_OBS_FRAC) & (w["stw_count"] >= MIN_WATER_PX)
          & w["stw_median"].between(*ST_RANGE))
    w = w[ok]
    M = w.pivot_table(index="date", columns="point", values="stw_median", aggfunc="median")
    L = d[d["point"].eq("LAND_regional")]
    L = L[(L["clear_frac"] >= 0.5) & (L["stl_count"] >= 20000)]
    land = L.groupby("date")["stl_median"].median()
    era = d.groupby("date")[["era5_t2m", "era5_skin"]].median()
    return M, land, era, int((~ok).sum())


def monthly_anom(s, start=CLIM[0], end=CLIM[1]):
    s = s.dropna()
    q = s[(s.index >= start) & (s.index <= end)]
    clim = q.groupby(q.index.month).mean()
    return s - clim.reindex(s.index.month).to_numpy()


def harmonic_anom(s, start=CLIM[0], end=CLIM[1]):
    s = s.dropna()
    doy = s.index.dayofyear.to_numpy(float)
    H = np.c_[np.ones(len(s)), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
              np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]
    m = ((s.index >= start) & (s.index <= end))
    c, *_ = np.linalg.lstsq(H[m], s.to_numpy()[m], rcond=None)
    return s - H @ c


def before_after(s, t0, w):
    s = s.dropna()
    pre = s[(s.index >= t0 - pd.Timedelta(days=w)) & (s.index < t0)]
    post = s[(s.index >= t0) & (s.index <= t0 + pd.Timedelta(days=w))]
    ok = len(pre) >= MIN_SCENES and len(post) >= MIN_SCENES
    return (float(post.median() - pre.median()) if ok else np.nan), len(pre), len(post)


def year_deltas(s, w):
    return {y: before_after(s, t0, w) for y, t0 in ONSETS.items()}


def rank_row(dd):
    """dd: {year: (delta, npre, npost)} -> summary of 2020 against placebo years."""
    d20 = dd[2020][0]
    pl = np.array([dd[y][0] for y in dd if y != 2020], float)
    pl = pl[np.isfinite(pl)]
    out = dict(delta_2020=d20, n_pre_2020=dd[2020][1], n_post_2020=dd[2020][2], n_placebo=len(pl),
               placebo_min=np.nan, placebo_max=np.nan, placebo_median=np.nan,
               rank_2020_signed_low=np.nan, rank_2020_abs_dev=np.nan, outside_placebo_range=False)
    if np.isfinite(d20) and len(pl):
        allv = np.r_[pl, d20]
        c = np.median(pl)
        out.update(placebo_min=pl.min(), placebo_max=pl.max(), placebo_median=c,
                   rank_2020_signed_low=int((allv <= d20).sum()),          # 1 = most negative
                   rank_2020_abs_dev=int((np.abs(allv - c) >= abs(d20 - c)).sum()),  # 1 = most extreme
                   outside_placebo_range=bool(d20 < pl.min() or d20 > pl.max()))
    for y in dd:
        out[f"d_{y}"] = dd[y][0]
    return out


def spatial_stats(deltas, meta, pts):
    """deltas: {point: delta}. SW-minus-others, slope vs distance, between-point SD."""
    v = np.array([deltas.get(p, np.nan) for p in pts], float)
    dist = np.array([meta.loc[p, "dist_from_main_crater_km"] for p in pts], float)
    ok = np.isfinite(v)
    sw = np.array([p in SW for p in pts])
    r = dict(n_points=int(ok.sum()), mean=np.nan, sd_between=np.nan, sw_minus_others=np.nan,
             slope_K_per_km=np.nan)
    if ok.sum() >= 4:
        r["mean"] = float(np.mean(v[ok]))
        r["sd_between"] = float(np.std(v[ok], ddof=1))
        if (ok & sw).any() and (ok & ~sw).any():
            r["sw_minus_others"] = float(np.mean(v[ok & sw]) - np.mean(v[ok & ~sw]))
        r["slope_K_per_km"] = float(np.polyfit(dist[ok], v[ok], 1)[0])
    return r


def stage_analyse():
    from pipeline.analysis import reference_model as RM
    meta = pd.read_csv(CACHE_POINTS).set_index("point")
    meta["dist_to_schematic_dike_axis_km"] = [round(dist_to_dike_axis(r.lat, r.lon), 2) for r in meta.itertuples()]
    meta.to_csv(OUT / "points.csv", encoding="utf-8-sig")
    d = load_scenes()
    M_all, land, era, n_rej = point_series(d)
    win = (M_all.index >= AN_START) & (M_all.index <= AN_END)
    M = M_all[win]
    notes = {"n_rows_raw": int(len(d)), "n_point_scene_rejected": n_rej,
             "pathrows": d["pathrow"].value_counts().to_dict(),
             "n_dates_2013_2023_any_point": int(len(M)),
             "valid_scenes_per_point_2013_2023": M.notna().sum().to_dict()}

    # ---- 3. per-point anomalies and before-after -------------------------------
    A = pd.DataFrame({p: monthly_anom(M[p]) for p in M.columns})
    AH = pd.DataFrame({p: harmonic_anom(M[p]) for p in M.columns})
    rows = []
    for p in ALLP:
        for w in (365, 180):
            for tag, X in (("monthly_clim", A), ("harmonic_clim", AH)):
                rr = rank_row(year_deltas(X[p], w))
                rows.append(dict(point=p, window_d=w, anomaly=tag, **meta.loc[p, ["sector", "r_m",
                                 "dist_from_main_crater_km", "bearing_deg", "dist_to_schematic_dike_axis_km",
                                 "meets_shore_criterion"]].to_dict(), n_scenes_2013_2023=int(M[p].notna().sum()),
                                 quiet_sd_K=float(X[p][(X.index <= CLIM[1])].std()), **rr))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_point_before_after.csv", index=False, encoding="utf-8-sig")

    # context series (same statistic)
    Pp = RM.prep("taal")
    dfp = RM.fit(Pp)
    pub = dfp.set_index(pd.to_datetime(dfp["date"]))
    ctx = {"published_caldera_lake_T_control": pub["T_control_K"],
           "published_island_ring": pub["bg_active_K"],
           "published_main_crater_T_active": pub["T_active_K"],
           "regional_land_ST": land, "ERA5L_t2m": era["era5_t2m"], "ERA5L_skin": era["era5_skin"]}
    crow = []
    for k, s in ctx.items():
        s = s[(s.index >= AN_START) & (s.index <= AN_END)].groupby(level=0).median()
        for w in (365, 180):
            crow.append(dict(series=k, window_d=w, n_scenes=int(s.notna().sum()),
                             **rank_row(year_deltas(monthly_anom(s), w))))
    # ERA5 on the clear-lake dates only (same sampling as the points)
    core_dates = M[CORE].notna().sum(axis=1) >= 5
    for k in ("era5_t2m", "era5_skin"):
        s = era[k].reindex(M.index[core_dates])
        for w in (365,):
            crow.append(dict(series=f"ERA5L_{k[5:]}_on_lake_clear_dates", window_d=w, n_scenes=int(s.notna().sum()),
                             **rank_row(year_deltas(monthly_anom(s), w))))

    # ---- 4. spatial signal ------------------------------------------------------
    common = A[CORE].where(A[CORE].notna().sum(axis=1) >= 5).median(axis=1)
    common = common.dropna()
    DEV = A.loc[common.index].sub(common, axis=0)
    # points differ in how strongly they follow scene-wide anomalies (gain). Fit each
    # point's gain on 2013-2018 and remove gain x common mode as well.
    gains = {}
    GDEV = pd.DataFrame(index=common.index)
    for p in A.columns:
        j = pd.concat([A[p], common], axis=1, keys=["a", "c"]).dropna()
        jq = j[j.index <= "2018-12-31"]          # fitted outside the 2020 test windows
        g = float(np.polyfit(jq["c"], jq["a"], 1)[0]) if len(jq) >= 10 else np.nan
        gains[p] = g
        GDEV[p] = A[p].reindex(common.index) - g * common
    notes["gain_on_common_mode_2013_2018"] = gains
    for w in (365, 180):
        crow.append(dict(series="lake_common_mode(median of core points, >=5 valid)", window_d=w,
                         n_scenes=int(common.notna().sum()), **rank_row(year_deltas(common, w))))
    pd.DataFrame(crow).to_csv(OUT / "context_series_before_after.csv", index=False, encoding="utf-8-sig")

    srows = []
    for w in (365, 180):
        for tag, X in (("own_scenes_anomaly", A), ("deviation_from_common_mode", DEV),
                       ("gain_adjusted_deviation", GDEV)):
            for y in ONSETS:
                dl = {p: before_after(X[p], ONSETS[y], w)[0] for p in CORE}
                srows.append(dict(window_d=w, quantity=tag, onset_year=y, **spatial_stats(dl, meta, CORE),
                                  common_mode_delta=before_after(common, ONSETS[y], w)[0]))
    sp = pd.DataFrame(srows)
    sp.to_csv(OUT / "spatial_stats_by_year.csv", index=False, encoding="utf-8-sig")
    srank = []
    for (w, q), g in sp.groupby(["window_d", "quantity"]):
        for stat in ("sw_minus_others", "slope_K_per_km", "sd_between", "common_mode_delta"):
            dd = {int(r.onset_year): (getattr(r, stat), 2, 2) for r in g.itertuples()}
            rr = rank_row(dd)
            srank.append(dict(window_d=w, quantity=q, statistic=stat,
                              **{k: rr[k] for k in ("delta_2020", "n_placebo", "placebo_min", "placebo_max",
                                                     "placebo_median", "rank_2020_abs_dev", "outside_placebo_range")}))
    pd.DataFrame(srank).to_csv(OUT / "spatial_tests_2020_vs_placebo.csv", index=False, encoding="utf-8-sig")

    # scene-wide check: late-2019 and 2020 scenes
    ld = monthly_anom(land[(land.index >= AN_START) & (land.index <= AN_END)])
    ea = monthly_anom(era["era5_t2m"][(era.index >= AN_START) & (era.index <= AN_END)])
    ring = monthly_anom(pub["bg_active_K"][(pub.index >= AN_START) & (pub.index <= AN_END)].groupby(level=0).median())
    tl = pd.DataFrame({"lake_common_anom": common, "SW_mean_anom": A[SW].mean(axis=1),
                       "nonSW_core_median_anom": A[[p for p in CORE if p not in SW]].median(axis=1),
                       "REF_anom": A["REF"], "regional_land_anom": ld, "island_ring_anom(published)": ring,
                       "ERA5L_t2m_anom": ea, "n_core_valid": M[CORE].notna().sum(axis=1)})
    tl = tl[(tl.index >= "2018-07-01") & (tl.index <= "2021-06-30")].dropna(subset=["lake_common_anom"])
    tl.to_csv(OUT / "scene_wide_check_2018_2021.csv", encoding="utf-8-sig", index_label="date")
    J = pd.DataFrame({"lake": common, "land": ld, "era": ea, "ring": ring}).dropna(subset=["lake"])
    cors = {f"corr_lake_common_vs_{k}": float(J["lake"].corr(J[k])) for k in ("land", "era", "ring")}
    cors.update({f"n_{k}": int(J[["lake", k]].dropna().shape[0]) for k in ("land", "era", "ring")})
    notes["correlations_2013_2023"] = cors
    notes["REF_vs_published_T_control"] = dict(
        corr=float(pd.concat([M["REF"], pub["T_control_K"].groupby(level=0).median()], axis=1).dropna().corr().iloc[0, 1]),
        median_diff_K=float((M["REF"] - pub["T_control_K"].groupby(level=0).median()).dropna().median()),
        n=int((M["REF"] - pub["T_control_K"].groupby(level=0).median()).notna().sum()))

    # ---- 5a. proxy: Main Crater lake minus point --------------------------------
    TA = pub["T_active_K"].groupby(level=0).median()
    TA = TA[(TA.index >= AN_START) & (TA.index <= AN_END)]
    prow = []
    series5 = {p: M[p] for p in ALLP}
    series5["published_T_control"] = pub["T_control_K"].groupby(level=0).median()
    for p, s in series5.items():
        c = (TA - s.reindex(TA.index)).dropna()
        for w in (365, 180):
            for tag, X in (("raw", c), ("monthly_anom", monthly_anom(c))):
                prow.append(dict(point=p, window_d=w, contrast="MainCrater_minus_point", series=tag,
                                 n_scenes=int(len(c)), **rank_row(year_deltas(X, w))))
    pd.DataFrame(prow).to_csv(OUT / "proxy_contrast_lake_minus_point.csv", index=False, encoding="utf-8-sig")

    # ---- 5b. full refit with the caldera-lake donor replaced ---------------------
    base = Pp["d"].copy()
    base["day"] = pd.to_datetime(base["t"], utc=True).dt.tz_localize(None).dt.normalize()
    M_full = M_all
    frow = []
    df0, det0 = RM.fit(Pp, return_detail=True)
    variants = [("published", None, ())] + [(p, p, ()) for p in ALLP] + \
               [(p + "+drop_c01", p, ("nearby water body c01",)) for p in ("N", "E", "S", "SW_a", "NW")]
    fill_info = {}
    for tag, p, drop in variants:
        P2 = dict(Pp)
        if p is not None:
            s = M_full[p]
            v = s.reindex(base["day"].to_numpy()).to_numpy(float)
            # gaps -> the point's own harmonic climatology (fitted on its 1990-2019 quiet dates)
            ss = s.dropna()
            doy = ss.index.dayofyear.to_numpy(float)
            H = np.c_[np.ones(len(ss)), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                      np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]
            mq = ss.index < "2019-07-01"
            cc, *_ = np.linalg.lstsq(H[mq], ss.to_numpy()[mq], rcond=None)
            db = base["day"].dt.dayofyear.to_numpy(float)
            Hb = np.c_[np.ones(len(db)), np.sin(2 * np.pi * db / 365.25), np.cos(2 * np.pi * db / 365.25),
                       np.sin(4 * np.pi * db / 365.25), np.cos(4 * np.pi * db / 365.25)]
            miss = ~np.isfinite(v)
            win20 = ((base["day"] >= "2019-01-12") & (base["day"] <= "2021-01-12")).to_numpy()
            fill_info[p] = dict(n_rows=int(len(v)), n_filled=int(miss.sum()),
                                n_rows_2019_2020=int(win20.sum()), n_filled_2019_2020=int((miss & win20).sum()))
            v = np.where(miss, Hb @ cc, v)
            P2["ctl"] = v
        dfx, detx = RM.fit(P2, drop_donors=drop, return_detail=True)
        wts = {RM.donor_label("taal", "paired", k)[1]: round(float(x), 3) for k, x in detx["weights"].items()}
        for w in (365, 180):
            for r in RM.contrasts(dfx["dnum"].to_numpy(), dfx["dT_syn_weather"].to_numpy(float), Pp["ke"], w):
                frow.append(dict(variant=tag, replacement_point=p, dropped=";".join(drop),
                                 w_caldera_donor=wts.get("caldera_lake", 0.0), weights=json.dumps(wts),
                                 quiet_sd_K=float(np.std(dfx["dT_syn_weather"][dfx["_fit_w"]], ddof=1)),
                                 **(fill_info.get(p, {}) if p else {}), **r))
    pd.DataFrame(frow).to_csv(OUT / "refit_alternative_caldera_donor.csv", index=False, encoding="utf-8-sig")
    (OUT / "notes.json").write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    print(json.dumps(notes, indent=1, default=str))


# ---------------------------------------------------------------------------
#  figure
# ---------------------------------------------------------------------------
def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    meta = pd.read_csv(OUT / "points.csv").set_index("point")
    per = pd.read_csv(OUT / "per_point_before_after.csv")
    per = per[(per.window_d == 365) & (per.anomaly == "monthly_clim")].set_index("point")
    g = np.load(CACHE_MAP)
    wgrid, ext = g["w"], g["extent"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 4.3), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    ax.imshow(np.where(wgrid > 0, 1.0, np.nan), extent=ext, origin="upper", cmap="Blues", vmin=0, vmax=3,
              interpolation="nearest")
    ax.contour(np.linspace(ext[0], ext[1], wgrid.shape[1]), np.linspace(ext[3], ext[2], wgrid.shape[0]),
               wgrid, levels=[0.5], colors="0.45", linewidths=0.5)
    vals = per["delta_2020"]
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1
    norm = TwoSlopeNorm(0, -vmax, vmax)
    cmap = plt.get_cmap("RdBu_r")
    kx = np.cos(np.radians(MC_LAT))
    for p, r in meta.iterrows():
        dv = per.loc[p, "delta_2020"] if p in per.index else np.nan
        col = cmap(norm(dv)) if np.isfinite(dv) else "0.8"
        rr = r.r_m / 111320.0
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((r.lon, r.lat), 2 * rr / kx, 2 * rr, fc=col, ec="k", lw=0.6,
                             ls="-" if r.meets_shore_criterion else "--", zorder=3))
        ax.text(r.lon, r.lat + rr + 0.0015, p.replace("_", " "), ha="center", va="bottom", fontsize=6.5, zorder=4,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))
    ax.plot(MC_LON, MC_LAT, marker="*", ms=10, mfc="k", mec="w", zorder=5)
    ax.annotate("Main Crater\nLake", (MC_LON, MC_LAT), xytext=(MC_LON - 0.012, MC_LAT + 0.012), fontsize=6.5,
                ha="right", va="bottom", zorder=6, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"),
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.3))
    ax.plot([MC_LON, LEMERY[1]], [MC_LAT, LEMERY[0]], ls=(0, (4, 2)), c="0.2", lw=0.9, zorder=2)
    ax.annotate("schematic NE-SW\ndike trend", (LEMERY[1] + 0.005, LEMERY[0] + 0.012), fontsize=6, color="0.2")
    ax.text(121.097, 13.863, "dashed outline: < 1 km from shore", ha="right", va="bottom", fontsize=5.5, color="0.3")
    ax.set_xlim(120.90, 121.10); ax.set_ylim(13.86, 14.11)
    ax.set_aspect(1 / kx)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title("(a) 2020 change in open-water ST anomaly", fontsize=8, loc="left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("365-day after minus before (K)")

    order = meta.sort_values("bearing_deg").index.tolist()
    y = np.arange(len(order))
    for i, p in enumerate(order):
        r = per.loc[p]
        pl = [r[f"d_{yy}"] for yy in PLACEBO_YEARS if np.isfinite(r[f"d_{yy}"])]
        if pl:
            bx.plot([min(pl), max(pl)], [i, i], c="0.75", lw=3, solid_capstyle="round", zorder=1)
            bx.scatter(pl, [i] * len(pl), s=9, c="0.45", zorder=2, lw=0)
        c = "#b2182b" if p in SW else "k"
        bx.scatter([r.delta_2020], [i], s=34, marker="D", c=c, zorder=3, edgecolor="w", lw=0.6)
    bx.axvline(0, c="0.6", lw=0.6)
    lab = [f"{p.replace('_', ' ')}  ({meta.loc[p, 'bearing_deg']:.0f}°, {meta.loc[p, 'dist_from_main_crater_km']:.1f} km)"
           for p in order]
    bx.set_yticks(y); bx.set_yticklabels(lab, fontsize=6.5)
    bx.invert_yaxis()
    bx.set_xlabel("After minus before, median anomaly (K)")
    bx.set_title("(b) 2020 onset vs placebo years", fontsize=8, loc="left")
    from matplotlib.lines import Line2D
    bx.legend(handles=[Line2D([], [], marker="D", ls="", c="k", label="2020 (12 Jan onset)"),
                       Line2D([], [], marker="D", ls="", c="#b2182b", label="2020, SW (dike) sector"),
                       Line2D([], [], marker="o", ls="", c="0.45", ms=3, label="placebo years (12 Jan)"),
                       Line2D([], [], c="0.75", lw=3, label="placebo range")],
              fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "taal_reference_spatial.png", dpi=300)
    print("figure saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["analyse", "figure", "all"])
    a = ap.parse_args()
    if a.stage in ("analyse", "all"):
        stage_analyse()
    if a.stage in ("figure", "all"):
        stage_figure()
