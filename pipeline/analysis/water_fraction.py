# -*- coding: utf-8 -*-
"""Water boundary and water-fraction provenance.

Outputs (results/11_analysis/water_fraction/):
    scene_inventory.csv          one row per Landsat scene x study lake
    site_summary.csv             counts by sensor x fate, first/last retained dates
    site_summary_by_reason.csv   counts by exclusion reason
    f_psf_local.csv              local re-computation of the PSF-weighted fraction (sigma grid)
    boundary_sensitivity.csv     primary event contrasts under alternative boundary / f assumptions
    unmix_reconstruction_check.csv

Pipeline modules and records are read only. Alternative scenarios are run by
pointing settings.PROCESSED_DIR at a scratch copy that carries a modified deltaT table,
then calling the unchanged s10_post_eruption.run_site().

Run:
    python -m pipeline.analysis.water_fraction
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import settings as S
from pipeline.config import thermal_episodes as TE
from pipeline.config.sites import get_site

SITES = ("taal", "santa_ana", "ruapehu")
OUT = S.RECORDS_DIR / "11_analysis" / "water_fraction"
AOI_DIR = S.DATA_DIR / "aoi"
F_MIN, F_REL = 0.70, 0.50
SENSOR_NATIVE_M = {"LT05": 120.0, "LE07": 60.0, "LC08": 100.0, "LC09": 100.0}
PSF_SIGMA_M = 58.0


# ---------------------------------------------------------------------------
# 1. Local re-computation of the PSF-weighted water fraction
# ---------------------------------------------------------------------------
def _aoi(site_key, lake_key):
    p = AOI_DIR / f"{site_key}_{lake_key}.geojson"
    gj = json.loads(p.read_text(encoding="utf-8"))
    return gj["geometry"], gj.get("properties", {})


def local_f(site_key, lake_key, sigmas=(0.0, 34.8, 40.0, 58.0, 69.6, 80.0),
            erode_back=None):
    """Recomputes the pipeline's water polygon and PSF-weighted water fraction locally.

    water polygon = eroded AOI buffered back by erode_m_used; paint 0/1 on a 10 m grid;
    Gaussian kernel (radius 3 sigma, normalised); mean-aggregate to 100 m cells;
    statistics over 100 m cells whose centres fall inside the eroded AOI.
    The 100 m grid phase is unknown (the extraction grid is EPSG:4326 at 100 m), so all 10x10
    phases are evaluated and the median / range reported.
    """
    from shapely.geometry import shape
    from shapely.ops import transform
    from shapely import contains_xy
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter

    site = get_site(site_key)
    geom, props = _aoi(site_key, lake_key)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{site.utm_epsg}", always_xy=True)
    aoi = transform(tr.transform, shape(geom))
    e = float(props.get("erode_m_used") or 0) if erode_back is None else erode_back
    water = aoi.buffer(e) if e > 0 else aoi
    res = 10.0
    margin = 600.0
    x0, y0, x1, y1 = water.bounds
    x0, y0 = np.floor((x0 - margin) / 100) * 100, np.floor((y0 - margin) / 100) * 100
    x1, y1 = np.ceil((x1 + margin) / 100) * 100, np.ceil((y1 + margin) / 100) * 100
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    XX, YY = np.meshgrid(xs, ys)
    base = contains_xy(water, XX, YY).astype(float)
    rows = []
    for sg in sigmas:
        img = base if sg <= 0 else gaussian_filter(base, sigma=sg / res, mode="constant",
                                                    truncate=3.0)
        fmax, fmean = [], []
        for ox in range(10):
            for oy in range(10):
                sub = img[oy:, ox:]
                ny, nx = (sub.shape[0] // 10) * 10, (sub.shape[1] // 10) * 10
                blk = sub[:ny, :nx].reshape(ny // 10, 10, nx // 10, 10).mean(axis=(1, 3))
                cx = x0 + ox * res + (np.arange(nx // 10) + 0.5) * 100
                cy = y0 + oy * res + (np.arange(ny // 10) + 0.5) * 100
                CX, CY = np.meshgrid(cx, cy)
                inside = contains_xy(aoi, CX, CY)
                if not inside.any():
                    continue
                fmax.append(blk[inside].max())
                fmean.append(blk[inside].mean())
        rows.append(dict(site=site_key, lake=lake_key, psf_sigma_m=sg,
                         erode_back_m=e,
                         f_max_median=float(np.median(fmax)),
                         f_max_min=float(np.min(fmax)), f_max_max=float(np.max(fmax)),
                         f_mean_median=float(np.median(fmean)),
                         n_phases=len(fmax),
                         water_polygon_area_m2=float(water.area),
                         sampling_polygon_area_m2=float(aoi.area)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Scene inventory
# ---------------------------------------------------------------------------
def _sensor(scene_id):
    return str(scene_id)[:4]


def _s2_area_interp(site_key, lake_key, dates):
    """Same rule as s2b.build_f_timevarying, but returns the interpolated S2 area."""
    w = pd.read_csv(S.INTERIM_DIR / f"{site_key}_water_extent.csv")
    w = w[(w["lake"] == lake_key) & w["qc_flag"].isin(["ok", "no_water"])]
    w = w.dropna(subset=["f_tir", "area_m2"])
    if len(w) < 2:
        return np.full(len(dates), np.nan)
    sx = pd.to_datetime(w["date"]).to_numpy().astype("datetime64[D]").astype(float)
    sy = w["area_m2"].to_numpy(float)
    o = np.argsort(sx)
    sx, sy = sx[o], sy[o]
    out = np.full(len(dates), np.nan)
    for i, d in enumerate(pd.to_datetime(dates)):
        dn = np.datetime64(d.date(), "D").astype(float)
        if sx[0] <= dn <= sx[-1] and np.min(np.abs(sx - dn)) <= 120:
            out[i] = np.interp(dn, sx, sy)
    return out


def retained_datetimes(site_key, f_min):
    from pipeline import s10_post_eruption as P
    df, _ke, buf = P.load(site_key, f_min)
    return set(df["datetime"].astype(str)), buf


def inventory(site_key):
    site = get_site(site_key)
    lake = site.active.key
    inv = pd.read_csv(S.INTERIM_DIR / f"{site_key}_scene_inventory.csv")
    inv = inv[inv["lake"] == lake].copy()
    lst = pd.read_csv(S.INTERIM_DIR / f"{site_key}_lst.csv")
    lst = lst[lst["lake"] == lake]
    dt = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_deltaT.csv")
    _geom, props = _aoi(site_key, lake)

    keep_l = ["scene_id", "lst_n", "lst_water_n", "water_frac_scene", "f_max", "f_mean",
              "f_geom_max", "f_geom_mean", "plume_flag", "lst_water_median_K", "bg_mean_K"]
    keep_d = ["scene_id", "f_est_act", "f_source_act", "f_est_ctl", "qc_flag",
              "unmix_delta_active_K", "T_active_K", "datetime"]
    x = (inv.merge(lst[keep_l], on="scene_id", how="left")
            .merge(dt[keep_d].rename(columns={"datetime": "dt_datetime"}),
                   on="scene_id", how="left"))
    x["sensor"] = x["scene_id"].map(_sensor)
    x["in_stage2"] = x["lst_n"].notna()
    x["in_stage3"] = x["qc_flag"].notna()
    x["lake_absent_listed"] = TE.absent_mask(site_key, pd.to_datetime(x["date"]))

    ret70, buf70 = retained_datetimes(site_key, F_MIN)
    ret50, _ = retained_datetimes(site_key, F_REL)
    x["retained_f070"] = x["dt_datetime"].astype(str).isin(ret70)
    x["retained_f050"] = x["dt_datetime"].astype(str).isin(ret50)

    # --- boundary provenance -------------------------------------------------
    poly_txt = (f"Sentinel-2 L2A 2019-2025 median-composite water polygon "
                f"(MNDWI>0.10, NDWI>0.05, B11<0.15, B8<0.20), eroded "
                f"{props.get('erode_m_used')} m")
    x["sampling_polygon"] = poly_txt
    x["temperature_water_mask"] = "same-scene Landsat MNDWI>0 or QA_PIXEL water bit, inside sampling polygon"
    src = x["f_source_act"].fillna("")
    x["f_boundary_source"] = np.select(
        [src.eq("s2_interpolated"),
         src.isin(["geometric_fallback", "geometric_constant"])],
        ["S2_scene_area_ratio_interpolated(<=120 d)",
         "static_modern_polygon(no S2 within 120 d)"],
        default="")
    # effective (assigned) lake area
    a_s2 = _s2_area_interp(site_key, lake, x["date"])
    x["assigned_lake_area_m2"] = np.where(
        src.eq("s2_interpolated"), a_s2,
        np.where(src.ne(""), float(props.get("area_raw_km2")) * 1e6, np.nan))
    x["sampled_water_area_m2"] = x["lst_water_n"] * 900.0

    # --- fate / exclusion reason (pipeline order) ----------------------------
    reason = np.full(len(x), "", dtype=object)
    f = pd.to_numeric(x["f_est_act"], errors="coerce")
    conds = [
        (~x["usable"].astype(bool), "clear_fraction_below_0.80"),
        (~x["in_stage2"], "not_extracted"),
        (~x["in_stage3"], ("control_lake_not_clear_same_scene" if site.controls
                           else "not_in_stage3")),
        (x["qc_flag"].ne("ok"), None),          # filled with the qc flag itself
        (f < F_MIN, "water_fraction_below_0.70"),
        (x["lake_absent_listed"], "lake_absent_interval"),
    ]
    for m, lab in conds:
        m = np.asarray(m, bool) & (reason == "")
        if lab is None:
            reason[m] = "qc_" + x.loc[m, "qc_flag"].astype(str)
        else:
            reason[m] = lab
    left = (reason == "") & ~x["retained_f070"].to_numpy()
    reason[left] = "other(fit_or_window_rule)"
    x["exclusion_reason"] = np.where(x["retained_f070"], "", reason)
    x["retained"] = np.where(x["retained_f070"], "y", "n")

    cols = ["site", "lake", "scene_id", "date", "datetime", "sensor", "spacecraft",
            "clear_frac", "sampling_polygon", "temperature_water_mask",
            "f_boundary_source", "assigned_lake_area_m2", "lst_n", "lst_water_n",
            "sampled_water_area_m2", "water_frac_scene", "f_max", "f_mean",
            "f_est_act", "f_est_ctl", "unmix_delta_active_K", "qc_flag",
            "lake_absent_listed", "retained", "retained_f050", "exclusion_reason"]
    x = x[cols].rename(columns={
        "lst_n": "n_cells_30m_sampling_polygon",
        "lst_water_n": "n_water_cells_30m_same_scene",
        "water_frac_scene": "landsat_same_scene_water_fraction",
        "f_max": "f_static_psf_max_cell", "f_mean": "f_static_psf_mean_cell",
        "f_est_act": "f_assigned", "f_est_ctl": "f_assigned_control_lake",
        "unmix_delta_active_K": "unmixing_increment_K",
        "retained_f050": "retained_relaxed_f050"})
    x["retained_relaxed_f050"] = np.where(x["retained_relaxed_f050"], "y", "n")
    return x.sort_values("datetime").reset_index(drop=True)


def summaries(inv):
    rows = []
    for s, g in inv.groupby("site"):
        r = g[g["retained"] == "y"]
        rec = dict(site=s, n_scenes_total=len(g), n_retained=len(r),
                   first_scene=g["date"].min(), last_scene=g["date"].max(),
                   first_retained=r["date"].min(), last_retained=r["date"].max(),
                   n_retained_relaxed_f050=int((g["retained_relaxed_f050"] == "y").sum()))
        for sen in ("LT05", "LE07", "LC08", "LC09"):
            rec[f"n_total_{sen}"] = int((g["sensor"] == sen).sum())
            rec[f"n_retained_{sen}"] = int((r["sensor"] == sen).sum())
        ret_src = r["f_boundary_source"].value_counts()
        for k, v in ret_src.items():
            rec["n_retained_" + k.split("(")[0]] = int(v)
        pre = r[pd.to_datetime(r["date"]) < "2016-01-01"]
        rec["n_retained_before_2016"] = len(pre)
        rec["f_assigned_retained_median"] = round(float(r["f_assigned"].median()), 3)
        rec["f_assigned_retained_min"] = round(float(r["f_assigned"].min()), 3)
        rows.append(rec)
    by = (inv.assign(fate=np.where(inv["retained"] == "y", "retained",
                                   inv["exclusion_reason"]))
             .groupby(["site", "fate", "sensor"]).size().unstack("sensor", fill_value=0)
             .reset_index())
    by["total"] = by[[c for c in ("LT05", "LE07", "LC08", "LC09") if c in by]].sum(axis=1)
    return pd.DataFrame(rows), by


# ---------------------------------------------------------------------------
# 3. Boundary sensitivity
# ---------------------------------------------------------------------------
def _unmix(T_raw, T_bg, f):
    from pipeline.s3_delta_t import unmix
    one = np.ones(len(T_raw))
    T, _s, ok = unmix(T_raw, T_bg, f, one, one)
    return T, ok


def _prepare(site_key):
    """deltaT with reconstructed raw temperatures and per-scene attributes."""
    d = pd.read_csv(S.PROCESSED_DIR / f"{site_key}_deltaT.csv")
    d["T_raw_act"] = d["T_active_K"] - d["unmix_delta_active_K"].fillna(0.0)
    d["T_raw_ctl"] = d["T_raw_act"] - d["dT_raw_K"]
    d["sensor"] = d["scene_id"].map(_sensor)
    site = get_site(site_key)
    lst = pd.read_csv(S.INTERIM_DIR / f"{site_key}_lst.csv")
    a = lst[lst["lake"] == site.active.key][["scene_id", "water_frac_scene", "f_max", "f_mean"]]
    d = d.merge(a.rename(columns={"water_frac_scene": "wfs_act", "f_max": "fmax_act",
                                  "f_mean": "fmean_act"}), on="scene_id", how="left")
    if site.controls:
        c = lst[lst["lake"] == site.controls[0].key][["scene_id", "water_frac_scene",
                                                       "f_max", "f_mean"]]
        d = d.merge(c.rename(columns={"water_frac_scene": "wfs_ctl", "f_max": "fmax_ctl",
                                      "f_mean": "fmean_ctl"}), on="scene_id", how="left")
    return d


def _check_reconstruction(site_key, d):
    T, ok = _unmix(d["T_raw_act"].to_numpy(), d["bg_active_K"].to_numpy(),
                   d["f_est_act"].to_numpy())
    err = np.abs(T - d["T_active_K"].to_numpy())
    out = dict(site=site_key, n=len(d), max_abs_err_active_K=float(np.nanmax(err)))
    if "fmax_ctl" in d and d["f_est_ctl"].notna().any():
        Tc, _ = _unmix(d["T_raw_ctl"].to_numpy(), d["bg_control_K"].to_numpy(),
                       d["f_est_ctl"].to_numpy())
        out["max_abs_err_control_K"] = float(np.nanmax(np.abs(Tc - d["T_control_K"])))
    return out


def scenario_frame(site_key, d, name, fconst_alt=None):
    """Return a modified deltaT for one scenario."""
    x = d.copy()
    has_ctl = x["f_est_ctl"].notna().any()
    fb_a = x["f_est_act"].to_numpy(float)
    fc_a = x["fmax_act"].median()                  # static constant used for fallback
    ratio_a = fb_a / fc_a                           # S2 area ratio (1 for static scenes)
    if has_ctl:
        fb_c = x["f_est_ctl"].to_numpy(float)
        fc_c = x["fmax_ctl"].median()
        ratio_c = fb_c / fc_c

    unmix_on = True
    screen_f = None                                  # f used for the >= F_MIN screen
    if name == "baseline":
        fa = fb_a; fc = fb_c if has_ctl else None
    elif name == "static_modern_polygon_all_years":
        fa = np.full(len(x), fc_a); fc = np.full(len(x), fc_c) if has_ctl else None
    elif name == "same_scene_landsat_fraction":
        fa = fc_a * x["wfs_act"].to_numpy(float)
        fc = fc_c * x["wfs_ctl"].to_numpy(float) if has_ctl else None
    elif name == "cell_mean_aggregation":
        fa = x["fmean_act"].median() * ratio_a
        fc = x["fmean_ctl"].median() * ratio_c if has_ctl else None
    elif name.startswith("psf_"):
        # fconst_alt: dict sensor -> (f_const_active, f_const_control)
        ka = x["sensor"].map(lambda s: fconst_alt[s][0]).to_numpy(float)
        fa = ka * ratio_a
        if has_ctl:
            kc = x["sensor"].map(lambda s: fconst_alt[s][1]).to_numpy(float)
            fc = kc * ratio_c
        else:
            fc = None
    elif name == "no_unmixing_same_scenes":
        unmix_on = False
        fa = fb_a; fc = fb_c if has_ctl else None
    else:
        raise KeyError(name)
    fa = np.clip(fa, 0.01, 1.0)
    if fc is not None:
        fc = np.clip(fc, 0.01, 1.0)

    if unmix_on:
        Ta, _ = _unmix(x["T_raw_act"].to_numpy(), x["bg_active_K"].to_numpy(), fa)
        if has_ctl:
            Tc, _ = _unmix(x["T_raw_ctl"].to_numpy(), x["bg_control_K"].to_numpy(), fc)
        else:
            Tc = x["T_control_K"].to_numpy()
    else:
        Ta = x["T_raw_act"].to_numpy()
        Tc = x["T_raw_ctl"].to_numpy() if has_ctl else x["T_control_K"].to_numpy()
    x["T_active_K"] = Ta
    x["T_control_K"] = Tc
    x["f_est_act"] = fa
    if has_ctl:
        x["f_est_ctl"] = fc
    return x


def run_scenario(site_key, d, name, f_min, fconst_alt=None, tmp=None):
    from pipeline import s10_post_eruption as P
    x = scenario_frame(site_key, d, name, fconst_alt)
    keep = [c for c in pd.read_csv(S.PROCESSED_DIR / f"{site_key}_deltaT.csv", nrows=1).columns]
    x[keep].to_csv(tmp / f"{site_key}_deltaT.csv", index=False)
    old = S.PROCESSED_DIR
    try:
        S.PROCESSED_DIR = tmp
        st, ev, _epo, _meta = P.run_site(site_key, f_min)
        df, _ke, _buf = P.load(site_key, f_min)
    finally:
        S.PROCESSED_DIR = old
    ev = ev[(ev["estimator"] == "dT_syn_weather") & ev["window_d"].isin([180, 365])].copy()
    ev["scenario"] = name
    ev["f_min"] = f_min
    ev["n_retained_scenes"] = 0 if df is None else len(df)
    ev["f_median_retained"] = np.nan
    return ev


def sensitivity(fpsf):
    OUTROWS = []
    tmp = Path(tempfile.mkdtemp(prefix="wf_sens_"))
    checks = []
    try:
        for s in SITES:
            for suf in ("dataset", "anomalies", "energy_balance"):
                p = S.PROCESSED_DIR / f"{s}_{suf}.csv"
                if p.exists():
                    shutil.copy(p, tmp / p.name)
            d = _prepare(s)
            checks.append(_check_reconstruction(s, d))
            site = get_site(s)
            lakes = [site.active.key] + ([site.controls[0].key] if site.controls else [])

            def fc_for(sig_map):
                out = {}
                for sen, sg in sig_map.items():
                    vals = []
                    for lk in lakes:
                        r = fpsf[(fpsf["site"] == s) & (fpsf["lake"] == lk)
                                 & np.isclose(fpsf["psf_sigma_m"], sg)]
                        vals.append(float(r["f_max_median"].iloc[0]))
                    # scale local value to the recorded value at 58 m so that only the
                    # sigma-dependence (not the grid-phase offset) is transferred
                    ref = []
                    for lk in lakes:
                        r0 = fpsf[(fpsf["site"] == s) & (fpsf["lake"] == lk)
                                  & np.isclose(fpsf["psf_sigma_m"], PSF_SIGMA_M)]
                        ref.append(float(r0["f_max_median"].iloc[0]))
                    ee_const = [d["fmax_act"].median()] + (
                        [d["fmax_ctl"].median()] if len(lakes) > 1 else [])
                    sc = [min(1.0, v / r * e) for v, r, e in zip(vals, ref, ee_const)]
                    out[sen] = (sc[0], sc[1] if len(sc) > 1 else np.nan)
                return out

            sens_native = {k: PSF_SIGMA_M * v / 100.0 for k, v in SENSOR_NATIVE_M.items()}
            scen = [("baseline", None), ("static_modern_polygon_all_years", None),
                    ("same_scene_landsat_fraction", None), ("cell_mean_aggregation", None),
                    ("psf_sensor_specific_sigma", fc_for(sens_native)),
                    ("psf_sigma_40m", fc_for({k: 40.0 for k in SENSOR_NATIVE_M})),
                    ("psf_sigma_80m", fc_for({k: 80.0 for k in SENSOR_NATIVE_M})),
                    ("no_unmixing_same_scenes", None)]
            for name, fca in scen:
                for fm in (F_MIN, F_REL):
                    print(f"  {s} {name} f>={fm}")
                    OUTROWS.append(run_scenario(s, d, name, fm, fca, tmp))
                    if fca is not None and fm == F_MIN:
                        OUTROWS[-1]["f_const_by_sensor"] = json.dumps(
                            {k: round(v[0], 3) for k, v in fca.items()})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = pd.concat(OUTROWS, ignore_index=True)
    return res, pd.DataFrame(checks)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. local PSF fractions
    fr = []
    for s in SITES:
        site = get_site(s)
        for lk in [site.active.key] + [c.key for c in site.controls]:
            fr.append(local_f(s, lk))
    fpsf = pd.concat(fr, ignore_index=True)
    fpsf.to_csv(OUT / "f_psf_local.csv", index=False)
    print(fpsf.round(3).to_string())

    # 2. inventory
    inv = pd.concat([inventory(s) for s in SITES], ignore_index=True)
    inv.to_csv(OUT / "scene_inventory.csv", index=False)
    summ, by = summaries(inv)
    summ.to_csv(OUT / "site_summary.csv", index=False)
    by.to_csv(OUT / "site_summary_by_reason.csv", index=False)
    print(summ.T.to_string())
    print(by.to_string())

    # 3. sensitivity
    res, chk = sensitivity(fpsf)
    chk.to_csv(OUT / "unmix_reconstruction_check.csv", index=False)
    keep = ["scenario", "f_min", "site", "window_d", "onset", "n_pre", "n_post",
            "delta", "placebo_n_years", "p_placebo_year", "n_retained_scenes",
            "f_const_by_sensor"]
    res = res[[c for c in keep if c in res.columns]]
    res.to_csv(OUT / "boundary_sensitivity.csv", index=False)
    print(chk.to_string())
    print(res[res["f_min"] == F_MIN].to_string())


if __name__ == "__main__":
    main()
