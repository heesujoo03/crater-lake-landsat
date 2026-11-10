# -*- coding: utf-8 -*-
"""Water-fraction uncertainty and per-scene error sources for unmixing.

Linear mixing in radiance space,

    L_pix = f·L_lake + (1−f)·L_bg   →   L_lake = L_bg + (L_pix − L_bg)/f
    ∂L_lake/∂f = −(L_pix − L_bg)/f²

makes the recovered lake temperature very sensitive to the water fraction f
when f is small and the lake–background contrast is large, so the uncertainty
of f propagates directly into the recovered temperature.

Contents
--------
psf_fraction         PSF-weighted water fraction of a circular lake, emulated
                     in numpy so that PSF width and grid offset can be varied.
f_uncertainty_table  σ_f per site from perturbing PSF width and geolocation.
scene_sigmas         per-scene standard error and ST_QA uncertainty.

The perturbation magnitudes are module constants. analysis/validation_energy.py
uses these functions in its Monte Carlo propagation of retrieval errors to the
lake and relative temperature.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from pipeline.config import settings as S
from pipeline.config.sites import get_site
from pipeline.s2_lst_extract import PSF_SIGMA_M

def _col(d, name, default):
    """Numeric column, or a default-valued Series if the column is missing.

    (`DataFrame.get` returns a scalar for a missing column, which breaks `.fillna`.)"""
    if name in d.columns:
        return pd.to_numeric(d[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=d.index)


SEED = 20260827

#  Perturbation magnitudes and their basis.
SIGMA_EPS = 0.005      # water emissivity — spread of literature values 0.988–0.996
SIGMA_ATM_FALLBACK_K = 1.0   # used only when ST_QA is unavailable
GEOLOC_M = (0.0, 25.0, 50.0)      # geolocation error [m]: 0, 1/4 pixel, 1/2 pixel
#  Assumed number of background-ring pixels (not recorded). The ring is several
#  pixels wide, so 100 is conservative; smaller values give larger difference uncertainty.
N_BG_PIXELS = 100
PSF_SCALE = (0.8, 1.0, 1.2)       # effective PSF width ±20 %


# ---------------------------------------------------------------------------
#  1. How well is f known? — PSF width and geolocation
# ---------------------------------------------------------------------------
def psf_fraction(radius_m: float, psf_sigma_m: float, shift_m: float,
                 tir_m: float = 100.0, grid_m: float = 5.0) -> float:
    """Zone-mean PSF-weighted water fraction of a circular lake of radius `radius_m`.

    Same computation as the PSF-weighted f: water = 1 on a fine grid, Gaussian blur
    with σ = `psf_sigma_m`, aggregation to `tir_m` pixels, then the mean over
    all TIR pixels inside the lake. Done in numpy so that PSF width and grid
    offset can be varied freely.

    The zone mean (including edge pixels) is used, as in the pipeline; the
    central pixel alone is ~1.0 for large lakes and insensitive to PSF and
    geolocation.
    """
    half = 3 * psf_sigma_m + radius_m + 2 * tir_m
    n = int(2 * half / grid_m) + 1
    ax = (np.arange(n) - (n - 1) / 2) * grid_m
    X, Y = np.meshgrid(ax, ax)
    water = ((X ** 2 + Y ** 2) <= radius_m ** 2)

    from scipy.ndimage import convolve1d
    k_half = max(int(3 * psf_sigma_m / grid_m), 1)
    kx = np.exp(-0.5 * ((np.arange(-k_half, k_half + 1) * grid_m) / psf_sigma_m) ** 2)
    kx /= kx.sum()
    blur = convolve1d(convolve1d(water.astype(float), kx, axis=0, mode="nearest"),
                      kx, axis=1, mode="nearest")

    #  Offset the TIR grid by shift_m and average each box.
    step = int(round(tir_m / grid_m))
    off = int(round(shift_m / grid_m)) % step
    ny = (n - off) // step
    if ny < 1:
        return float("nan")
    sub = blur[off:off + ny * step, off:off + ny * step]
    wat = water[off:off + ny * step, off:off + ny * step]
    Xs = X[off:off + ny * step, off:off + ny * step]
    Ys = Y[off:off + ny * step, off:off + ny * step]
    box = sub.reshape(ny, step, ny, step).mean(axis=(1, 3))
    #  Lake pixel = box more than half water, analogous to the eroded water zone.
    inb = wat.reshape(ny, step, ny, step).mean(axis=(1, 3)) >= 0.5
    if not inb.any():
        #  Lake smaller than one pixel: use the box nearest the centre.
        cx = Xs.reshape(ny, step, ny, step).mean(axis=(1, 3))
        cy = Ys.reshape(ny, step, ny, step).mean(axis=(1, 3))
        inb = np.zeros_like(box, bool)
        inb[np.unravel_index(np.argmin(cx ** 2 + cy ** 2), box.shape)] = True
    return float(box[inb].mean())


def f_uncertainty_table(site_keys) -> pd.DataFrame:
    rows = []
    for k in site_keys:
        site = get_site(k)
        lake = getattr(site, "active", None)
        area = None
        #  Prefer the reference lake area (`area_km2_ref`); the Sentinel-2 observed
        #  area is a fallback, as it includes scenes with failed water detection.
        for attr in ("area_km2_ref", "area_km2", "area_m2"):
            v = getattr(lake, attr, None) if lake is not None else None
            if v:
                area = float(v) * (1.0 if attr == "area_m2" else 1e6)
                break
        if not area:
            p = S.records_dir("2b", k, create=False) / "water_extent.csv"
            if p.exists():
                d = pd.read_csv(p)
                for c in ("area_m2", "water_area_m2"):
                    if c in d.columns and d[c].notna().any():
                        area = float(pd.to_numeric(d[c], errors="coerce").median())
                        break
        if not area or not np.isfinite(area):
            continue
        R = float(np.sqrt(area / np.pi))
        base = psf_fraction(R, PSF_SIGMA_M, 0.0)
        vals = [psf_fraction(R, PSF_SIGMA_M * s, g)
                for s in PSF_SCALE for g in GEOLOC_M]
        rows.append(dict(
            site=k, lake_area_km2=round(area / 1e6, 4),
            equiv_radius_m=round(R, 1), f_nominal=round(base, 4),
            f_min=round(min(vals), 4), f_max=round(max(vals), 4),
            #  σ_f — spread of f from width/position errors, uniform assumption: range/√12.
            sigma_f=round(float((max(vals) - min(vals)) / np.sqrt(12)), 4),
            f_psf_only=round(psf_fraction(R, PSF_SIGMA_M * 1.2, 0.0), 4),
            f_geoloc_only=round(psf_fraction(R, PSF_SIGMA_M, 50.0), 4)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  2. Per-scene error sources
# ---------------------------------------------------------------------------
def scene_sigmas(site_key: str) -> pd.DataFrame:
    """Read two per-scene error sources of absolute temperature from the per-scene extraction table.

        SE      standard error of the lake-pixel mean = lst_std / √n (random, small).
        ST_QA   atmospheric-correction / emissivity uncertainty provided per pixel
                by Landsat Collection 2; dominates absolute accuracy.

    They are kept separate because ST_QA is largely common within a scene and
    cancels in lake − background differences, whereas SE does not.
    """
    p = S.records_dir(2, site_key, create=False) / "lst_long.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d.get("kind", "active").astype(str).eq("active")]
    if not len(d):
        return pd.DataFrame()
    se = (_col(d, "lst_std_K", 1.0)
          / np.sqrt(_col(d, "lst_n", 1.0).clip(lower=1)))
    out = pd.DataFrame(dict(
        date=pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        se_K=se.fillna(se.median()).to_numpy(),
        st_qa_K=_col(d, "st_qa_mean_K", SIGMA_ATM_FALLBACK_K).to_numpy(),
        bg_std_K=_col(d, "bg_std_K", 2.0).to_numpy()))
    return out.dropna(subset=["date"]).drop_duplicates("date")


