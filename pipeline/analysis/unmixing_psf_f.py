# -*- coding: utf-8 -*-
"""Local, analytic recomputation of the PSF-weighted
water fraction f for alternative (sensor-specific) spatial-response models.

Reproduces, in numpy, the PSF-weighted water fraction of the pipeline:
    base   = 0/1 water mask of the *un-eroded* water polygon (AOI buffered by erode_m_used)
    f(x)   = (base (*) G_sigma (*) Box_a)(x)      Gaussian PSF convolved, then box-averaged
    f_max  = max of f over 100 m grid cells whose centre lies in the eroded AOI
    f_mean = mean of f over the same cells
The pipeline uses sigma = 58 m and a = 100 m for every sensor.
Nothing here writes to existing records; this module only returns numbers.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import ndimage
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from shapely import vectorized

from pipeline.config import settings as S

AOI_DIR = S.DATA_DIR / "aoi"


def _load(site, lake):
    gj = json.loads((AOI_DIR / f"{site}_{lake}.geojson").read_text(encoding="utf-8"))
    g = shape(gj["geometry"])
    pr = gj.get("properties", {})
    m = float(pr.get("erode_m_used") or pr.get("erode_m_requested") or 0)
    c = g.centroid
    tr = Transformer.from_crs("EPSG:4326",
                              f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m",
                              always_xy=True)
    aoi = transform(tr.transform, g)
    water = aoi.buffer(m) if m > 0 else aoi
    return aoi, water, m


def f_field(site, lake, sigma_m, box_m, dx=5.0, gaussian_only=False):
    """Return (xx, yy, f, aoi) on a dx grid. gaussian_only -> no box term."""
    aoi, water, _ = _load(site, lake)
    pad = 4 * sigma_m + box_m + 200
    minx, miny, maxx, maxy = water.bounds
    xs = np.arange(minx - pad, maxx + pad, dx)
    ys = np.arange(miny - pad, maxy + pad, dx)
    xx, yy = np.meshgrid(xs, ys)
    base = vectorized.contains(water, xx, yy).astype(float)
    f = ndimage.gaussian_filter(base, sigma=sigma_m / dx, mode="constant",
                                truncate=3.0) if sigma_m > 0 else base
    if not gaussian_only and box_m > 0:
        n = max(1, int(round(box_m / dx)))
        f = ndimage.uniform_filter(f, size=n, mode="constant")
    return xx, yy, f, aoi


def f_stats(site, lake, sigma_m, box_m, grid_m=100.0, n_phase=16, dx=5.0,
            gaussian_only=False, seed=0):
    """f_max / f_mean over grid-cell centres inside the eroded AOI.

    The grid phase of the 100 m extraction grid is arbitrary relative to the lake,
    so the statistic is averaged over n_phase random phases; the phase-free
    continuous maximum is also returned (upper bound)."""
    xx, yy, f, aoi = f_field(site, lake, sigma_m, box_m, dx, gaussian_only)
    inside = vectorized.contains(aoi, xx, yy)
    rng = np.random.default_rng(seed)
    step = int(round(grid_m / dx))
    fmax, fmean = [], []
    for _ in range(n_phase):
        oy, ox = rng.integers(step, size=2)
        sub_f = f[oy::step, ox::step]
        sub_in = inside[oy::step, ox::step]
        if sub_in.any():
            fmax.append(sub_f[sub_in].max())
            fmean.append(sub_f[sub_in].mean())
    return dict(f_max=float(np.mean(fmax)) if fmax else np.nan,
                f_max_phase_min=float(np.min(fmax)) if fmax else np.nan,
                f_max_phase_max=float(np.max(fmax)) if fmax else np.nan,
                f_mean=float(np.mean(fmean)) if fmean else np.nan,
                f_max_continuous=float(f[inside].max()) if inside.any() else np.nan,
                f_mean_continuous=float(f[inside].mean()) if inside.any() else np.nan)


def encircled_sigma(box_m, frac=0.374):
    """sigma such that [erf(box/2 / (sigma*sqrt2))]^2 = frac (pipeline convention)."""
    from scipy.optimize import brentq
    return brentq(lambda s: math.erf(box_m / 2 / (s * math.sqrt(2))) ** 2 - frac, 1, 1000)
