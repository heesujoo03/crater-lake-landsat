# -*- coding: utf-8 -*-
"""
Time-varying water fraction f for the Landsat thermal pixels.

A shrinking lake lowers the water fraction f inside each 100 m TIR pixel and
pulls the observed temperature towards the surrounding land, so a constant f
would turn a level change into a spurious temperature trend. The functions here
replace the constant f with a date-specific value:

    geometric_f_ref      constant reference f of a lake (from the extracted
                         per-scene f, or an analytic disc approximation)
    build_f_timevarying  f for every Landsat scene date, interpolated in time
                         between Sentinel-2 water-area observations, with the
                         constant f as fallback

The analysis (analysis/unmixing_psf.py) calls build_f_timevarying with
alternative reference fractions to test the sensitivity to the spatial-response
model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


F_TIR_FLOOR = 0.01            # lower bound on f; unmixing divides by f, so it must not be 0
F_INTERP_MAX_GAP_DAYS = 120   # do not interpolate from S2 observations farther away than this
TIR_PIXEL_M = 100.0           # nominal Landsat TIRS GSD


def geometric_f_ref(lake, lst_df=None, tir_scale=TIR_PIXEL_M):
    """Time-invariant water fraction f of a lake at its reference area.

    Uses the median f_est (or f_max) of the extracted per-scene table when
    available, so that the constant reference matches the per-scene values.
    Otherwise approximates the maximum coverage of a TIR pixel (L x L) by a
    disc of radius r.
    """
    if lst_df is not None and len(lst_df):
        g = lst_df[lst_df["lake"] == lake.key]
        for col in ("f_est", "f_max"):
            if col in g.columns:
                v = pd.to_numeric(g[col], errors="coerce").median()
                if np.isfinite(v) and v > 0:
                    return float(min(1.0, v)), "stage2_lst"
    r, L = float(lake.radius_m), float(tir_scale)
    if r >= L * math.sqrt(2.0) / 2.0:
        return 1.0, "analytic_disc"      # at least one pixel lies entirely inside the lake
    return float(min(1.0, math.pi * r * r / (L * L))), "analytic_disc"


def build_f_timevarying(water, site, inventory=None, lst_df=None):
    """Build the f lookup for every Landsat scene date.

    f is interpolated in time between S2 observations. Where S2 coverage is
    missing (no interpolation possible) the constant geometric f is used. The
    origin is always recorded in f_source.
    """
    rows = []
    for lake in site.lakes:
        w = water[water["lake"] == lake.key]
        # Threshold failures (area_outlier) and clouds are not used for
        # interpolation; no_water is a real state and is always included.
        good = w[w["qc_flag"].isin(["ok", "no_water"])].dropna(subset=["f_tir"])
        f_const, _ = geometric_f_ref(lake, lst_df)

        if inventory is not None and len(inventory):
            inv = inventory[inventory["lake"] == lake.key]
            dates = pd.to_datetime(inv["date"], errors="coerce").dropna().unique()
        else:
            dates = pd.to_datetime(w["date"], errors="coerce").dropna().unique()
        dates = np.sort(pd.to_datetime(pd.Series(dates)).unique())

        if len(good) >= 2:
            sx = pd.to_datetime(good["date"]).to_numpy().astype("datetime64[D]").astype(float)
            sy = good["f_tir"].to_numpy(dtype=float)
            order = np.argsort(sx)
            sx, sy = sx[order], sy[order]
        else:
            sx = sy = None

        for dt in dates:
            dnum = np.datetime64(pd.Timestamp(dt).date(), "D").astype(float)
            if sx is not None:
                gap = float(np.min(np.abs(sx - dnum)))
                inside = (dnum >= sx[0]) and (dnum <= sx[-1])
                if inside and gap <= F_INTERP_MAX_GAP_DAYS:
                    f = float(np.interp(dnum, sx, sy))
                    src = "s2_interpolated"
                else:
                    f, src = float(f_const), "geometric_fallback"
            else:
                f, src = float(f_const), "geometric_fallback"
            rows.append(dict(site=site.key, lake=lake.key,
                             date=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                             f_tir=float(np.clip(f, F_TIR_FLOOR, 1.0)),
                             f_source=src))
    out = pd.DataFrame(rows, columns=["site", "lake", "date", "f_tir", "f_source"])
    return out.sort_values(["lake", "date"]).reset_index(drop=True)


