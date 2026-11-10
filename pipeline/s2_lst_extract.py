# -*- coding: utf-8 -*-
"""
Landsat thermal-infrared point-spread constant and a diagnostic plot.

PSF_SIGMA_M is the Gaussian width of the effective TIR point-spread function
that defines the PSF-weighted water fraction f; s9d_retrieval_unc uses it as the
nominal width when it perturbs f. fig_timeseries plots the per-lake surface
temperature and background-ring temperature with the site's known events.
"""

from __future__ import annotations


import pandas as pd


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Per-pixel water fraction f
# ---------------------------------------------------------------------------

#  Gaussian approximation of the effective TIR point-spread function, sigma [m].
#
#  A Landsat TIR pixel is nominally 100 m, but only 37.4 % of its signal
#  originates inside that 100 m square. For a 2-D Gaussian the energy inside a
#  100 m box is [erf(50 / (sigma*sqrt(2)))]^2; solving for 0.374 gives sigma = 58 m.
#  The physical (PSF-weighted) water fraction is therefore always smaller than
#  the geometric one: signal from land beyond the shoreline leaks into the pixel.
PSF_SIGMA_M = 58.0


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def fig_timeseries(df, site):
    """LST and background temperature time series per lake, with known events."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = {"active": "#A83E2F", "control": "#2F6D8C"}
    for lk, g in df.groupby("lake"):
        t = pd.to_datetime(g["datetime"], utc=True, format="mixed")
        kind = g["kind"].iloc[0]
        ax.plot(t, g["lst_median_K"] - 273.15, ".-", ms=4, lw=0.7, alpha=0.85,
                color=colors.get(kind, "#7A8B94"), label=f"{lk} ({kind})")
        ax.plot(t, g["bg_mean_K"] - 273.15, ":", lw=0.9, alpha=0.45,
                color=colors.get(kind, "#7A8B94"), label=f"{lk} background")
    for ev in site.known_events:
        ax.axvline(pd.Timestamp(ev["date"], tz="UTC"), color="#C07A2B",
                   ls="--", lw=1.1, alpha=0.7)
    ax.set_ylabel("LST (°C)")
    ax.set_title(f"{site.label} — lake and background temperature (vertical lines: known activity)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


