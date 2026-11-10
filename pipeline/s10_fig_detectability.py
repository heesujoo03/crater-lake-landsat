# -*- coding: utf-8 -*-
"""Plotting function for detectability and lake persistence.

  fig_detectability   noise geometry, variance partition and persistence per lake

Input: results/10_detectability/ (s10_detectability). PRE points to
results/10_pre_eruption/, which analysis/figures.py reads for the pre-onset
figure. analysis/figures.py re-executes fig_detectability with revised labels
and layout and writes the final figure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import settings as S
from pipeline import s10_figures as F

DET = S.ROOT / "results" / "10_detectability"
PRE = S.ROOT / "results" / "10_pre_eruption"
OUT = F.OUT


def fig_detectability():
    """(a) Predicted vs measured noise, (b) noise vs pair separation,
    (c) variance partition, (d) observed persistence vs depth-based tau."""
    d = pd.read_csv(DET / "detectability_by_site.csv")
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.2))

    # (a) predicted vs measured; no free parameters
    a = ax[0, 0]
    for r in d.itertuples():
        c = F.COLOR.get(r.site, F.GREY)
        a.plot(r.sigma_meas_predicted_K, r.sigma_meas_K, "o", ms=11, color=c,
               mec="#0b0b0b", mew=0.8, zorder=3)
        a.annotate(F.NAME[r.site], (r.sigma_meas_predicted_K, r.sigma_meas_K),
                   xytext=(9, -4), textcoords="offset points", fontsize=9,
                   color=F.INK2)
    lim = float(np.nanmax([d.sigma_meas_K.max(), d.sigma_meas_predicted_K.max()])) * 1.25
    a.plot([0, lim], [0, lim], "--", color=F.GREY, lw=1.0, zorder=1)
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("Predicted from pixel count and water fraction (K)")
    a.set_ylabel("Measured from close scene pairs (K)")
    a.set_title("(a) Measurement noise is predictable from geometry",
                loc="left", fontsize=10)

    # (b) by pair separation: flat = measurement noise, rising = real change entering
    b = ax[0, 1]
    seps = [3, 8, 16]
    for r in d.itertuples():
        y = [getattr(r, f"sigma_meas_{k}d_K") for k in seps]
        c = F.COLOR.get(r.site, F.GREY)
        b.plot(seps, y, "-o", color=c, ms=7, lw=1.8, label=F.NAME[r.site])
    b.set_xticks(seps)
    b.set_xlabel("Maximum separation of the scene pair (days)")
    b.set_ylabel("Estimated noise σ (K)")
    b.set_title("(b) Rising means real change is entering", loc="left", fontsize=10)
    b.legend(fontsize=8, frameon=False)

    # (c) variance partition
    c_ax = ax[1, 0]
    x = np.arange(len(d))
    meas = d.sigma_meas_K ** 2
    real = d.sigma_real_K ** 2
    c_ax.bar(x, meas, 0.55, color="#c2415a", label="Measurement")
    c_ax.bar(x, real, 0.55, bottom=meas, color="#2a78d6", label="Lake, whole record")
    #  variability within quiet segments (eruption windows removed); at Taal and
    #  Santa Ana the whole-record lake share is dominated by eruption-related steps
    q = d.sigma_real_quiet_K ** 2
    c_ax.plot(x, meas + q, "_", ms=26, mew=2.4, color="#0b0b0b", zorder=4,
              label="Lake, quiet periods only")
    for i, r in enumerate(d.itertuples()):
        c_ax.text(i, (meas.iloc[i] + real.iloc[i]) * 1.02,
                  f"{r.meas_variance_share*100:.0f} % measurement",
                  ha="center", fontsize=8, color=F.INK2)
    c_ax.set_xticks(x); c_ax.set_xticklabels([F.NAME[s] for s in d.site])
    c_ax.set_ylabel("Variance (K²)")
    c_ax.set_title("(c) How much of the variance is the instrument",
                   loc="left", fontsize=10)
    c_ax.legend(fontsize=8, frameon=False)
    c_ax.margins(y=0.16)

    # (d) observed persistence vs tau computed from mean depth.
    #  rho*c*h/K is the relaxation time after the heat supply stops, so it is a lower
    #  bound on observed persistence; points above 1:1 indicate a sustained supply.
    e = ax[1, 1]
    e.plot([1, 300], [1, 300], "--", color=F.GREY, lw=1.0, zorder=0)
    e.annotate("1 : 1", (170, 190), fontsize=8, color=F.GREY, rotation=38)
    for r in d.itertuples():
        col = F.COLOR.get(r.site, F.GREY)
        if not np.isfinite(r.tau_obs_d):
            continue
        lo_t, hi_t = r.tau_days_lo, r.tau_days_hi
        if not np.isfinite(lo_t):
            continue
        #  horizontal: range of depth-based tau; vertical: range of observed tau
        e.plot([lo_t, hi_t], [r.tau_obs_d] * 2, "-", color=col, lw=3,
               solid_capstyle="round", alpha=0.85)
        e.plot([(lo_t + hi_t) / 2] * 2, [r.tau_obs_lo_d, r.tau_obs_hi_d], "-",
               color=col, lw=1.6, solid_capstyle="round")
        e.plot((lo_t + hi_t) / 2, r.tau_obs_d, "o", ms=10, color=col,
               mec="#0b0b0b", mew=0.8, zorder=3)
        e.annotate(F.NAME[r.site], ((lo_t + hi_t) / 2, r.tau_obs_d),
                   xytext=(11, -3), textcoords="offset points", fontsize=9,
                   color=F.INK2, va="center")
    e.set_xscale("log"); e.set_yscale("log")
    e.set_xlabel("Predicted from mean depth, τ = ρch/K (days)")
    e.set_ylabel("Measured from the record (days)")
    e.set_title("(d) How long a lake actually holds its state",
                loc="left", fontsize=10)
    e.set_xlim(2, 300); e.set_ylim(1, 300)

    for a_ in ax.ravel():
        a_.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    return F._save(fig, "fig_detectability")


