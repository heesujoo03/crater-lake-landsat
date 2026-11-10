# -*- coding: utf-8 -*-
"""
Radiance-space sub-pixel unmixing and local outlier scoring.

A Landsat thermal pixel over a small crater lake mixes water and the
surrounding land. Linear mixing holds for radiance, not for temperature, so the
lake temperature is recovered by inverting L_pix = f*L_lake + (1-f)*L_bg in
radiance space and converting back with the Planck function.

Contents
    planck_radiance, planck_temperature, dL_dT
        Planck conversion at the effective wavelength of Landsat TIRS band 10
        and its temperature derivative (converts radiance errors to K).
    unmix
        Mixing inversion with error propagation; not applied below f_floor.
    local_robust_z
        Robust z-score against temporal neighbours, used for outlier flagging.

The analysis modules call these functions to recompute the corrected lake
temperature under alternative water fractions, spatial-response models and
thermal-filter settings, and to propagate retrieval errors.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Radiance <-> temperature (Planck, effective wavelength of Landsat TIRS band 10)
# ---------------------------------------------------------------------------
# Unmixing must be done in radiance space: linear mixing holds for radiance,
# not for temperature. Mixing in temperature space introduces a systematic
# error equal to the Planck non-linearity, which exceeds 1 K for a 40 K contrast.

_C1 = 1.191042e8      # W·m^-2·sr^-1·µm^4
_C2 = 1.4387752e4     # µm·K
LAMBDA_UM = 10.9      # TIRS band 10 effective wavelength


def planck_radiance(T_K, lam=LAMBDA_UM):
    """Spectral radiance [W m-2 sr-1 um-1] at wavelength lam for temperature T_K."""
    T = np.asarray(T_K, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _C1 / (lam ** 5 * (np.exp(_C2 / (lam * T)) - 1.0))


def planck_temperature(L, lam=LAMBDA_UM):
    """Brightness temperature [K] from spectral radiance (inverse Planck)."""
    L = np.asarray(L, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _C2 / (lam * np.log(_C1 / (lam ** 5 * L) + 1.0))


def dL_dT(T_K, lam=LAMBDA_UM):
    """Temperature derivative of the Planck function (converts radiance errors to K)."""
    T = np.asarray(T_K, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = _C2 / (lam * T)
        return _C1 * _C2 * np.exp(x) / (lam ** 6 * T ** 2 * (np.exp(x) - 1.0) ** 2)


# ---------------------------------------------------------------------------
# Sub-pixel unmixing
# ---------------------------------------------------------------------------

def unmix(T_pix_K, T_bg_K, f, sigma_pix_K, sigma_bg_K, sigma_f=0.10,
          f_floor=0.15):
    """Recover the pure water temperature by inverting L_pix = f*L_lake + (1-f)*L_bg.

    Returns (T_lake_K, sigma_K, applied).
    The inversion becomes ill-conditioned as f decreases; for f < f_floor it is
    not applied and the input value is returned unchanged (applied = False).
    """
    T_pix = np.asarray(T_pix_K, float)
    T_bg = np.asarray(T_bg_K, float)
    f = np.asarray(f, float)
    s_pix = np.asarray(sigma_pix_K, float)
    s_bg = np.asarray(sigma_bg_K, float)

    ok = np.isfinite(f) & (f >= f_floor) & (f <= 1.0) & np.isfinite(T_bg)
    f_ = np.where(ok, f, 1.0)

    L_pix, L_bg = planck_radiance(T_pix), planck_radiance(T_bg)
    L_lake = L_bg + (L_pix - L_bg) / f_
    L_lake = np.where(L_lake > 0, L_lake, np.nan)
    T_lake = planck_temperature(L_lake)

    # Error propagation: partial derivatives in radiance space, converted to K.
    sL_pix = s_pix * dL_dT(T_pix)
    sL_bg = s_bg * dL_dT(T_bg)
    var = ((sL_pix / f_) ** 2
           + (sL_bg * (1.0 - 1.0 / f_)) ** 2
           + (((L_pix - L_bg) / f_ ** 2) * sigma_f) ** 2)
    sigma_T = np.sqrt(var) / np.maximum(dL_dT(T_lake), 1e-12)

    T_out = np.where(ok, T_lake, T_pix)
    s_out = np.where(ok, sigma_T, s_pix)
    return T_out, s_out, ok


# ---------------------------------------------------------------------------
# Local outlier score
# ---------------------------------------------------------------------------

def local_robust_z(t, x, min_periods=5):
    """Robust z-score relative to temporal neighbours; used for outlier flagging.

    A z-score against the global median flags a sustained regime change (for
    example the post-eruption state of a lake) as a block of outliers, so the
    filter would remove the very signal being sought. An outlier is a point
    that differs from its neighbours, not from the overall mean. The reference
    is therefore a centred rolling median, and the spread is also estimated
    locally: sustained changes (neighbours also high) pass, isolated spikes are
    flagged. The window adapts to the median sampling gap.

    Returns (z, window_string).
    """
    t = pd.to_datetime(pd.Series(t), utc=True, format="mixed")
    s = pd.Series(np.asarray(x, float)).set_axis(pd.DatetimeIndex(t))
    s = s.sort_index()
    n = len(s)
    if n >= 3:
        gap_d = float(np.median(np.diff(s.index.view("int64"))) / 8.64e13)
    else:
        gap_d = 30.0
    win = f"{int(max(60, min(730, gap_d * 12)))}D"

    base = s.rolling(win, center=True, min_periods=min_periods).median()
    dev = s - base
    # Local spread, in case a quiet record becomes noisy.
    mad = dev.abs().rolling(win, center=True, min_periods=min_periods).median()
    scale = (1.4826 * mad).replace(0, np.nan)
    scale = scale.fillna(1.4826 * float(np.nanmedian(np.abs(dev))) or 1.0)
    z = (dev / scale).reindex(pd.DatetimeIndex(t)).to_numpy()
    return z, win


