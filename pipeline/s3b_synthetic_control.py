# -*- coding: utf-8 -*-
"""Synthetic control and the alternative reference designs for the crater-lake temperature.

A simple active-minus-control difference implicitly assumes the active lake responds
to weather with unit gain (beta1 = 1). The two water bodies differ in thermal inertia,
depth and dilution rate, so that gain is estimated instead. The synthetic control is a
weighted combination of donor series that best reproduces the active lake during the
quiet reference period; it does not depend on a single paired lake, so it remains
usable where no suitable control lake exists.

Reference designs
-----------------
    A simple difference   dT = T_active - T_control                 (beta1 forced to 1)
    B regression diff.    dT = T_active - (beta0 + beta1*T_control)  (beta1 estimated)
    C climatology         dT = T_active - DOY harmonics              (no spatial reference)
    D synthetic control   dT = T_active - sum_i w_i * donor_i        (convex combination)

Donors are the control-lake temperature, the background land ring around the active
lake, the seasonal harmonic fit and, where available, nearby water bodies. Weights are
fitted on the quiet period only, constrained to w >= 0 and sum(w) = 1 (the standard
synthetic-control form), which restricts the reference to interpolation within the
donor range rather than extrapolation.

`series` returns the design series and fit metadata; s9b_event_stats.frame builds the
weather-normalised relative temperature on top of design D. The analysis modules reuse
`_core`, `_donor_pool`, `_doy_terms` and `_simplex_fit` to refit the reference under
alternative fitting samples.

Inputs
------
    data/processed/<site>_deltaT.csv       mixing-corrected per-scene temperatures
    data/interim/<site>_donor_lst.csv      (optional nearby water-body donors)
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from pipeline.common import baseline as B
from pipeline.config import settings as S
from pipeline.config.sites import get_site


def _doy_terms(t):
    """Annual and semi-annual harmonics of day-of-year.

    Used both as the climatology reference (design C) and as a donor in design D.
    """
    doy = pd.to_datetime(t, utc=True).dt.dayofyear.to_numpy(float)
    return np.c_[np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                 np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)]


def _simplex_fit(Y, X, ridge=1e-6):
    """Solve Y ~ X @ w subject to w >= 0 and sum(w) = 1.

    The sum-to-one constraint is imposed as one heavily weighted extra row, so the
    problem reduces to non-negative least squares (scipy.optimize.nnls) without an
    additional optimisation dependency. A small ridge term stabilises collinear donors.
    """
    from scipy.optimize import nnls
    n, k = X.shape
    big = 1e3 * (np.abs(Y).max() + 1.0)
    A = np.vstack([X, big * np.ones((1, k))])
    b = np.concatenate([Y, [big]])
    if ridge:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(k)])
        b = np.concatenate([b, np.zeros(k)])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(k, 1.0 / k)


def _donor_pool(site_key, t_index):
    """Read nearby water-body LST series (<site>_donor_lst.csv) as donors.

    Only water bodies confirmed by JRC surface-water occurrence (>= 50 %) are used.
    A lake that correlates too weakly to serve as a control on its own can still
    improve a weighted combination, which is the point of a synthetic control.

    Returns a dict {label: array aligned to t_index}.
    """
    p = S.INTERIM_DIR / f"{site_key}_donor_lst.csv"
    if not p.exists():
        return {}
    try:
        d = pd.read_csv(p)
    except Exception:                                          # noqa: BLE001
        return {}
    if not {"donor", "date", "T"} <= set(d.columns):
        return {}
    d = d[pd.to_numeric(d.get("jrc_occurrence"), errors="coerce").fillna(0) >= 50]
    if not len(d):
        return {}
    d["day"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    day = pd.to_datetime(t_index, utc=True).dt.tz_localize(None).dt.normalize()
    out = {}
    for k, g in d.groupby("donor"):
        ser = (g.dropna(subset=["day"]).groupby("day")["T"].median()
               .reindex(day.to_numpy()))
        v = ser.to_numpy(float)
        #  A quiet-period fit needs most dates; require at least half to be present.
        if np.isfinite(v).mean() < 0.5:
            continue
        #  Fill remaining gaps with the donor's own median. A donor acts as a proxy
        #  for regional weather, so a missing value is set to its typical state.
        v = np.where(np.isfinite(v), v, np.nanmedian(v))
        out[f"nearby water body {k}"] = v
    return out


def series(site_key, events=None, exclude_days=None, f_min=None):
    """Return the four reference-design series as a DataFrame plus fit metadata.

    The series are the input of the weather normalisation in
    s9b_event_stats.frame. If `events` is given, it defines the quiet/unrest
    split. Because design weights are fitted on the quiet period only, they are
    refitted for each event definition; sensitivity analyses that vary the
    activity definition rely on this.
    """
    d, out, quiet, unrest, meta = _core(site_key, events, exclude_days, f_min)
    if d is None:
        return None
    df = pd.DataFrame({"datetime": d["datetime"], "t": d["t"],
                       "T_active_K": d["T_active_K"]})
    for k, v in out.items():
        df[k] = v
    df["_quiet"] = quiet
    df["_unrest"] = unrest
    return df, meta


def _core(site_key, events=None, exclude_days=None, f_min=None):
    """Build the four reference-design series and the quiet/unrest classification.

    events=None uses the site's default activity list (eruptions and unrest windows).
    exclude_days is the buffer added around each event. Pass 0 when events are given
    as explicit intervals; adding a +-180-day buffer to intervals that already span
    the activity would classify almost the whole record as unrest.
    f_min drops scenes whose estimated lake fraction of the thermal pixel is below
    this value: unmixing amplifies noise by 1/f, and scenes with a falsely low f
    (failed water detection) can inflate the retrieved temperature by tens of K.
    None disables the screen.
    """
    site = get_site(site_key)
    p = S.PROCESSED_DIR / f"{site_key}_deltaT.csv"
    if not p.exists():
        return (None,) * 5
    d = pd.read_csv(p)
    d["t"] = pd.to_datetime(d["datetime"], utc=True, format="mixed")
    d = d[d["qc_flag"].eq("ok")]
    if f_min is not None and "f_est_act" in d.columns:
        d = d[pd.to_numeric(d["f_est_act"], errors="coerce") >= float(f_min)]
    d = d.sort_values("t").reset_index(drop=True)
    if len(d) < 20:
        return (None,) * 5

    #  Shared baseline rule (common.baseline). When the event list starts early in
    #  the record there may be no pre-event samples, so the baseline rule is allowed
    #  to widen (min_n) and reports whether it did.
    _acts = list(events)
    _buf = S.BASELINE_EXCLUDE_DAYS if exclude_days is None else int(exclude_days)
    cls = B.classify(d["t"], _acts, _buf).to_numpy()
    brep = {}
    quiet = B.in_baseline(d["t"], _acts, _buf,
                          qc_flag=d.get("qc_flag"), min_n=60,
                          report=brep).to_numpy()
    unrest = cls == "unrest"

    y = pd.to_numeric(d["T_active_K"], errors="coerce").to_numpy(float)
    ctl = pd.to_numeric(d.get("T_control_K"), errors="coerce").to_numpy(float)
    bg = pd.to_numeric(d.get("bg_active_K"), errors="coerce").to_numpy(float)
    dow = _doy_terms(d["t"])

    ok = np.isfinite(y)
    fit = quiet & ok & np.isfinite(ctl) & np.isfinite(bg)
    if fit.sum() < 12:
        return (None,) * 5

    out = {}

    # A: simple difference (beta1 = 1)
    out["A simple difference"] = y - ctl

    # B: regression difference (beta0, beta1 fitted on the quiet period)
    X = np.c_[np.ones(len(y)), ctl]
    beta, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
    out["B regression difference"] = y - X @ beta
    b1 = float(beta[1])

    # C: climatology (seasonal harmonics only, no spatial reference)
    Xc = np.c_[np.ones(len(y)), dow]
    bc, *_ = np.linalg.lstsq(Xc[fit], y[fit], rcond=None)
    out["C climatology"] = y - Xc @ bc

    # D: synthetic control, a convex combination of several donors
    clim = Xc @ bc
    donors = [ctl, bg, clim]
    names = ["control_lake", "background_ring", "seasonal_cycle"]

    #  Add nearby water bodies as donors. Without them, after de-duplication below,
    #  the combination may reduce to background ring plus seasonal cycle only.
    for nm, ser in _donor_pool(site_key, d["t"]).items():
        donors.append(ser)
        names.append(nm)

    D = np.column_stack(donors)
    #  Remove duplicate donors. In bg_ref mode T_control_K is the background ring
    #  itself; keeping both would split one weight across identical columns.
    keep, seen = [], []
    for i in range(D.shape[1]):
        col = D[:, i]
        if not np.isfinite(col[fit]).all():
            continue
        if any(np.allclose(col[fit], D[:, j][fit], atol=1e-9) for j in seen):
            continue
        seen.append(i)
        keep.append(i)
    D, names = D[:, keep], [names[i] for i in keep]
    w = _simplex_fit(y[fit], D[fit])
    out["D synthetic control"] = y - D @ w
    weights = dict(zip(names, np.round(w, 3)))
    n_donor = len(names)

    meta = dict(beta1=round(b1, 3), weights=weights, n_donor=n_donor,
                mode=str(d["mode"].iloc[0]) if "mode" in d.columns else "?",
                baseline_rule=brep.get("baseline_rule"),
                baseline_widened=bool(brep.get("widened")))
    return d, out, quiet, unrest, meta


