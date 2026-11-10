# -*- coding: utf-8 -*-
"""Weather-normalised relative temperature and temporally aware event statistics.

`frame` builds the analysis table of one site: the reference-design series of
s3b_synthetic_control, ERA5-Land meteorology and surface heat loss, with
lake-absent periods removed. It adds the primary estimator

    dT_syn_weather = synthetic-control residual minus the part explained by
                     ERA5-Land meteorology (MET_COLS)

The weather regression is fitted on quiet scenes only, and is refitted for every
activity definition, so that definitions are compared with the same estimator.

Satellite scenes within one activity episode are days to weeks apart and are not
independent samples, so the statistics avoid label shuffling:

  circular_shift_p     null distribution from shifting the whole set of event
                       windows along the time axis (window length, number and
                       spacing preserved)
  block_bootstrap_ci   confidence interval from resampling contiguous time blocks

`insitu_heating_events` defines Ruapehu heating episodes from the in-situ lake
temperature and `definitions` lists the activity definitions of a site. These are
used by s10_post_eruption, s9j_within_volcano and the analysis modules.

Inputs
------
`data/processed/<site>_deltaT.csv` (via s3b_synthetic_control),
`<site>_dataset.csv` (ERA5-Land meteorology), `<site>_energy_balance.csv`
(surface heat loss), eruption lists from `config/sites.py` and the optional
activity intervals of `config/thermal_episodes.py`.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from pipeline import s3b_synthetic_control as SC
from pipeline.config import settings as S
from pipeline.config import thermal_episodes as TE
from pipeline.config.sites import get_site

N_PERM = 20000
SEED = 20260827
WINDOWS = (30, 60, 90, 180, 365)

#  Predictors for weather normalisation — all ERA5-Land, independent of volcanic activity.
MET_COLS = ["Ta_K", "RH", "VPD_kPa", "wind_ms", "SWdown_Wm2", "LWdown_Wm2",
            "precip_mm_sum24h", "Ta_K_lag24h", "SWdown_Wm2_lag24h"]


# ---------------------------------------------------------------------------
#  1. Analysis frame
# ---------------------------------------------------------------------------
def frame(site_key: str, events, buffer_days: int,
          f_min: float | None = None) -> pd.DataFrame | None:
    """Estimators, meteorology and heat loss in one table; `events` define quiet vs. active."""
    got = SC.series(site_key, events, buffer_days, f_min=f_min)
    if got is None:
        return None
    df, _meta = got
    df = df.rename(columns={"A simple difference": "dT_simple", "B regression difference": "dT_regress",
                            "C climatology": "dT_clim", "D synthetic control": "dT_syn"})
    df["date"] = pd.to_datetime(df["t"], utc=True).dt.tz_localize(None).dt.normalize()

    #  ERA5-Land meteorology from the <site>_dataset.csv table.
    p = S.PROCESSED_DIR / f"{site_key}_dataset.csv"
    if p.exists():
        ds = pd.read_csv(p)
        ds["date"] = pd.to_datetime(ds["date"], errors="coerce").dt.normalize()
        keep = ["date"] + [c for c in MET_COLS if c in ds.columns]
        df = df.merge(ds[keep].drop_duplicates("date"), on="date", how="left")

    #  Surface heat loss
    p = S.PROCESSED_DIR / f"{site_key}_energy_balance.csv"
    if p.exists():
        eb = pd.read_csv(p)
        eb["date"] = pd.to_datetime(eb["date"], errors="coerce").dt.normalize()
        col = [c for c in ("Q_total_MW", "Q_geo_MW") if c in eb.columns]
        if col:
            df = df.merge(eb[["date"] + col].drop_duplicates("date"),
                          on="date", how="left")

    #  Drop periods without a lake: those pixels are dry crater floor.
    df = df[~TE.absent_mask(site_key, df["date"])].reset_index(drop=True)

    #  SCW — remove the weather-explained part of the synthetic-control residual.
    #  The regression is fitted on quiet scenes only; including active scenes
    #  would let the weather terms absorb (and erase) the signal.
    met = [c for c in MET_COLS if c in df.columns
           and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7]
    df["dT_syn_weather"] = np.nan
    if met:
        y = pd.to_numeric(df["dT_syn"], errors="coerce").to_numpy(float)
        X = df[met].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        #  Fill gaps with the quiet-period median of each predictor.
        fit = df["_quiet"].to_numpy(bool) & np.isfinite(y)
        for j in range(X.shape[1]):
            med = np.nanmedian(X[fit, j]) if fit.any() else np.nanmedian(X[:, j])
            X[~np.isfinite(X[:, j]), j] = med if np.isfinite(med) else 0.0
        Xd = np.c_[np.ones(len(y)), X]
        good = fit & np.isfinite(Xd).all(axis=1)
        if good.sum() >= 3 * Xd.shape[1]:
            beta, *_ = np.linalg.lstsq(Xd[good], y[good], rcond=None)
            df["dT_syn_weather"] = y - Xd @ beta
    return df


ESTIMATORS = [
    ("dT_syn_weather", "Synthetic control, weather-normalised"),   # primary estimator
    ("dT_syn", "Synthetic control"),
    ("dT_simple", "Paired difference"),
    ("dT_regress", "Regression difference"),
    ("dT_clim", "Climatology residual"),
    ("T_active_K", "Lake surface temperature"),
    ("Q_total_MW", "Total surface heat loss"),
]
#  Primary estimator, fixed before looking at results; all others are sensitivity analyses.
PRIMARY = "dT_syn_weather"


def _labels_from_windows(dates: np.ndarray, wins) -> np.ndarray:
    m = np.zeros(len(dates), bool)
    for t0, t1 in wins:
        m |= (dates >= t0) & (dates <= t1)
    return m


def _masked_median_diff(Y: np.ndarray, M: np.ndarray, min_b=3, min_a=5):
    """Per row: median where M minus median where not M; NaN if too few samples.

    Vectorised with `np.where` + a single `nanmedian` call, since the number of
    rows (permutations) is large.
    """
    with np.errstate(invalid="ignore"):
        b = np.nanmedian(np.where(M, Y, np.nan), axis=1)
        a = np.nanmedian(np.where(M, np.nan, Y), axis=1)
    nb, na = np.sum(M & np.isfinite(Y), axis=1), np.sum(~M & np.isfinite(Y), axis=1)
    out = b - a
    out[(nb < min_b) | (na < min_a)] = np.nan
    return out, nb, na


def circular_shift_p(dates: pd.Series, y: np.ndarray, wins,
                     n_perm=N_PERM, seed=SEED):
    """Null distribution from shifting all event windows together along the time axis.

    Shuffling labels individually counts adjacent scenes as independent samples
    and makes p too small. Shifting the windows as a block preserves their
    length, number and spacing, so the temporal correlation of the data enters
    the null distribution.
    """
    d = dates.to_numpy("datetime64[D]").astype("int64")
    span = int(d.max() - d.min()) + 1
    #  Discard windows with missing bounds (NaT becomes int64 min and would overflow).
    lim = 400 * 365
    wins = [(t0, t1) for t0, t1 in wins
            if abs(int(t0)) < abs(int(d.min())) + lim
            and abs(int(t1)) < abs(int(d.max())) + lim]
    if span < 30 or not wins:
        return np.nan, np.nan, 0
    obs_mask = _labels_from_windows(d, wins)
    ok = np.isfinite(y)
    if (obs_mask & ok).sum() < 3 or ((~obs_mask) & ok).sum() < 5:
        return np.nan, np.nan, 0
    obs = float(np.median(y[obs_mask & ok]) - np.median(y[~obs_mask & ok]))

    #  Map observation dates to circular coordinates from the record start.
    #  Shifting observations instead of windows handles all shifts in one
    #  (n_perm × n_obs) matrix.
    rel = (d - int(d.min())) % span
    rng = np.random.default_rng(seed)
    shifts = rng.integers(span, size=int(n_perm))
    M = np.zeros((len(shifts), len(d)), bool)
    for t0, t1 in wins:
        r0 = (t0 - int(d.min())) % span
        r1 = (t1 - int(d.min())) % span
        pos = (rel[None, :] - shifts[:, None]) % span
        M |= ((pos >= r0) & (pos <= r1)) if r0 <= r1 else ((pos >= r0) | (pos <= r1))
    diff, _, _ = _masked_median_diff(np.broadcast_to(y, M.shape), M)
    good = np.isfinite(diff)
    used = int(good.sum())
    if not used:
        return obs, np.nan, 0
    p = (np.sum(np.abs(diff[good]) >= abs(obs)) + 1) / (used + 1)
    return obs, float(p), used


def block_bootstrap_ci(dates: pd.Series, y: np.ndarray, mask: np.ndarray,
                       block_days=90, n_boot=2000, seed=SEED):
    """95 % interval of the contrast from resampling contiguous time blocks."""
    d = dates.to_numpy("datetime64[D]").astype("int64")
    ok = np.isfinite(y)
    if (mask & ok).sum() < 3 or ((~mask) & ok).sum() < 5:
        return np.nan, np.nan
    blk = (d - d.min()) // int(block_days)
    ids, inv = np.unique(blk, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ids))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(len(ids), size=len(ids))
        idx = np.concatenate([members[p] for p in pick])
        yy, mm, oo = y[idx], mask[idx], ok[idx]
        b, a = yy[mm & oo], yy[~mm & oo]
        if len(b) < 3 or len(a) < 5:
            continue
        out.append(np.median(b) - np.median(a))
    if len(out) < 100:
        return np.nan, np.nan
    return (round(float(np.percentile(out, 2.5)), 3),
            round(float(np.percentile(out, 97.5)), 3))


# ---------------------------------------------------------------------------
#  2. Activity definitions
# ---------------------------------------------------------------------------
def insitu_heating_events(site_key: str, rise_K=3.0, window_d=60):
    """Heating intervals defined by in-situ lake temperature, as a list of windows.

    On a daily grid, days on which the temperature rose by more than `rise_K`
    over the previous `window_d` days are joined into runs; runs less than
    30 days apart are merged.

    This definition is circular (lake temperature defines the periods that
    satellite lake temperature is tested against), so it serves only as a
    validation anchor: it asks whether the satellite can see real heating,
    not whether it detects unrest.
    """
    from pipeline.insitu_validate import _insitu_daily
    ins = _insitu_daily(site_key)
    if len(ins) < 100:
        return []
    grid = ins.resample("D").mean().interpolate(limit=30)
    rise = grid - grid.shift(window_d)
    hot = (rise > rise_K).fillna(False)
    out, run = [], None
    for t, v in hot.items():
        if v and run is None:
            run = t
        elif not v and run is not None:
            out.append((run, t - pd.Timedelta(days=1)))
            run = None
    if run is not None:
        out.append((run, hot.index[-1]))
    #  Runs within 30 days of each other are treated as one heating episode.
    merged = []
    for s, e in out:
        if merged and (s - merged[-1][1]).days <= 30:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [dict(date=str(s.date()), start=str(s.date()), end=str(e.date()),
                 label="in-situ heating", severity="unrest", kind="insitu_heating",
                 source="GeoNet in-situ", verified=True,
                 episode_id=f"insitu-{s.date()}")
            for s, e in merged if (e - s).days >= 7]


def definitions(site_key: str):
    """Build every activity definition available for the site."""
    site = get_site(site_key)
    ke = list(site.known_events or [])
    out = []
    ih = insitu_heating_events(site_key)
    if ih:
        out.append(dict(name="insitu_heating", family="insitu", buffer=0,
                        events=ih))
    for w in WINDOWS:
        out.append(dict(name=f"eruption_pm{w}d", family="eruption", buffer=w,
                        events=[dict(d, kind="eruption") for d in ke]))
    #  Optional activity catalogue (not required). Events with a known end date
    #  and events with only a known onset are kept apart:
    #  `catalogue_primary` uses only events whose return to background is
    #  reported, so both window bounds come from the source.
    #  `catalogue_onset` also includes onset-only events with a fixed window
    #  (onset + ONSET_WINDOW_D); that window is an analysis choice, not a claim
    #  about event duration.
    for lvl, nm in (("include", "catalogue_primary"),
                    ("include+uncertain", "catalogue_primary_incl_uncertain")):
        ce = TE.catalogue_unrest_events(site_key, lvl, require_end_known=True)
        if ce:
            out.append(dict(name=nm, family="catalogue", buffer=0, events=ce))
    for lvl, nm in (("include", "catalogue_onset"),
                    ("include+uncertain", "catalogue_onset_incl_uncertain")):
        ce = TE.catalogue_unrest_events(
            site_key, lvl, onset_window_days=TE.ONSET_WINDOW_D)
        if ce:
            out.append(dict(name=nm, family="catalogue_onset", buffer=0,
                            events=ce))
    #  Eruptions plus catalogue unrest, both treated as active.
    ce = TE.catalogue_unrest_events(site_key, "include")
    if ce:
        out.append(dict(name="eruption_pm180d_plus_catalogue", family="combined",
                        buffer=0,
                        events=[dict(d, kind="eruption",
                                     start=str((pd.Timestamp(str(d.get('date'))) -
                                                pd.Timedelta(days=180)).date()),
                                     end=str((pd.Timestamp(str(d.get('end') or d.get('date'))) +
                                              pd.Timedelta(days=180)).date()))
                                for d in ke] + ce))
    return out


