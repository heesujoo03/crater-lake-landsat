# -*- coding: utf-8 -*-
"""Daily in-situ lake temperature for comparison with the satellite record.

`_insitu_daily` reads the converted GeoNet lake-temperature record
(`data/raw/ground_thermal/<site>/geonet_converted.csv`) and returns the daily
median across sensors. It supplies the ground series used to define the
Ruapehu heating episodes (s9b_event_stats.insitu_heating_events) and the
before/after ground comparison (s10_post_eruption.insitu_check).

RISE_WINDOW_D and RISE_K set the heating-episode rule: a rise of more than
RISE_K over the preceding RISE_WINDOW_D days, a rate criterion that avoids
confounding with season.
"""

from __future__ import annotations


import pandas as pd

from pipeline.config import settings as S

#  Heating period: rise of more than RISE_K over the preceding RISE_WINDOW_D days
#  (a rate criterion rather than absolute level, which would mix in season).
RISE_WINDOW_D = 60
RISE_K = 3.0


def _insitu_daily(site_key: str) -> pd.Series:
    """Daily median in-situ lake temperature.

    The median across sensors (different depths/positions) is robust to
    outliers from a faulty sensor.
    """
    p = (S.ROOT / "data" / "raw" / "ground_thermal" / site_key
         / "geonet_converted.csv")
    if not p.exists():
        return pd.Series(dtype=float)
    g = pd.read_csv(p)
    g = g[g["variable"].eq("temperature")]
    if g.empty:
        return pd.Series(dtype=float)
    t = pd.to_datetime(g["datetime"], utc=True, errors="coerce", format="mixed")
    v = pd.to_numeric(g["value"], errors="coerce")
    d = pd.DataFrame({"t": t.dt.floor("D"), "v": v}).dropna()
    return d.groupby("t")["v"].median().sort_index()


