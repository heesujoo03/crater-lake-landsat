# -*- coding: utf-8 -*-
"""Crater-lake heating intervals, lake-absent periods and optional activity intervals.

EPISODES
    Heating intervals recorded by ground observations, independent of the
    eruption list and of the satellite analysis. For the study lakes, the
    Ruapehu entries list crater-lake heating cycles without eruptions.
LAKE_ABSENT
    Periods when a lake had dried out; the "surface temperature" then belongs
    to the dry crater floor. `absent_mask` turns them into a boolean mask that
    s9b_event_stats.frame uses to drop those scenes.
catalogue_unrest_events
    Reads optional activity intervals from a precomputed table in the stage-9
    records folder; s9b_event_stats.definitions adds them as extra activity
    definitions when the table exists and skips them otherwise.

Sources of the Ruapehu entries (tags used in ``source``)
---------------------------------------------------------
[TB08]  Trunk & Bernard (2008) J. Volcanol. Geotherm. Res. 178, 259–270 —
        ASTER crater-lake temperature time series compared with ground observations.
[GNS]   GeoNet (New Zealand) Volcanic Activity Bulletins — Ruapehu Crater Lake
        heating cycles.
"""

from __future__ import annotations

#  Intervals are inclusive [start, end] and transcribe the range stated by the
#  source without extension. Coarse dates (e.g. "mid-2004") use the whole month.
EPISODES: dict[str, list[dict]] = {

    # -- Ruapehu Crater Lake ---------------------------------------------
    #  TB08 §5.2 reports several heating cycles without eruptions after 1999.
    "ruapehu": [
        dict(start="2000-04-01", end="2000-07-31",
             label="onset of moderate volcanic tremor", source="TB08 §5.2"),
        dict(start="2002-09-01", end="2003-02-28",
             label="tremor and volcanic earthquakes → crater lake 40 ℃ in 2003-01",
             source="TB08 §5.2"),
        dict(start="2003-03-01", end="2003-06-15",
             label="renewed heating to 41.6 ℃ with tremor and volcanic earthquakes",
             source="TB08 §5.2"),
        dict(start="2016-03-01", end="2016-08-31",
             label="crater lake reached 46 ℃ (no eruption)", source="GNS"),
        dict(start="2020-02-01", end="2021-02-28",
             label="heating from 2020-02 → 43 ℃ by late 2020-12", source="GNS"),
        dict(start="2021-03-01", end="2021-07-31",
             label="41 ℃ peak in 2021-04", source="GNS"),
        dict(start="2022-02-01", end="2022-06-30",
             label="31 ℃ in 2022-03 · strong tremor · alert level 2", source="GNS"),
    ],


}


#  -- Periods without a lake -------------------------------------------------
#  The "surface temperature" in these intervals is the dry crater floor; they
#  are excluded from analysis entirely.
LAKE_ABSENT: dict[str, list[dict]] = {

}


#  -- Optional activity intervals ---------------------------------------------
#  ONSET_WINDOW_D is the window [days] attached to events with a known onset but
#  no reported end. It is an analysis parameter, not a claim about the event's
#  duration; analyses that use it are labelled as onset-based.
ONSET_WINDOW_D = 90


def catalogue_unrest_events(site_key, level: str = "include",
                            require_end_known: bool = False,
                            onset_window_days: int | None = None):
    """Read optional activity intervals from the precomputed stage-9 table.

    level="include"            intervals classified as include
    level="include+uncertain"  also undecided intervals

    require_end_known=True
        Return only events with a reported return to normal. Quiet-vs-active
        comparisons should use these, since attaching a chosen window to events
        of unknown end lets the window drive the result.

    onset_window_days
        If given, the window runs from the onset for this many days (the end
        date is ignored). Used for onset-based analyses that include events with
        unknown end; the window length is an analysis choice.

    Returns None if the table does not exist; the caller decides on the
    fallback.
    """
    import pandas as pd
    from pipeline.config import settings as S

    p = (S.records_dir(9, create=False) / "independent_unrest_catalogue.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    want = {"include"} | ({"uncertain"} if "uncertain" in level else set())
    key = site_key
    d = d[(d["volcano"] == key) & d["decision"].isin(want)]
    if require_end_known:
        if "end_known" not in d.columns:
            return []
        d = d[d["end_known"].astype(str) == "yes"]

    out = []
    for _, r in d.iterrows():
        st = str(r["window_start"])
        en = str(r["window_end"])
        if onset_window_days:
            en = str((pd.Timestamp(st)
                      + pd.Timedelta(days=int(onset_window_days))).date())
        out.append(dict(date=st, start=st, end=en,
                        label=str(r["evidence_sentence"])[:80],
                        severity="unrest", source=r["source_citation"],
                        verified=True, kind="unrest", episode_id=r["episode_id"],
                        decision=r["decision"],
                        end_known=str(r.get("end_known", "unknown")),
                        onset_precision=str(r.get("onset_precision", "unknown"))))
    return out


def absent_mask(site_key, t):
    """Boolean mask, True where the lake was absent (for exclusion from analysis)."""
    import numpy as np
    import pandas as pd
    tt = pd.Series(pd.to_datetime(t))
    if tt.dt.tz is not None:
        tt = tt.dt.tz_localize(None)
    m = np.zeros(len(tt), bool)
    for e in LAKE_ABSENT.get(site_key, []):
        m |= ((tt >= pd.Timestamp(e["start"]))
              & (tt <= pd.Timestamp(e["end"]))).to_numpy()
    return m
