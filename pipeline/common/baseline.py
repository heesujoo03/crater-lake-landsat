# -*- coding: utf-8 -*-
"""Single definition of the quiet (baseline) period used throughout the package.

Every module that needs a baseline imports it from here so that the fitting
sample of the reference and weather models and the activity classification
always use the same rule.

Why the post-eruption interval is not treated as quiet
------------------------------------------------------
If every observation outside the event windows were counted as quiet, the
years after the last listed eruption would enter the baseline even when the
lake has not returned to its pre-eruption regime. That inflates the quiet-period
spread, and because the spread sets the detection limit, it degrades every
detection threshold derived from it.

Classes
-------
    quiet        before the first known event and outside every event window
    unrest       inside an event window (event ± exclude_days, or event interval ± buffer)
    post_unrest  after the first event but outside event windows — may look calm
                 but is not the pre-event regime
    unknown      no basis for a decision (empty event list)

`in_baseline` is True only for ``quiet`` (with a documented fallback, see below).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pandas as pd

QUIET = "quiet"
UNREST = "unrest"
POST_UNREST = "post_unrest"
UNKNOWN = "unknown"


def event_windows(events: Iterable[dict], exclude_days: int) -> List[Tuple]:
    """Return (t0, t1, label) windows, padded by ``exclude_days`` on both sides.

    Events given as an interval (``start``/``end``) use the full interval;
    point events (``date``) use the date. Eruptions can last months, and
    recording them as a single day would let the observations in between be
    classified as quiet.
    """
    out = []
    w = pd.Timedelta(days=int(exclude_days))
    for ev in (events or []):
        start = ev.get("start") or ev.get("date")
        end = ev.get("end") or start
        if start is None:
            continue
        t0 = pd.Timestamp(start, tz="UTC") - w
        t1 = pd.Timestamp(end, tz="UTC") + w
        out.append((t0, t1, ev.get("label", "")))
    return out


def first_event_time(events: Iterable[dict]) -> Optional[pd.Timestamp]:
    """Earliest start/date among ``events`` (UTC), or None if there is none."""
    ts = []
    for ev in (events or []):
        s = ev.get("start") or ev.get("date")
        if s is not None:
            ts.append(pd.Timestamp(s, tz="UTC"))
    return min(ts) if ts else None


def classify(datetimes, events, exclude_days: int) -> pd.Series:
    """Label each observation time as quiet / unrest / post_unrest / unknown."""
    dt = pd.to_datetime(pd.Series(datetimes), utc=True)
    if not events:
        return pd.Series([UNKNOWN] * len(dt), index=dt.index)

    cls = pd.Series(POST_UNREST, index=dt.index, dtype=object)
    first = first_event_time(events)
    cls[dt < first] = QUIET
    for t0, t1, _ in event_windows(events, exclude_days):
        cls[(dt >= t0) & (dt <= t1)] = UNREST
    return cls


def in_baseline(datetimes, events, exclude_days: int,
                qc_flag=None, min_n: int = 0, report: dict = None) -> pd.Series:
    """Boolean mask of observations usable for baseline statistics.

    The primary rule is ``quiet`` (before the first event). If that leaves
    fewer than ``min_n`` samples (e.g. when the first eruption falls early in
    the record), the rule is widened to everything outside the event windows.
    The rule actually used and the number of post-unrest samples admitted are
    written to ``report`` so the widening, which increases the spread, is
    always visible.
    """
    cls = classify(datetimes, events, exclude_days)
    qc_ok = pd.Series(True, index=cls.index)
    if qc_flag is not None:
        q = pd.Series(qc_flag).fillna("").astype(str)
        qc_ok = pd.Series(q.to_numpy() == "ok", index=cls.index)

    strict = (cls == QUIET) & qc_ok
    used, ok = "quiet_pre_first_event", strict
    if min_n and int(strict.sum()) < min_n:
        wide = (cls != UNREST) & qc_ok
        if int(wide.sum()) > int(strict.sum()):
            used, ok = "outside_event_windows", wide
    if report is not None:
        report.update(baseline_rule=used,
                      n_strict=int(strict.sum()),
                      n_used=int(ok.sum()),
                      n_post_unrest_included=int(((cls == POST_UNREST) & ok).sum()),
                      widened=(used != "quiet_pre_first_event"))
    return ok


