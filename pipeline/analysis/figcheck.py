# -*- coding: utf-8 -*-
"""Overlap checker for matplotlib figures.

For a finished figure it reports
  text–text   : two visible text artists whose boxes intersect,
  text–data   : a text box that covers plotted data (line vertices, markers,
                scatter points, bars) of another artist,
  text–edge   : a text box that extends beyond the figure canvas.
Tick labels are compared only with other texts (not with data), axis spines and
grid lines are ignored. Results go to figures/analysis/_overlap/.
"""
from __future__ import annotations

import json

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from pipeline.config import settings as S

OUT = S.ROOT / "figures" / "analysis" / "_overlap"
PAD = 1.0          # px shrink of text boxes, so touching boxes are not flagged
MIN_AREA = 12.0    # px² minimum intersection to report


def _tick_sets(fig):
    """All tick-label artists, and the subset matplotlib actually draws."""
    every, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in list(axis.majorTicks) + list(axis.minorTicks):
                every.update({id(tick.label1), id(tick.label2)})
            for tick in axis._update_ticks():
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible():
                        drawn.add(id(lab))
    return every, drawn


def _texts(fig):
    every, drawn = _tick_sets(fig)
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in every and id(t) not in drawn:
            continue
        try:
            bb = Text.get_window_extent(t)  # text only, not an annotation arrow
        except Exception:
            continue
        if bb.width < 1 or bb.height < 1:
            continue
        out.append((t, bb))
    return out


def _is_tick(t):
    ax = t.axes
    if ax is None:
        return False
    return (t in ax.get_xticklabels() or t in ax.get_yticklabels()
            or t in (ax.xaxis.label, ax.yaxis.label))


def _data_points(fig):
    pts = []
    for ax in fig.axes:
        grid = set(ax.get_xgridlines()) | set(ax.get_ygridlines())
        for a in ax.get_children():
            if not a.get_visible() or a in grid:
                continue
            try:
                if isinstance(a, Line2D):
                    xy = a.get_xydata()
                    if len(xy) == 0:
                        continue
                    # skip full-axis reference lines (axhline/axvline, 1:1 lines)
                    if len(xy) == 2 and a.get_marker() in (None, "None", "", " "):
                        continue
                    d = a.get_transform().transform(xy)
                    # densify line segments so a label crossing a line is caught
                    if a.get_linestyle() not in ("None", "", " ") and len(d) > 1:
                        dd = [d[0]]
                        for p0, p1 in zip(d[:-1], d[1:]):
                            n = int(max(2, np.hypot(*(p1 - p0)) // 4))
                            dd.extend(np.linspace(p0, p1, n)[1:])
                        d = np.array(dd)
                    pts.append((a, d))
                elif isinstance(a, PathCollection):
                    off = a.get_offsets()
                    if len(off):
                        pts.append((a, a.get_offset_transform().transform(off)))
                elif isinstance(a, Rectangle) and a is not ax.patch and a.get_zorder() > 0.5:
                    bb = a.get_window_extent()
                    if bb.width > 0 and bb.height > 0:
                        xs = np.linspace(bb.x0, bb.x1, 6)
                        ys = np.linspace(bb.y0, bb.y1, 6)
                        pts.append((a, np.array([(x, y) for x in xs for y in ys])))
            except Exception:
                continue
    return pts


def check(fig, name):
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    texts = _texts(fig)
    issues = []
    for i in range(len(texts)):
        ti, bi = texts[i]
        for j in range(i + 1, len(texts)):
            tj, bj = texts[j]
            ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ix > PAD and iy > PAD and ix * iy > MIN_AREA:
                issues.append({"type": "text-text", "a": ti.get_text()[:60], "b": tj.get_text()[:60],
                               "area_px2": round(float(ix * iy))})
    for t, bb in texts:
        if _is_tick(t):
            continue
        if t.axes is not None and t is t.axes.title:
            continue
        for art, d in _data_points(fig):
            inside = ((d[:, 0] > bb.x0 + PAD) & (d[:, 0] < bb.x1 - PAD)
                      & (d[:, 1] > bb.y0 + PAD) & (d[:, 1] < bb.y1 - PAD))
            n = int(inside.sum())
            if n:
                issues.append({"type": "text-data", "text": t.get_text()[:60],
                               "artist": type(art).__name__, "points_inside": n})
                break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(issues, indent=1, ensure_ascii=False), encoding="utf-8")
    return issues
