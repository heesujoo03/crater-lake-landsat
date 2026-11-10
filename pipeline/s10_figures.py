# -*- coding: utf-8 -*-
"""Plotting functions for the post-eruption relative-temperature response of the study lakes.

  fig_timeseries           relative-temperature records with eruption onsets
  fig_per_eruption         per-onset contrasts against the same-season placebo distribution
  fig_epoch                superposed-epoch response curves
  fig_ruapehu_validation   Ruapehu lake thermometer versus satellite
  fig_sensitivity          per-onset contrasts for four estimators

The functions read the s10_post_eruption records (per_eruption_delta_paper.csv,
epoch_response_paper.csv, site_meta_paper.json, insitu_before_after.csv).
analysis/figures.py re-executes them with revised labels and layout and writes the
final figures; the module-level `_save` writes to figures/post_eruption/ with a
copy in the stage-10 check folder.

Colour encodes site identity (categorical, fixed order); each site has its own
panel, so the panel title identifies the site and colour is a secondary cue.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import s10_post_eruption as P
from pipeline.config import settings as S
from pipeline.config import thermal_episodes as TE

plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "font.size": 9})

SITES4 = ("taal", "santa_ana", "ruapehu")   # the three study lakes
NAME = {"taal": "Taal", "ruapehu": "Ruapehu", "santa_ana": "Santa Ana",
        }
#  categorical palette in fixed order; panel titles and direct labels back up colour
COLOR = {"taal": "#2a78d6", "ruapehu": "#eb6834", "santa_ana": "#c2415a",
         }
GREY = "#9a9994"
INK2 = "#52514e"
REC = S.records_dir("10")
OUT = S.ROOT / "figures" / "post_eruption"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=250, bbox_inches="tight")
    plt.close(fig)
    dst = S.check_dir("10") / p.name
    shutil.copy2(p, dst)
    print(f"    · {p.name}")
    return p


#  Per-site estimator and water-fraction screen used for the figures, written by
#  s10_post_eruption as the *_paper records.
def _meta_paper():
    return pd.read_json(REC / "site_meta_paper.json").set_index("site")


def _est(site):
    return _meta_paper().loc[site, "estimator_used"]


def _fmin(site):
    return float(_meta_paper().loc[site, "f_min"])


def is_detected(r):
    """Detection rule: the event contrast exceeds every same-season placebo year
    (rank 1 of n + 1) and at least MIN_YEARS placebo years are available."""
    n = r.get("placebo_n_years", np.nan)
    p = r.get("p_placebo_year", np.nan)
    return bool(np.isfinite(n) and n >= P.MIN_YEARS and np.isfinite(p)
                and abs(p - 1.0 / (n + 1)) < 2e-4)


# ---------------------------------------------------------------------------
#  Figure: relative-temperature time series
# ---------------------------------------------------------------------------
def fig_timeseries():
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    for ax, k in zip(axes, SITES4):
        df, ke, _ = P.load(k, _fmin(k))
        col = _est(k)
        y = pd.to_numeric(df[col], errors="coerce")
        #  break the line across gaps longer than 400 days so no data are implied
        gap = df["date"].diff().dt.days.fillna(0).to_numpy() > 400
        xs, ys = [], []
        for t, v, g in zip(df["date"], y, gap):
            if g:
                xs.append(t); ys.append(np.nan)
            xs.append(t); ys.append(v)
        ax.plot(xs, ys, color=COLOR[k], lw=0.8, alpha=0.55, zorder=2)
        ax.plot(df["date"], y, "o", ms=3.2, mfc=COLOR[k], mec="white", mew=0.4,
                zorder=3)
        for e in TE.LAKE_ABSENT.get(k, []):
            ax.axvspan(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]),
                       color="#d9d8d2", alpha=0.8, lw=0, zorder=0)
        ev_all = pd.read_csv(REC / "per_eruption_delta_paper.csv")
        det = ev_all[(ev_all["site"] == k) & (ev_all["estimator"] == col)
                     & ev_all["window_d"].isin([P.PRIMARY_W, 365])
                     & ev_all["delta"].notna()]
        hit = {r["onset"] for _, r in det.iterrows() if is_detected(r)}
        for o in P.onsets(ke, P.PRIMARY_W):
            t = o["t0"]
            if t < df["date"].min() - pd.Timedelta(days=200):
                continue
            is_hit = str(t.date()) in hit
            ax.axvline(t, color=("#0b0b0b" if is_hit else GREY),
                       lw=(1.1 if is_hit else 0.7), ls=("-" if is_hit else "--"),
                       zorder=1)
        ax.axhline(0, color=INK2, lw=0.6)
        ttl = f"({chr(97 + SITES4.index(k))}) {NAME[k]}"
        if col != P.PRIMARY or abs(_fmin(k) - P.F_MIN) > 1e-6:
            ttl += ("  - relaxed screen (f >= 0.50), synthetic control without "
                    "weather normalisation")
        ax.set_title(ttl, loc="left", fontsize=9.5)
        ax.set_ylabel("Relative temperature (K)")
    axes[-1].set_xlabel("Year")
    #  legend explains line styles only; panel titles identify the sites
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="#0b0b0b", lw=1.1, label="Eruption with a detected change"),
         Line2D([], [], color=GREY, lw=0.7, ls="--", label="Other eruption onset"),
         Patch(color="#d9d8d2", label="Lake absent (documented)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, "fig_timeseries")


# ---------------------------------------------------------------------------
#  Figure: per-onset contrasts against the placebo distribution
# ---------------------------------------------------------------------------
def fig_per_eruption():
    ev = pd.read_csv(REC / "per_eruption_delta_paper.csv")
    rows = []
    for k in list(SITES4):
        try:
            col = _est(k)
        except KeyError:
            continue
        for w in (P.PRIMARY_W, 365):
            e = ev[(ev["site"] == k) & (ev["window_d"] == w) & (ev["estimator"] == col)
                   & ev["delta"].notna()]
            for _, r in e.iterrows():
                rows.append(dict(r.to_dict(), w=w))
    d = pd.DataFrame(rows)
    keys = (d[["site", "onset"]].drop_duplicates()
            .assign(o=lambda x: x["site"].map({s: i for i, s in
                                              enumerate(list(SITES4))}))
            .sort_values(["o", "onset"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.34 * len(keys) + 1.6), sharey=True)
    ylab = [f"{NAME[s]}  {o}" for s, o in zip(keys["site"], keys["onset"])]
    ypos = {(s, o): i for i, (s, o) in enumerate(zip(keys["site"], keys["onset"]))}
    for ax, w in zip(axes, (P.PRIMARY_W, 365)):
        x = d[d["w"] == w]
        for _, r in x.iterrows():
            yy = ypos[(r["site"], r["onset"])]
            if np.isfinite(r["placebo_q025"]):
                if abs(r["placebo_q975"] - r["placebo_q025"]) < 1e-6:
                    #  single placebo value: draw a tick instead of a zero-width bar
                    ax.plot(r["placebo_q025"], yy, "|", ms=16, mew=3, color="#9a9994",
                            zorder=1)
                else:
                    ax.plot([r["placebo_q025"], r["placebo_q975"]], [yy, yy],
                            color="#cfcec8", lw=6, solid_capstyle="round", zorder=1)
            c = COLOR.get(r["site"], GREY)
            sig = is_detected(r)
            ax.plot(r["delta"], yy, "o", ms=7, mfc=(c if sig else "white"), mec=c,
                    mew=1.6, zorder=3)
            if sig:
                ax.annotate(f"p={r['p_placebo_year']:.3f}", (r["delta"], yy),
                            xytext=(6, -3), textcoords="offset points",
                            fontsize=7, color=INK2)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_title(f"({'a' if w == P.PRIMARY_W else 'b'}) {w}-day windows",
                     loc="left", fontsize=9.5)
        ax.set_xlabel("After minus before, relative temperature (K)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ylab)))
    axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#cfcec8", lw=6, label="Same-season placebo, 2.5-97.5 %"),
         Line2D([], [], ls="none", marker="o", ms=7, mfc=INK2, mec=INK2,
                label="Larger than every placebo year"),
         Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec=INK2, mew=1.6,
                label="Not resolved")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, "fig_per_eruption")


# ---------------------------------------------------------------------------
#  Figure: superposed-epoch response curves
# ---------------------------------------------------------------------------
def fig_epoch():
    ep = pd.read_csv(REC / "epoch_response_paper.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharex=True)
    for ax, k in zip(axes.ravel(), SITES4):
        e = ep[(ep["site"] == k)]
        pts = e[e["n_scenes"] == -1]
        bins = e[e["n_scenes"] >= 0]
        onsets = sorted(pts["onset"].unique())
        #  onsets detected in the 180- or 365-day window in site colour, others in grey
        ev = pd.read_csv(REC / "per_eruption_delta_paper.csv")
        sub = ev[(ev["site"] == k) & (ev["estimator"] == _est(k))
                 & ev["window_d"].isin([P.PRIMARY_W, 365]) & ev["delta"].notna()]
        hit = {r["onset"] for _, r in sub.iterrows() if is_detected(r)}
        last_y = []
        for o in onsets:
            det = o in hit
            q = pts[pts["onset"] == o]
            ax.plot(q["bin_lo_d"], q["rel_K"], "o", ms=2.6,
                    color=(COLOR[k] if det else GREY), alpha=(0.45 if det else 0.2),
                    mew=0, zorder=(3 if det else 1))
            b = bins[(bins["onset"] == o) & bins["rel_K"].notna()]
            if len(b):
                xm = (b["bin_lo_d"] + b["bin_hi_d"]) / 2
                ax.plot(xm, b["rel_K"], "-", color=(COLOR[k] if det else GREY),
                        lw=(1.8 if det else 0.8), alpha=(1.0 if det else 0.6),
                        zorder=(4 if det else 2))
                if det:
                    yv = float(b["rel_K"].iloc[-1])
                    #  offset end labels that would overlap
                    dy = 0
                    for yy in last_y:
                        if abs(yy - yv) < 0.06 * (np.nanmax(e["rel_K"]) - np.nanmin(e["rel_K"])):
                            dy -= 9
                    last_y.append(yv)
                    ax.annotate(o[:7], (xm.iloc[-1], yv), xytext=(3, dy),
                                textcoords="offset points", fontsize=7,
                                color="#0b0b0b")
        ax.axvline(0, color="#0b0b0b", lw=0.9)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvspan(0, P.PRIMARY_W, color="#f0efec", lw=0, zorder=0)
        ax.set_title(f"({chr(97 + SITES4.index(k))}) {NAME[k]}", loc="left",
                     fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("Days since eruption onset")
    axes[0].set_ylabel("Relative temperature minus\npre-eruption year median (K)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#0b0b0b", lw=1.8, label="Eruption with a detected change"),
         Line2D([], [], color=GREY, lw=0.8, label="Other eruption")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, "fig_epoch_response")


# ---------------------------------------------------------------------------
#  Figure: Ruapehu, lake thermometer versus satellite
# ---------------------------------------------------------------------------
def fig_ruapehu_validation():
    from pipeline.insitu_validate import _insitu_daily
    ins = _insitu_daily("ruapehu")
    ins.index = ins.index.tz_localize(None)
    df, ke, _ = P.load("ruapehu")
    a, b = pd.Timestamp("2006-01-01"), pd.Timestamp("2008-07-01")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                             gridspec_kw=dict(width_ratios=[2.1, 1]))
    ax = axes[0]
    s = ins[(ins.index >= a) & (ins.index <= b)]
    ax.plot(s.index, s.values, "s-", color="#0b0b0b", ms=3, lw=0.8,
            label="Lake thermometer (GeoNet)")
    q = df[(df["date"] >= a) & (df["date"] <= b)]
    ax.plot(q["date"], q["T_active_K"] - 273.15, "o", ms=4.5, mfc=COLOR["ruapehu"],
            mec="white", mew=0.5, label="Landsat lake surface temperature")
    for o in P.onsets(ke, P.PRIMARY_W):
        t = o["t0"]
        if a <= t <= b:
            ax.axvline(t, color="#0b0b0b", lw=1.0)
            ax.axvspan(t - pd.Timedelta(days=180), t, color="#f0efec", lw=0, zorder=0)
            ax.text(t, ax.get_ylim()[1] if False else 38, f" eruption\n {t.date()}",
                    fontsize=7, va="top", color=INK2)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("(a) Lake temperature around the 2006 and 2007 eruptions",
                 loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_ylim(5, 40)
    ic = pd.read_csv(REC / "insitu_before_after.csv")
    ic = ic.dropna(subset=["insitu_delta_K", "sat_delta_rel_K"])
    ax = axes[1]
    for o, mk in zip(sorted(ic["onset"].unique()), ("o", "D")):
        z = ic[ic["onset"] == o]
        ax.plot(z["insitu_delta_K"], z["sat_delta_rel_K"], mk, ms=6,
                mfc=COLOR["ruapehu"], mec="#0b0b0b", mew=0.6, ls="none",
                label=f"eruption {o}")
        seen = []
        for _, r in z.iterrows():
            #  shift overlapping labels downward
            dy = 2
            for xv, yv in seen:
                if abs(xv - r["insitu_delta_K"]) < 1.5 and abs(yv - r["sat_delta_rel_K"]) < 1.5:
                    dy -= 9
            seen.append((r["insitu_delta_K"], r["sat_delta_rel_K"]))
            ax.annotate(f"{int(r['window_d'])} d", (r["insitu_delta_K"], r["sat_delta_rel_K"]),
                        xytext=(4, dy), textcoords="offset points", fontsize=6.5,
                        color=INK2)
    lim = [min(-3, ic[["insitu_delta_K", "sat_delta_rel_K"]].min().min() - 1),
           ic[["insitu_delta_K", "sat_delta_rel_K"]].max().max() + 2]
    ax.plot(lim, lim, color=INK2, lw=0.7, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Thermometer: after minus before (K)")
    ax.set_ylabel("Satellite relative temperature:\nafter minus before (K)")
    ax.set_title("(b) Same windows, both instruments", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    return _save(fig, "fig_ruapehu_validation")


def fig_sensitivity():
    """Estimator sensitivity: per-onset contrasts for four estimators (180 and 365 d)."""
    ev = pd.read_csv(REC / "per_eruption_delta_paper.csv")
    est = [("dT_syn_weather", "Weather-normalised\nsynthetic control"),
           ("dT_syn", "Synthetic control"),
           ("dT_simple", "Lake minus\nreference lake or ring"),
           ("T_active_K", "Lake surface\ntemperature")]
    mk = {"dT_syn_weather": "o", "dT_syn": "s", "dT_simple": "^", "T_active_K": "D"}
    e = ev[ev["site"].isin(SITES4) & ev["window_d"].isin([P.PRIMARY_W, 365])
           & ev["delta"].notna() & ev["estimator"].isin([c for c, _ in est])]
    keys = (e[["site", "onset"]].drop_duplicates()
            .assign(o=lambda x: x["site"].map({s: i for i, s in enumerate(SITES4)}))
            .sort_values(["o", "onset"]))
    ypos = {(s, o): i for i, (s, o) in enumerate(zip(keys["site"], keys["onset"]))}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.33 * len(keys) + 1.8), sharey=True)
    for ax, w in zip(axes, (P.PRIMARY_W, 365)):
        for j, (c, lab) in enumerate(est):
            x = e[(e["window_d"] == w) & (e["estimator"] == c)]
            for _, r in x.iterrows():
                yy = ypos[(r["site"], r["onset"])] + (j - 1.5) * 0.16
                ax.plot(r["delta"], yy, mk[c], ms=5.5, mfc=COLOR[r["site"]],
                        mec="#0b0b0b", mew=0.4, ls="none",
                        label=lab.replace("\n", " ") if (r["site"], r["onset"]) ==
                        tuple(keys.iloc[0][["site", "onset"]]) or True else None)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_title(f"({'a' if w == P.PRIMARY_W else 'b'}) {w}-day windows",
                     loc="left", fontsize=9.5)
        ax.set_xlabel("After minus before (K)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([f"{NAME[s]}  {o}" for s, o in
                             zip(keys["site"], keys["onset"])], fontsize=8)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="none", marker=mk[c], ms=6, mfc=GREY, mec="#0b0b0b",
                mew=0.4, label=lab.replace("\n", " ")) for c, lab in est]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, "figS3_estimator_sensitivity")


