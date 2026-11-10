# -*- coding: utf-8 -*-
"""Render the final figures, with labels that state the decision rule used
(maximum-placebo exceedance).

The plotting functions of s10_figures, s10_figures_v3, s10_site_maps and s10_magma
are reused: `_patched` applies exact (old, new) source replacements to labels and
layout and re-executes the function. Output goes to figures/analysis/ (PNG at
250 dpi, plus PDF and 600-dpi PNG in figures/analysis/submission/); the figures
written by the source modules are left unchanged.

    python -m pipeline.analysis.figures
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import s10_figures as F
from pipeline import s10_figures_v3 as F3
from pipeline import s10_magma as M
from pipeline import s10_post_eruption as P
from pipeline.config import settings as S

OUT = S.ROOT / "figures" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

EXC = "Maximum-placebo exceedance"
NOEXC = "Tested without exceedance"


def _save(fig, name):
    from pipeline.analysis.figcheck import check
    iss = check(fig, name)
    if iss:
        print(f"   ! {name}: {len(iss)} overlap issue(s)")
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=250, bbox_inches="tight")
    # separate artwork files: vector PDF and 600-dpi PNG
    sub = OUT / "submission"; sub.mkdir(exist_ok=True)
    fig.savefig(sub / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(sub / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("   ·", p.name)
    return p


def _patched(func, repl, extra_globals=None):
    """Return a copy of `func` whose source has the (old, new) replacements applied."""
    src = inspect.getsource(func)
    for old, new in repl:
        if old not in src:
            raise ValueError(f"label not found in {func.__name__}: {old!r}")
        src = src.replace(old, new)
    g = dict(func.__globals__)
    g["_save"] = _save
    g["F"] = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F)
                                      if not k.startswith("__")})
    g["F"]._save = _save
    if extra_globals:
        g.update(extra_globals)
    ns = {}
    exec(compile(src, f"<patched {func.__name__}>", "exec"), g, ns)
    return ns[func.__name__]


# Geometry, variability and persistence
_tb = pd.read_csv(S.ROOT / "results/11_analysis/variability_timescale/tau_block_bootstrap.csv")
_tb = _tb[_tb.block_d.astype(str).eq("365")]
_TB = {r.site: [r.tau_16, r.tau_84] for r in _tb.itertuples()}

fig5 = _patched(F3.fig_detectability, [
    ('"(a) Measurement noise is predictable from geometry"',
     '"(a) Short-lag variability proxy and lake geometry"'),
    ('"Predicted from pixel count and water fraction (K)"',
     r'r"Geometry scaling, $\sigma_{\mathrm{pred}}$ (K)"'),
    ('"Measured from close scene pairs (K)"', r'r"Short-lag proxy, $\sigma_{\mathrm{pair}}$ (K)"'),
    ('"Estimated noise σ (K)"', r'r"Short-lag proxy $\sigma_{\mathrm{pair}}$ (K)"'),
    ('"(b) Rising means real change is entering"',
     '"(b) Proxy versus maximum pair separation"'),
    ('label="Measurement"', 'label=r"Short-lag component ($\sigma_{\mathrm{pair}}^2$)"'),
    ('label="Lake, whole record"', 'label="Residual, full record"'),
    ('label="Lake, quiet periods only"', 'label="Total, quiet segments"'),
    ('f"{r.meas_variance_share*100:.0f} % measurement"',
     'f"short-lag {r.meas_variance_share*100:.1f} %"'),
    ('"(c) How much of the variance is the instrument"',
     '"(c) Variance partition"'),
    ('"Predicted from mean depth, τ = ρch/K (days)"',
     r'r"Idealised thermal-response scale $\rho c h/K_s$ (days)"'),
    ('"Measured from the record (days)"',
     r'r"Structure-function timescale $\tau_{\mathrm{obs}}$ (days)"'),
    ('[r.tau_obs_lo_d, r.tau_obs_hi_d]', '_TB[r.site]'),
    ('xytext=(11, -3), textcoords="offset points", fontsize=9,\n                   color=F.INK2, va="center")',
     'xytext=(-10, 13), textcoords="offset points", fontsize=9,\n                   color=F.INK2, va="center", ha="right")'),
    ('c_ax.legend(fontsize=8, frameon=False)', 'c_ax.legend(fontsize=8, frameon=False, handlelength=1.6, handletextpad=1.6)'),
    ('"(d) How long a lake actually holds its state"',
     '"(d) Fitted persistence and thermal-response scale"'),
    ('if not np.isfinite(r.tau_obs_d):',
     'if not np.isfinite(r.tau_obs_d) or r.site == "taal":'),
    ('e.set_xlim(2, 300); e.set_ylim(1, 300)',
     'e.set_xlim(2, 300); e.set_ylim(1, 300)\n'
     '    e.text(0.97, 0.05, "Taal: not constrained", transform=e.transAxes, ha="right",'
     ' fontsize=8, color=F.INK2)'),
], extra_globals={"_TB": _TB})


# Relative-temperature records
fig6 = _patched(F.fig_timeseries, [
    ('label="Eruption with a detected change"', f'label="Onset with {EXC.lower()}"'),
    ('label="Other eruption onset"', 'label="Other catalogue onset"'),
    # documented Taal lake absence (Perttu et al., 2023; refilled by 11 March 2020)
    ('TE.LAKE_ABSENT.get(k, [])',
     '{"taal": [{"start": "2020-01-12", "end": "2020-03-11"}]}.get(k, [])'),
])

# Event contrasts against the placebo distribution
fig7 = _patched(F.fig_per_eruption, [
    ('label="Larger than every placebo year"', f'label="{EXC}"'),
    ('label="Not resolved")', f'label="{NOEXC}")'),
    ('color="#cfcec8", lw=6, solid_capstyle="round", zorder=1)', 'color="#cfcec8", lw=6, solid_capstyle="butt", zorder=1)'),
    ('label="Same-season placebo, 2.5-97.5 %"', 'label="Same-season placebo, 2.5–97.5 %"'),
    ('ax.grid(axis="y", visible=False)', 'ax.grid(axis="y", visible=False); ax.margins(x=0.1)'),
    ('xytext=(6, -3), textcoords="offset points",\n                            fontsize=7, color=INK2)',
     'xytext=(0, 7), textcoords="offset points", ha="center",\n                            fontsize=7, color=INK2)'),
])

# Sensitivity to the analysed quantity
fig8 = _patched(F.fig_sensitivity, [
    ('"Weather-normalised\\nsynthetic control"', '"Relative temperature\\n(primary)"'),
    ('("dT_syn", "Synthetic control")', '("dT_syn", "Reference-relative, without\\nmeteorological adjustment")'),
    ('"Lake minus\\nreference lake or ring"', '"Lake minus\\nreference water or ring"'),
])

# Superposed-epoch response
fig10 = _patched(F.fig_epoch, [
    ('label="Eruption with a detected change"', f'label="Onset with {EXC.lower()} (site colour)"'),
    ('label="Other eruption"', 'label="Other catalogue onset"'),
    ('ax.plot(xm, b["rel_K"], "-", color=(COLOR[k] if det else GREY),',
     'ax.plot(xm, b["rel_K"], _LS.get(o, "-"), color=(COLOR[k] if det else GREY),'),
    ('ax.annotate(o[:7], (xm.iloc[-1], yv), xytext=(3, dy),\n                                textcoords="offset points", fontsize=7,',
     'ax.annotate(o[:7], (xm.iloc[-1], yv), xytext=(1150, yv + dy * 0.012 * (np.nanmax(e["rel_K"]) - np.nanmin(e["rel_K"]))),\n                                textcoords="data", fontsize=7, va="center",\n                                arrowprops=dict(arrowstyle="-", lw=0.5, color="#9a9994"),'),
    ('ax.set_xlabel("Days since eruption onset")', 'ax.set_xlabel("Days since catalogue onset")'),
    ('"Relative temperature minus\\npre-eruption year median (K)"', '"Relative temperature minus\\npreceding-year median (K)"'),
    ('Line2D([], [], color="#0b0b0b", lw=1.8, label=', 'Line2D([], [], color="#2a78d6", lw=1.8, label='),
    ('ax.axvline(0, color="#0b0b0b", lw=0.9)', 'ax.axvline(0, color="#0b0b0b", lw=0.9); ax.set_xlim(-450, 1330)'),
], extra_globals={"_LS": {"2020-01-12": "-", "2021-07-01": "--", "2022-10-05": "-.", "2024-04-12": ":"}})


# Source geometry and event contrasts
def fig11():
    rec = S.ROOT / "results" / "10_magma_geometry"
    geo = pd.read_csv(rec / "magma_geometry_lake_datum.csv")
    resp = pd.read_csv(rec / "response_vs_geometry.csv")
    f = _patched(M.fig_geometry, [
        ('"(a) Magma sources beneath each lake (literature)"',
         '"(a) Published source geometry below each lake"'),
        ('"(b) Post-eruption response of each lake"',
         '"(b) Event-window relative-temperature contrasts"'),
        ('label="Resolved"', f'label="{EXC}"'),
        ('label="Tested, not resolved"', f'label="{NOEXC}"'),
        ('"no depth\\npublished"', '"near-surface intrusion;\\nno numerical depth"'),
        ('label="Deep reservoir"', 'label="Deeper reservoir"'),
        ('            if sig:\n                # Annotate onset year', '            if True:\n                # Annotate onset year'),
    ])
    return f(geo, resp)


# Pre-onset state; (b) 180-day and (c) 365-day pre-onset windows
def fig9():
    DET = F3.DET
    st = pd.read_csv(DET / "pre_eruption_state.csv")
    st = st[st.site.isin(F.SITES4)].reset_index(drop=True)
    pr = pd.read_csv(F3.PRE / "per_eruption_pre.csv")
    pr = pr[pr.site.isin(F.SITES4) & pr.estimator.eq(P.PRIMARY) & pr.delta_pre.notna()]

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 5.0),
                           gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    a = ax[0]
    a.axhspan(0, 20, color="#2a78d6", alpha=0.10)
    a.axhspan(80, 100, color="#c2415a", alpha=0.10)
    a.text(0.01, 10, "lowest fifth", fontsize=8, color="#2a78d6",
           transform=a.get_yaxis_transform(), va="center")
    a.text(0.01, 90, "highest fifth", fontsize=8, color="#c2415a",
           transform=a.get_yaxis_transform(), va="center")
    lab, xs = [], []
    for i, r in enumerate(st.itertuples()):
        c = F.COLOR.get(r.site, F.GREY)
        a.plot(i, r.percentile_in_prior_record, "o", ms=10, color=c,
               mec="#0b0b0b", mew=0.8, zorder=3)
        dy = -15 if r.percentile_in_prior_record > 88 else 11
        a.annotate(f"n={int(r.n_pre)}", (i, r.percentile_in_prior_record),
                   xytext=(0, dy), textcoords="offset points", fontsize=7,
                   ha="center", color=F.INK2)
        lab.append(f"{F.NAME[r.site]}\n{r.onset[:7]}"); xs.append(i)
    a.set_xticks(xs); a.set_xticklabels(lab, fontsize=7.5)
    a.set_ylim(0, 100)
    a.set_ylabel("Percentile among earlier observations")
    a.set_title("(a) Pre-onset 180-day median, percentile", loc="left", fontsize=10)
    a.grid(axis="y", alpha=0.25, lw=0.6)

    for b, w, tag in ((ax[1], 180, "b"), (ax[2], 365, "c")):
        sub = pr[pr.window_d.eq(w)].copy()
        sub["_o"] = sub.site.map({"taal": 0, "santa_ana": 1, "ruapehu": 2})
        sub = sub.sort_values(["_o", "onset"]).reset_index(drop=True)
        for i, r in enumerate(sub.itertuples()):
            c = F.COLOR.get(r.site, F.GREY)
            if np.isfinite(r.placebo_q025) and np.isfinite(r.placebo_q975):
                b.plot([r.placebo_q025, r.placebo_q975], [i, i], "-",
                       color=F.GREY, lw=6, alpha=0.45, solid_capstyle="butt")
            tested = np.isfinite(r.p_placebo_year) and r.placebo_n_years >= P.MIN_YEARS
            sig = tested and abs(r.p_placebo_year - 1.0 / (r.placebo_n_years + 1)) < 2e-4
            if tested:
                b.plot(r.delta_pre, i, "o", ms=9, mfc=(c if sig else "white"),
                       mec=c, mew=1.6, zorder=3)
            else:
                b.plot(r.delta_pre, i, "x", ms=8, color=c, mew=1.6, zorder=3)
            txt = (f"p={r.p_placebo_year:.3f} ({int(r.placebo_n_years)} yr)" if tested
                   else f"< {P.MIN_YEARS} comparison years")
            b.annotate(txt, (r.delta_pre, i), xytext=(0, 9), textcoords="offset points",
                       fontsize=6.5, ha="center", color=F.INK2,
                       bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=5)
        b.axvline(0, color=F.INK2, lw=0.8)
        b.set_yticks(range(len(sub)))
        b.set_yticklabels([f"{F.NAME[r.site]}  {r.onset[:7]}" for r in sub.itertuples()],
                          fontsize=8)
        b.set_ylim(len(sub) - 0.5, -0.8)
        b.set_xlabel(f"Pre-onset change, W = {w} d (K)")
        b.set_title(f"({tag}) Pre-onset change, {w}-day windows", loc="left", fontsize=10)
        b.grid(axis="x", alpha=0.25, lw=0.6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=F.GREY, lw=6, alpha=0.45, label="Same-season placebo, 2.5–97.5 %"),
         Line2D([], [], ls="none", marker="o", ms=8, mfc=F.INK2, mec=F.INK2, label=EXC),
         Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=F.INK2, mew=1.6,
                label=NOEXC),
         Line2D([], [], ls="none", marker="x", ms=8, color=F.INK2, mew=1.6,
                label="Fewer than five comparison years")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.6, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, "fig9_pre_onset_state")


# Ruapehu ground comparison; GeoNet satellite-derived entries (MC02) excluded
def _table4_clean():
    t = pd.read_csv(S.ROOT / "results/11_analysis/validation_energy/A6_ground_comparison.csv")
    t = t.dropna(subset=["sat_dT_syn_weather_delta", "ground_clean_delta"])
    t = t[~((t.onset == "2007-09-25") & (t.window_d == 365))]   # merged in the 365-d analysis
    return t.rename(columns={"ground_clean_delta": "insitu_delta_K",
                             "sat_dT_syn_weather_delta": "sat_delta_rel_K"})


def fig4():
    from pipeline.analysis.validation_energy import ground_daily
    f = _patched(F.fig_ruapehu_validation, [
        ('from pipeline.insitu_validate import _insitu_daily', 'pass'),
        ('ins = _insitu_daily("ruapehu")', 'ins = ground_daily("clean")'),
        ('label="Lake thermometer (GeoNet)"', 'label="GeoNet lake temperature (hand samples)"'),
        ('"(a) Lake temperature around the 2006 and 2007 eruptions"',
         '"(a) Ruapehu Crater Lake temperature"'),
        ('ic = pd.read_csv(REC / "insitu_before_after.csv")', 'ic = _table4_clean()'),
        ('"Satellite relative temperature:\\nafter minus before (K)"',
         '"Relative temperature:\\nafter minus before (K)"'),
        ('"(b) Same windows, both instruments"', '"(b) Before–after temperature change"'),
        ('ax.legend(loc="upper left", fontsize=7.5, frameon=False)',
         'ax.legend(loc="lower right", fontsize=7.5, frameon=False)'),
        ('"Thermometer: after minus before (K)"', '"Ground (GeoNet): after minus before (K)"'),
        ('f" eruption\\n {t.date()}"', 'f"eruption \\n{t.date()} "'),
        ('fontsize=7, va="top", color=INK2)', 'fontsize=7, va="top", ha="right", color=INK2)'),
        ('xytext=(4, dy), textcoords="offset points", fontsize=6.5,',
         'xytext=((4, dy) if mk == "o" else (5, -11)), textcoords="offset points", fontsize=6.5,'),
    ], extra_globals={"ground_daily": ground_daily, "_table4_clean": _table4_clean})
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    return f()


# Site maps and thermal scenes, rendered from the cached imagery
def _site_maps_patched():
    from pipeline import s10_site_maps as SM
    ov = _patched(SM._overlay, [('label="Reference lake"', 'label="Reference water (Taal)"'),
                                ('ax.annotate("reference lake",', 'ax.annotate("reference water",')])
    return SM, ov


def fig1():
    plt.rcParams["axes.grid"] = False
    SM, ov = _site_maps_patched()
    f = _patched(SM.plot_site_overview, [
        ('ax.set_title(f"{LAKE_LABEL[k]}", fontsize=8.5, loc="left")',
         'ax.set_title(f"({chr(101 + i)}) {LAKE_LABEL[k]}", fontsize=8.5, loc="left")'),
        ('"Sentinel-2 true colour, cloud-masked median of 2019-2025 "\n             "dry-season scenes. Middle row: summit area; bottom row: crater close-up."',
         '"Sentinel-2 true colour, cloud-masked median of January–April scenes (January–May at Taal), "\n             "2019–2025. Middle row: summit area; bottom row: crater close-up."'),
        ('axw.set_xlim(-180, 180)', 'axw.set_xlim(-180, 184)'),
        ('fig.text(0.5, 0.012, "Sentinel-2 true colour', 'if False: fig.text(0.5, 0.012, "Sentinel-2 true colour'),
        ('fig.savefig(OUT / "site_overview.png", dpi=220, bbox_inches="tight")\n    plt.close(fig)',
         '_save(fig, "fig1_study_sites")'),
    ], extra_globals={"_overlay": ov, "_save": _save})
    return f()


def fig3():
    plt.rcParams["axes.grid"] = False
    SM, ov = _site_maps_patched()
    arrow = _patched(SM._north_arrow, [
        ('facecolor="w", edgecolor="k", lw=0.6', 'facecolor="k", edgecolor="w", lw=0.9'),
        ('fontweight="bold", color="w",', 'fontweight="bold", color="k",'),
        ('foreground="k")', 'foreground="w")'),
    ])
    bar = _patched(SM._scalebar, [('alpha=0.75, pad=1)', 'alpha=0.95, pad=1.5)')])
    pair = _patched(SM.plot_thermal_pair, [],
                    extra_globals={"_overlay": ov, "_north_arrow": arrow, "_scalebar": bar})
    f = _patched(SM.plot_fig_thermal, [
        ('f"eruption {PAIR_ONSET[k]}", fontsize=9)', 'f"onset {PAIR_ONSET[k]}", fontsize=9)'),
        ('fig.savefig(OUT / "thermal_pre_post.png", dpi=220, bbox_inches="tight")\n    plt.close(fig)',
         '_save(fig, "fig5_thermal_pre_post")'),
    ], extra_globals={"plot_thermal_pair": pair, "_save": _save})
    return f()


def main():
    fig1()
    fig3()
    plt.rcParams["axes.grid"] = True
    fig4()
    fig5(); fig6(); fig7(); fig8(); fig9(); fig10(); fig11()


if __name__ == "__main__":
    main()
