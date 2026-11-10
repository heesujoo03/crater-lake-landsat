# -*- coding: utf-8 -*-
"""Stage 10m: published magma-source geometry beside the post-eruption lake response.

Steps
-----
1. Convert published source depths (GEOM; sources in the *_src fields) to
   depth below the lake surface. Publications use different datums (below sea
   level, below the summit, or unstated), so values are not comparable without
   this conversion.
2. Attach the Stage 10 event contrasts (relative temperature, after minus before).
3. Compute two physical scales used for interpretation (not results in themselves):
   (a) Conduction time tau = z^2 / kappa: time for heat from a source at depth z
       to reach the lake by rock conduction alone; kappa = 1e-6 m^2 s^-1
       (typical crystalline rock).
   (b) Heat stored in the lake: the before/after lake surface temperature change
       dT and the lake volume V give E = rho c V dT; the magma volume able to
       supply it is V_m = E / (rho_m (L + c_m dT_m)), with rho = 1000 kg m^-3 and
       c = 4186 J kg^-1 K^-1 (water), rho_m = 2600 kg m^-3, L = 3.5e5 J kg^-1,
       c_m = 1200 J kg^-1 K^-1, dT_m = 300 K (andesitic magma cooling while it
       crystallises, as in Hurst et al. 1991 for Ruapehu).
   Conduction times of thousands to millions of years rule out a conductive
   response within months, so an observed response implies advective heat
   transport by gas or hydrothermal fluid; the equivalent magma volume is small
   compared with reservoir sizes.

Outputs: results/10_magma_geometry/ (magma_geometry_lake_datum,
response_vs_geometry) and the source-geometry figure.

    python -m pipeline.s10_magma
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import s10_figures as F
from pipeline import s10_post_eruption as P
from pipeline.common.report import StageCheck
from pipeline.config import settings as S

KAPPA = 1e-6
RHO_W, C_W = 1000.0, 4186.0
RHO_M, L_M, C_M, DT_M = 2600.0, 3.5e5, 1200.0, 300.0
YEAR = 3.156e7

#  ── Published source depths converted to km below the lake surface ─────────
#  lake_asl_km: lake surface elevation (km asl). shallow/deep: (central, low, high)
#  in km below the lake surface. Comments give the conversion; sources are in
#  the *_src fields.
GEOM = {
    "taal": dict(
        lake_asl_km=0.005, vent_offset_km=0.0, vent_note="vent in the lake",
        #  Morales Rivera et al. (2019) Mogi 3.1 (0.9–5.3) km below the summit;
        #  Volcano Island summit ≈ 0.3 km asl, lake ≈ 0 → ≈ 2.8 (0.6–5.0) below the lake
        shallow=(2.8, 0.6, 5.0), shallow_src="Morales Rivera et al. (2019)",
        #  Bato et al. (2021) ~5 km reservoir; Balangue-Tarriela et al. (2022) 5–6 km
        deep=(5.0, 5.0, 6.0), deep_src="Bato et al. (2021)",
        hydrothermal=(0.5, 4.0), style="phreatomagmatic (VEI 4, 2020)",
        lake_lost="2020-01 (refilled by 2020-03)", volume_m3=4.2e7,
        vol_src="Perttu et al. (2023)"),
    "ruapehu": dict(
        lake_asl_km=2.53, vent_offset_km=0.0, vent_note="vent beneath the lake",
        #  Kilgour et al. (2014) dyke–sill complex from ~2 km; Arens et al. (2022)
        #  mush 2–9 km bsl → 4.5–11.5 below the lake; centre 5.5 bsl → 8.0
        shallow=(2.0, 2.0, 4.5), shallow_src="Kilgour et al. (2014)",
        deep=(8.0, 4.5, 11.5), deep_src="Arens et al. (2022)",
        hydrothermal=(0.0, 1.0), style="phreatic (2006, 2007)",
        lake_lost="not lost in 2006-2007", volume_m3=9.0e6,
        vol_src="Christenson (1994)"),
    "santa_ana": dict(
        lake_asl_km=2.187, vent_offset_km=0.0, vent_note="vent in the lake",
        #  No published depth: the 2004-2005 activity is described as gradual
        #  intrusion of very shallow, undegassed magma (Scolamacchia et al., 2010)
        #  without a depth or datum, so the entry is left empty.
        shallow=(np.nan, np.nan, np.nan),
        shallow_src="no depth published to the compilation standard",
        deep=(np.nan, np.nan, np.nan), deep_src="not constrained",
        hydrothermal=(np.nan, np.nan), style="phreatomagmatic (VEI 3, 2005)",
        lake_lost="not lost; area fell to 70 % during 2006-2007",
        volume_m3=4.7e5, vol_src="Bernard et al. (2004)"),
}


def build():
    ev = pd.read_csv(S.records_dir("10") / "per_eruption_delta_paper.csv")
    meta = pd.read_json(S.records_dir("10") / "site_meta_paper.json").set_index("site")
    rows = []
    for k, g in GEOM.items():
        if k not in meta.index or k not in F.SITES4:   # responses only for the study sites
            continue
        est = meta.loc[k, "estimator_used"]
        #  Use the window in which each onset was tested: 180 d, or 365 d where
        #  the 180-d window has too few scenes (Taal).
        ee = ev[(ev["site"] == k) & ev["window_d"].isin([P.PRIMARY_W, 365])]
        rel_all = ee[ee["estimator"] == est]
        for onset in sorted(rel_all["onset"].unique()):
            cand = rel_all[(rel_all["onset"] == onset) & rel_all["delta"].notna()]
            if not len(cand):
                continue
            #  window with the smallest year-placebo p (ties: shorter window)
            cand = cand.sort_values(["p_placebo_year", "window_d"])
            r = cand.iloc[0]
            w = int(r["window_d"])
            e = ee[ee["window_d"] == w]
            lst = e[e["estimator"] == "T_active_K"].set_index("onset")
            q = e[e["estimator"] == "Q_total_MW"].set_index("onset")
            dT = float(lst.loc[onset, "delta"]) if onset in lst.index else np.nan
            E = RHO_W * C_W * g["volume_m3"] * dT if np.isfinite(dT) else np.nan
            vm = E / (RHO_M * (L_M + C_M * DT_M)) if np.isfinite(E) else np.nan
            z = g["shallow"][0] if np.isfinite(g["shallow"][0]) else g["deep"][0]
            rows.append(dict(
                site=k, onset=onset, estimator=est, window_d=w,
                delta_rel_K=round(float(r["delta"]), 2),
                p_placebo_year=r["p_placebo_year"],
                placebo_n_years=r["placebo_n_years"],
                p_placebo_grid=r["p_placebo_season"],
                delta_lst_K=(round(dT, 2) if np.isfinite(dT) else np.nan),
                delta_Qtotal_MW=(round(float(q.loc[onset, "delta"]), 1)
                                 if onset in q.index and
                                 np.isfinite(q.loc[onset, "delta"]) else np.nan),
                lake_volume_m3=g["volume_m3"],
                stored_heat_J=(float(f"{E:.3g}") if np.isfinite(E) else np.nan),
                magma_equiv_m3=(float(f"{vm:.3g}") if np.isfinite(vm) else np.nan),
                shallow_depth_km=g["shallow"][0], deep_depth_km=g["deep"][0],
                vent_offset_km=g["vent_offset_km"],
                conduction_time_yr=(round((z * 1e3) ** 2 / KAPPA / YEAR, -2)
                                    if np.isfinite(z) else np.nan)))
    resp = pd.DataFrame(rows)
    geo = pd.DataFrame([dict(site=k, lake_asl_km=g["lake_asl_km"],
                             shallow_km=g["shallow"][0], shallow_lo=g["shallow"][1],
                             shallow_hi=g["shallow"][2], shallow_src=g["shallow_src"],
                             deep_km=g["deep"][0], deep_lo=g["deep"][1],
                             deep_hi=g["deep"][2], deep_src=g["deep_src"],
                             hydro_top_km=g["hydrothermal"][0],
                             hydro_bottom_km=g["hydrothermal"][1],
                             vent_offset_km=g["vent_offset_km"], vent_note=g["vent_note"],
                             eruption_style=g["style"], lake_lost=g["lake_lost"],
                             volume_m3=g["volume_m3"], volume_src=g["vol_src"])
                        for k, g in GEOM.items()])
    return geo, resp


VENT_LABEL = {"taal": "vent in\nthe lake",
              "santa_ana": "vent in\nthe lake",
              "ruapehu": "vent beneath\nthe lake",
              }


def fig_geometry(geo: pd.DataFrame, resp: pd.DataFrame):
    """(a) Source depths below the lake surface; (b) post-eruption contrasts."""
    #  Ordered by shallowest source; Santa Ana (no published depth) goes last.
    order = ["ruapehu", "taal", "santa_ana"]
    g = geo.set_index("site").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    for i, k in enumerate(order):
        r = g.loc[k]
        c = F.COLOR.get(k, F.GREY)
        ax.add_patch(plt.Rectangle((i - 0.32, r["hydro_top_km"]), 0.64,
                                   r["hydro_bottom_km"] - r["hydro_top_km"],
                                   color="#dfe9f5", lw=0, zorder=0))
        if np.isfinite(r["shallow_km"]):
            ax.plot([i - 0.12, i - 0.12], [r["shallow_lo"], r["shallow_hi"]], color=c,
                    lw=3, solid_capstyle="round")
            ax.plot(i - 0.12, r["shallow_km"], "D", ms=8, mfc=c, mec="#0b0b0b", mew=0.6)
        if np.isfinite(r["deep_lo"]):
            ax.plot([i + 0.12, i + 0.12], [r["deep_lo"], r["deep_hi"]], color=c, lw=3,
                    alpha=0.55, solid_capstyle="round")
            if np.isfinite(r["deep_km"]):
                ax.plot(i + 0.12, r["deep_km"], "o", ms=8, mfc="white", mec=c, mew=1.8)
        ax.plot(i, -0.25, "v", ms=11, mfc="#00b4d8" if r["vent_offset_km"] < 0.5
                else "white", mec="#0b0b0b", mew=0.7, clip_on=False)
        ax.text(i, -0.75, VENT_LABEL[k], ha="center", va="bottom", fontsize=6.5,
                color=F.INK2)
        #  Label a site without a published depth explicitly rather than
        #  leaving its column blank.
        if not (np.isfinite(r["shallow_km"]) or np.isfinite(r["deep_lo"])):
            ax.text(i, 5.5, "no depth\npublished", ha="center", va="center",
                    fontsize=7.5, color=F.INK2, style="italic")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([F.NAME[k] for k in order], fontsize=8.5)
    ax.set_ylim(12, -1.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Depth below the lake surface (km)")
    ax.set_title("(a) Magma sources beneath each lake (literature)", loc="left",
                 fontsize=9.5)
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], ls="none", marker="D", ms=7, mfc=F.GREY, mec="#0b0b0b",
                label="Shallowest magma or intrusion"),
         Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec=F.GREY, mew=1.6,
                label="Deep reservoir"),
         Patch(color="#dfe9f5", label="Hydrothermal system"),
         Line2D([], [], ls="none", marker="v", ms=8, mfc="#00b4d8", mec="#0b0b0b",
                label="Eruptive vent under or at the lake")]
    ax.legend(handles=h, loc="lower right", fontsize=7, frameon=True, framealpha=0.95)

    ax = axes[1]
    from matplotlib.lines import Line2D
    for i, k in enumerate(order):
        r = resp[resp["site"] == k].sort_values("onset")
        c = F.COLOR.get(k, F.GREY)
        n = len(r)
        for j, (_, rr) in enumerate(r.iterrows()):
            x = i + (j - (n - 1) / 2) * 0.09      # offset onsets of the same site
            sig = (np.isfinite(rr["p_placebo_year"]) and rr["placebo_n_years"] >= 5
                   and abs(rr["p_placebo_year"] - 1.0 / (rr["placebo_n_years"] + 1))
                   < 2e-4)
            ax.plot(x, rr["delta_rel_K"], "o", ms=7,
                    mfc=(c if sig else "white"), mec=c, mew=1.6, zorder=3)
            if sig:
                # Annotate onset year and window length (365 d at Taal, 180 d elsewhere)
                ax.annotate(f"{rr['onset'][:4]}\n{int(rr['window_d'])} d",
                            (x, rr["delta_rel_K"]),
                            xytext=(6, -2), textcoords="offset points", fontsize=6.2,
                            va="center", color=F.INK2)
    ax.axhline(0, color=F.INK2, lw=0.7)
    #  Panel (b) keeps the site order of (a) but carries no depth labels.
    lab = []
    for k in order:
        lab.append(F.NAME[k])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.15)
    ax.grid(axis="x", visible=False)
    #  explicit line break keeps the label within the panel width
    ax.set_xlabel("Volcanoes in the same order as (a); published depths\n"
                  "overlap or are absent and do not order these responses.",
                  fontsize=8)
    ax.set_ylabel("Change in relative temperature,\nafter minus before (K)")
    ax.set_title("(b) Post-eruption response of each lake", loc="left", fontsize=9.5)
    h2 = [Line2D([], [], ls="none", marker="o", ms=7, mfc=F.INK2, mec=F.INK2,
                 label="Resolved"),
          Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec=F.INK2, mew=1.6,
                 label="Tested, not resolved")]
    ax.margins(y=0.12)
    ax.legend(handles=h2, loc="lower left", fontsize=7, frameon=True, framealpha=0.95)
    fig.tight_layout()
    return F._save(fig, "fig_magma_geometry")


def main():
    chk = StageCheck(stage="10m", title="Magma geometry, post-eruption response and physical scales",
                     desc="Converts published magma depths to the lake-surface datum and "
                          "places them beside the Stage 10 responses, with conduction "
                          "time and stored heat as interpretive scales.")
    geo, resp = build()
    chk.table(geo, "magma_geometry_lake_datum",
              desc="Magma and hydrothermal-system depths below the lake surface")
    chk.table(resp, "response_vs_geometry",
              desc="Per-onset response, stored heat, equivalent magma volume, conduction time")
    chk.param(kappa=KAPPA, rho_w=RHO_W, c_w=C_W, rho_m=RHO_M, L_m=L_M, c_m=C_M,
              dT_m=DT_M)
    chk.note("Conduction time and equivalent magma volume are interpretive scales, "
             "not results; V is the published (pre-eruption) lake volume, so the "
             "values are approximate.")
    fig_geometry(geo, resp)
    chk.finish()


if __name__ == "__main__":
    main()
