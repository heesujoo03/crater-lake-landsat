# -*- coding: utf-8 -*-
"""Workflow diagram of the analysis, with every input connected to the module
that uses it: Landsat and water-boundary data -> Module A, ERA5-Land -> Module B,
GeoNet lake temperature -> Module C (ground comparison), GVP catalogue ->
Module D (event windows) and Module B (quiet fitting sample).

    python -m pipeline.analysis.fig2_workflow
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pipeline.config import settings as S

OUT = S.ROOT / "figures" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f1f1f"
MOD = {"A": ("#2b7a8c", "#e8f3f5"), "B": ("#3d8b57", "#eaf4ec"),
       "C": ("#b7852a", "#faf2e2"), "D": ("#b0434f", "#f8e8ea")}

# inputs, each placed directly above the module that uses it
INPUTS = [  # (module, title, subtitle)
    ("A", "Landsat 5–9\nCollection-2 Level-2\nsurface temperature", "ST and QA bands,\n1990–2026"),
    ("A", "Water-boundary\ninformation", "Sentinel-2 polygon\nand water area;\nLandsat water class"),
    ("B", "ERA5-Land", "Hourly meteorology at the\nlake grid cell"),
    ("C", "GeoNet lake temperature", "Ruapehu hand samples and\ntelemetry (station RU001)"),
    ("D", "Global Volcanism Program\ncatalogue", "Confirmed eruption onsets\nand end dates"),
]

MODULES = {
    "A": ("Scene-level measurement", [
        "Sampling polygon and\nbackground ring",
        "Cloud and quality screening\n(≥ 80 % clear)",
        "PSF water fraction f and\nradiance unmixing",
        "Water-fraction screen\nf ≥ 0.70",
    ]),
    "B": ("Relative temperature", [
        "Synthetic reference\n(non-negative donor weights)",
        "ERA5-Land component, fitted on\nscenes > 180 d from onsets",
        "Relative temperature y(t)",
    ]),
    "C": ("Observability diagnostics", [
        r"Short-lag variability" "\n" r"proxy $\sigma_{\mathrm{pair}}$",
        "Variance partition and\nstructure function",
        "Geometry scaling",
        "Ground comparison (Ruapehu)",
    ]),
    "D": ("Event-window comparison", [
        "Onset anchors, episode merging\n(≥ 2 scenes per side)",
        r"Before–after contrast $\Delta_W$",
        "Same-season placebo\nyears, rank index",
        "Maximum-placebo exceedance\nand sensitivity analyses",
    ]),
}

def main():
    fig = plt.figure(figsize=(13.5, 6.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.6); ax.axis("off")

    mw, mg = 3.05, 0.3
    mx = {k: 0.3 + i * (mw + mg) for i, k in enumerate("ABCD")}
    top, bot = 4.05, 0.25

    # input row: boxes aligned with their modules, straight arrows down
    ax.text(0.32, 6.42, "INPUT DATA", fontsize=10.5, color="#555", weight="bold", va="center")
    iy0, ih = 4.72, 1.45
    per = {}
    for k, *_ in INPUTS:
        per[k] = per.get(k, 0) + 1
    seen = {}
    for k, t, sub in INPUTS:
        n, j = per[k], seen.get(k, 0); seen[k] = j + 1
        gap = 0.12
        w = (mw - gap * (n - 1)) / n
        x = mx[k] + j * (w + gap)
        ax.add_patch(FancyBboxPatch((x, iy0), w, ih, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="white", ec="#9aa1ab", lw=1))
        fs = 10 if n == 1 else 8.8
        ax.text(x + w / 2, iy0 + ih * 0.68, t, ha="center", va="center", fontsize=fs, weight="bold",
                color=INK)
        ax.text(x + w / 2, iy0 + ih * 0.25, sub, ha="center", va="center", fontsize=fs - 0.8, color="#444")
        ax.annotate("", xy=(x + w / 2, top + 0.03), xytext=(x + w / 2, iy0 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=MOD[k][0], lw=1.5, shrinkA=0, shrinkB=0))

    for k, (title, items) in MODULES.items():
        ec, fc = MOD[k]
        x = mx[k]
        ax.add_patch(FancyBboxPatch((x, bot), mw, top - bot, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.6))
        ax.add_patch(FancyBboxPatch((x, top - 0.62), mw, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=ec, ec=ec, lw=1.6))
        ax.text(x + 0.18, top - 0.31, k, fontsize=19, weight="bold", color="white", va="center")
        ax.text(x + 0.5, top - 0.31, title, fontsize=11.2, weight="bold", color="white", va="center")
        hy = 0.62
        for j, it in enumerate(items):
            y = top - 0.85 - j * (hy + 0.14)
            ax.add_patch(FancyBboxPatch((x + 0.12, y - hy + 0.12), mw - 0.24, hy - 0.06,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        fc="white", ec=ec, lw=0.8))
            ax.text(x + mw / 2, y - hy / 2 + 0.12, it, ha="center", va="center", fontsize=10.2,
                    color=INK)
        if k != "D":
            ax.annotate("", xy=(x + mw + mg - 0.02, 2.15), xytext=(x + mw + 0.02, 2.15),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))

    from pipeline.analysis.figcheck import check
    iss = check(fig, "fig2_workflow")
    if iss:
        print(f"   ! fig2_workflow: {len(iss)} overlap issue(s)")
    p = OUT / "fig2_workflow.png"
    fig.savefig(p, dpi=250)
    sub = OUT / "submission"; sub.mkdir(exist_ok=True)
    fig.savefig(sub / "fig2_workflow.pdf")
    fig.savefig(sub / "fig2_workflow.png", dpi=600)
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    main()
