# -*- coding: utf-8 -*-
"""Location map, summit imagery and before/after thermal scenes of the study lakes.

Renders from cached imagery under S.DATA_DIR / "imagery" (Sentinel-2 true colour
and Landsat surface temperature GeoTIFFs per site, plus a world relief image),
the lake AOI polygons in S.DATA_DIR / "aoi", and the scene dates and lake medians
in results/10_site_maps/<site>_thermal_pair.json.

    plot_figS1         location map with summit and crater views
    plot_fig_thermal   before/after thermal scenes (one row per site)
    plot_thermal_pair  one before/after pair, used by plot_fig_thermal

analysis/figures.py re-executes these functions with revised labels and writes
the final figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from pipeline.config import settings as S
from pipeline.config.sites import get_site

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

#  the three study lakes
SITES4 = ("taal", "santa_ana", "ruapehu")
LABEL = {"taal": "Taal (Philippines)", "ruapehu": "Ruapehu (New Zealand)",
         "santa_ana": "Santa Ana / Ilamatepec (El Salvador)",
         }
LAKE_LABEL = {"taal": "Main Crater Lake", "ruapehu": "Crater Lake",
              "santa_ana": "Crater lake", }
#  Half-width (m) of the crater close-up (thermal panels, bottom map row);
#  the lake fills about a quarter of the panel.
HALF_CRATER = {"taal": 2600, "ruapehu": 900, "santa_ana": 700, }
#  Onset shown in the before/after thermal pair (one detected onset per site);
#  the scene dates are read from <site>_thermal_pair.json.
PAIR_ONSET = {"taal": "2020-01-12", "ruapehu": "2007-09-25",
              "santa_ana": "2005-06-16", }

OUT = S.DATA_DIR / "imagery"
OUT.mkdir(parents=True, exist_ok=True)
REC = S.records_dir("10s")


# ---------------------------------------------------------------------------
#  Coordinate helpers: lat/lon box from a half-width in metres
# ---------------------------------------------------------------------------
def _box_ll(lat, lon, half_m):
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _control(site):
    """First reference water body of the site, or None. Ruapehu has none (the
    background ring is the reference); the Taal reference lies outside the summit."""
    cs = [c for c in (site.controls or []) if c.kind == "control"]
    return cs[0] if cs else None


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
def _read(p: Path, bands=None):
    import rasterio
    from rasterio.plot import plotting_extent
    with rasterio.open(p) as s:
        a = s.read(bands) if bands else s.read()
        return a.astype(float), plotting_extent(s), s.nodata


def _crop(a, ext, lat, lon, half_m):
    """Crop a (band, row, col) or (row, col) array to a centre and half-width."""
    x0, x1, y0, y1 = ext
    ny, nx = a.shape[-2:]
    bx0, by0, bx1, by1 = _box_ll(lat, lon, half_m)
    c0 = int(np.clip((bx0 - x0) / (x1 - x0) * nx, 0, nx - 1))
    c1 = int(np.clip((bx1 - x0) / (x1 - x0) * nx, c0 + 1, nx))
    r0 = int(np.clip((y1 - by1) / (y1 - y0) * ny, 0, ny - 1))
    r1 = int(np.clip((y1 - by0) / (y1 - y0) * ny, r0 + 1, ny))
    sub = a[..., r0:r1, c0:c1]
    e2 = (x0 + c0 / nx * (x1 - x0), x0 + c1 / nx * (x1 - x0),
          y1 - r1 / ny * (y1 - y0), y1 - r0 / ny * (y1 - y0))
    return sub, e2


def _square(ax, ext):
    """Equal-distance aspect for a lat/lon panel so high-latitude sites are not distorted."""
    ax.set_aspect(1.0 / math.cos(math.radians((ext[2] + ext[3]) / 2)))


def _rgb(a):
    rgb = np.moveaxis(a[:3], 0, -1)
    lo, hi = np.nanpercentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
    return np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8


def _lake_rings(site_key: str, lake):
    p = S.DATA_DIR / "aoi" / f"{site_key}_{lake.key}.geojson"
    if not p.exists():
        return []
    g = json.loads(p.read_text(encoding="utf-8"))["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    return [np.asarray(pp[0]) for pp in polys]


def _water_ring(site_key, lake):
    """Water outline before erosion: the AOI buffered back by its recorded erosion
    width, the outline used for the water fraction."""
    from shapely.geometry import shape
    p = S.DATA_DIR / "aoi" / f"{site_key}_{lake.key}.geojson"
    if not p.exists():
        return []
    gj = json.loads(p.read_text(encoding="utf-8"))
    m = float(gj["properties"].get("erode_m_used") or 0)
    geom = shape(gj["geometry"])
    k = 111_320.0 * math.cos(math.radians(lake.lat))
    g2 = geom.buffer(m / k) if m else geom
    polys = [g2] if g2.geom_type == "Polygon" else list(g2.geoms)
    return [np.asarray(pp.exterior.coords) for pp in polys]


def _circle_ll(lat, lon, r_m, n=180):
    th = np.linspace(0, 2 * np.pi, n)
    return np.c_[lon + r_m * np.cos(th) / (111_320.0 * math.cos(math.radians(lat))),
                 lat + r_m * np.sin(th) / 111_320.0]


def _overlay(ax, site_key, ring=True, control=True, lw=1.3, note=True):
    site = get_site(site_key)
    a = site.active
    for xy in _water_ring(site_key, a):
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, ec="#00e5ff", lw=lw,
                                label="Crater lake"))
    for xy in _lake_rings(site_key, a):
        ax.add_patch(MplPolygon(xy, closed=True, fill=False, ec="#ffd400", lw=lw * 0.8,
                                ls="-", label="Sampling area (eroded)"))
    if ring:
        for r, lab in ((a.radius_m * 1.6, "Background ring"), (a.radius_m * 3.0, None)):
            xy = _circle_ll(a.lat, a.lon, r)
            ax.plot(xy[:, 0], xy[:, 1], color="#ff4d4d", lw=lw * 0.8, ls="--",
                    label=lab)
    c = _control(site)
    if control and c is not None:
        for xy in _water_ring(site_key, c) or [_circle_ll(c.lat, c.lon, c.radius_m)]:
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, ec="#7CFC00",
                                    lw=lw * 1.6, label="Reference lake"))
        ax.annotate("reference lake", (c.lon, c.lat), fontsize=7.5,
                    color="#2f6d00", fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#7CFC00",
                              lw=0.8, alpha=0.85))
    elif control and note:
        #  state explicitly when a site has no reference water
        ax.text(0.5, 0.045, "no reference lake — background ring only",
                transform=ax.transAxes, fontsize=7.5, ha="center", color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#8a8983",
                          lw=0.6, alpha=0.85))


def _scalebar(ax, ext, km):
    x0, x1, y0, y1 = ext
    lat = (y0 + y1) / 2
    dx = km * 1000 / (111_320.0 * math.cos(math.radians(lat)))
    xs, ys = x0 + 0.06 * (x1 - x0), y0 + 0.07 * (y1 - y0)
    ax.plot([xs, xs + dx], [ys, ys], color="w", lw=4, solid_capstyle="butt")
    ax.plot([xs, xs + dx], [ys, ys], color="k", lw=2, solid_capstyle="butt")
    ax.text(xs + dx / 2, ys + 0.025 * (y1 - y0), f"{km:g} km", ha="center",
            va="bottom", fontsize=8, color="k",
            bbox=dict(fc="w", ec="none", alpha=0.75, pad=1))


def _north_arrow(ax, x=0.90, y=0.80, size=0.13):
    """North arrow (all images are on a lat/lon grid, so north is up)."""
    import matplotlib.patheffects as pe
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(facecolor="w", edgecolor="k", lw=0.6,
                                width=2.8, headwidth=8.5, headlength=7.5))
    ax.text(x, y + size + 0.01, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9.5, fontweight="bold", color="w",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="k")])


def _ll_ticks(ax, ext, fs=6, margin=0.08):
    """Add lat/lon ticks (°N/°S, °E/°W) on the panel frame.

    One spacing from the 1-2-5 series is used for both axes, and the number of
    label decimals is derived from it so that labels are not rounded unevenly.
    """
    from matplotlib.ticker import FuncFormatter
    def _ticks(lo, hi, step):
        d = (hi - lo) * margin
        k0, k1 = math.ceil((lo + d) / step - 1e-9), math.floor((hi - d) / step + 1e-9)
        return [k * step for k in range(k0, k1 + 1)]

    #  smallest 1-2-5 spacing giving 2-3 ticks on both axes
    e = math.floor(math.log10(min(ext[1] - ext[0], ext[3] - ext[2]))) - 1
    cands = sorted(m * 10.0 ** k for k in (e - 1, e, e + 1, e + 2) for m in (1, 2, 5))
    step = next(c for c in cands
                if all(2 <= len(_ticks(lo, hi, c)) <= 3
                       for lo, hi in ((ext[0], ext[1]), (ext[2], ext[3]))))
    nd = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    xt, yt = _ticks(ext[0], ext[1], step), _ticks(ext[2], ext[3], step)

    def _fmt(hemi):
        def f(v, _pos):
            h = hemi[0] if v >= 0 else hemi[1]
            return f"{abs(v):.{nd}f}°{h}"
        return f
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt("EW")))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt("NS")))
    ax.tick_params(labelsize=fs, length=2.5, pad=1.5, direction="out")
    ax.tick_params(axis="y", labelrotation=90)
    for lab in ax.get_yticklabels():
        lab.set_va("center")


def _dedupe_legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for hh, ll in zip(h, l):
        if ll and ll not in seen:
            seen.add(ll); H.append(hh); L.append(ll)
    if H:
        ax.legend(H, L, **kw)


def _km_for(ext):
    w_km = (ext[1] - ext[0]) * 111.32 * math.cos(math.radians((ext[2] + ext[3]) / 2))
    for k in (0.5, 1, 2, 5, 10):
        if k >= w_km / 6:
            return k
    return 10


def _thermal(site_key, tag):
    p = OUT / site_key / f"thermal_{tag}.tif"
    if not p.exists():
        return None, None
    a, ext, _ = _read(p)
    lk = get_site(site_key).active
    a, ext = _crop(a, ext, lk.lat, lk.lon, HALF_CRATER[site_key])
    st, qa = a[0], a[1].astype(np.int64)
    bad = ((qa & (1 << 3)) | (qa & (1 << 4)) | (qa & (1 << 1)) | (qa & 1)) > 0
    st = np.where(bad | (st < 200), np.nan, st) - 273.15
    return st, ext


def _fmt_axes(ax):
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Longitude (°)", fontsize=8)
    ax.set_ylabel("Latitude (°)", fontsize=8)
    ax.ticklabel_format(useOffset=False)


def plot_thermal_pair(site_key, axes=None, compact=False):
    """Before and after scenes, each annotated with the lake (sampling-area) median.

    No difference map is drawn: a box-median reference is biased by seasonal ground
    heating, and a background-ring reference differs from the reference used for
    sites with a synthetic control. Reference-relative values are in the results tables.
    """
    pre, ext = _thermal(site_key, "pre")
    post, _ = _thermal(site_key, "post")
    if pre is None or post is None:
        return None
    meta = json.loads((REC / f"{site_key}_thermal_pair.json").read_text(encoding="utf-8"))
    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    vals = np.r_[pre[np.isfinite(pre)], post[np.isfinite(post)]]
    lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 99.3)
    for ax, arr, tag in ((axes[0], pre, "pre"), (axes[1], post, "post")):
        im = ax.imshow(arr, extent=ext, cmap="inferno", vmin=lo, vmax=hi)
        _overlay(ax, site_key, ring=False, control=False, lw=1.0)
        ax.set_title(f"{'Before' if tag == 'pre' else 'After'}: {meta[tag]['date']}",
                     fontsize=8 if compact else 9)
        #  lake median from the sampling area, as in the results tables
        ax.text(0.03, 0.97, f"lake {meta[tag]['T_lake_K'] - 273.15:.0f} °C",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8 if compact else 9, color="#0b0b0b",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); _square(ax, ext)
        if compact:
            _ll_ticks(ax, ext)
            _north_arrow(ax, x=0.91, y=0.77, size=0.12)
        else:
            _fmt_axes(ax)
    cb = plt.colorbar(im, ax=list(axes[:2]), fraction=0.03, pad=0.01)
    cb.set_label("Surface temperature (°C)", fontsize=7); cb.ax.tick_params(labelsize=6)
    _scalebar(axes[0], ext, _km_for(ext))
    if own:
        fig.suptitle(f"{LABEL[site_key]} - Landsat surface temperature around the "
                     f"eruption of {meta['onset']}", fontsize=10)
        return fig
    return None


def plot_figS1():
    """Location map with summit and crater views of the study lakes."""
    fig = plt.figure(figsize=(10, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.30, wspace=0.28)
    axw = fig.add_subplot(gs[0, :])
    w = OUT / "_world_etopo.png"
    if w.exists():
        axw.imshow(plt.imread(w), extent=[-180, 180, -60, 72])
    for i, k in enumerate(SITES4):
        s = get_site(k)
        axw.plot(s.active.lon, s.active.lat, "^", ms=11, mfc="#d62728", mec="k",
                 label="Study volcanoes")
        axw.annotate(f"({chr(98 + i)}) {LABEL[k].split(' (')[0]}",
                     (s.active.lon, s.active.lat),
                     xytext=((-78, 8) if s.active.lon > 150 else (6, 6)),
                     textcoords="offset points", fontsize=9, weight="bold",
                     bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    axw.set_xlim(-180, 180); axw.set_ylim(-60, 72)
    axw.set_xlabel("Longitude (°)", fontsize=8); axw.set_ylabel("Latitude (°)", fontsize=8)
    axw.tick_params(labelsize=7)
    axw.set_title("(a) Location of the three study volcanoes", fontsize=10, loc="left")
    _dedupe_legend(axw, loc="lower left", fontsize=8, framealpha=0.9)
    for i, k in enumerate(SITES4):
        #  middle row: summit true colour (lake, ring, reference); bottom row: crater close-up
        p = OUT / k / "volcano_truecolor.tif"
        if not p.exists():
            continue
        a, ext0, _ = _read(p)
        lk = get_site(k).active
        for row in (1, 2):
            ax = fig.add_subplot(gs[row, i])
            if row == 1:
                arr, ext = a, ext0
            else:
                pc = OUT / k / "crater_truecolor.tif"
                src = _read(pc)[:2] if pc.exists() else (a, ext0)
                arr, ext = _crop(src[0], src[1], lk.lat, lk.lon, HALF_CRATER[k])
            ax.imshow(_rgb(arr), extent=ext)
            #  no "no reference" note here (it would overlap the scale bar); the legend covers it
            _overlay(ax, k, ring=(row == 1), control=(row == 1), lw=1.0, note=False)
            if row == 1:
                ax.set_title(f"({chr(98 + i)}) {LABEL[k].split(' (')[0]}",
                             fontsize=9, loc="left")
            else:
                ax.set_title(f"{LAKE_LABEL[k]}", fontsize=8.5, loc="left")
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); _square(ax, ext)
            _ll_ticks(ax, ext)
            _north_arrow(ax, x=0.09, y=0.76, size=0.12)
            _scalebar(ax, ext, _km_for(ext))
    #  single legend collected from the map panels
    h, l = [], []
    for ax in fig.axes[1:]:
        hh, ll = ax.get_legend_handles_labels()
        for a_, b_ in zip(hh, ll):
            if b_ and b_ not in l:
                h.append(a_); l.append(b_)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.012, "Sentinel-2 true colour, cloud-masked median of 2019-2025 "
             "dry-season scenes. Middle row: summit area; bottom row: crater close-up.",
             ha="center", fontsize=8)
    fig.savefig(OUT / "figS1_study_sites.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fig_thermal():
    """Before/after thermal scenes of the study lakes (one row per site)."""
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 10.8),
                             gridspec_kw=dict(hspace=0.28, wspace=0.28))
    for i, k in enumerate(SITES4):
        plot_thermal_pair(k, axes=axes[i], compact=True)
        axes[i, 0].set_ylabel(f"({chr(97 + i)}) {LABEL[k].split(' (')[0]}\n"
                              f"eruption {PAIR_ONSET[k]}", fontsize=9)
    fig.savefig(OUT / "figS2_thermal_pre_post.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


