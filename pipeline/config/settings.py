# -*- coding: utf-8 -*-
"""
Global settings: paths, per-stage output folders and shared constants.

Modules read their constants from this module rather than hard-coding values.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # parent folder of pipeline/
CHECK_DIR = ROOT / "figures" / "check"                          # visual outputs: PNG · GeoTIFF · GeoJSON

DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RECORDS_DIR = ROOT / "results"  # per-stage numeric records: CSV · JSON · check summaries

# Per-stage output sub-folders, keyed by stage; keys are used as-is so that
# folder names stay stable.
CHECK_STAGES = {
    2:    "02_lst",
    "2b": "02b_water_extent",     # water-surface area and level change (Sentinel-2)
    3:    "03_deltaT",
    9:    "09_unrest_catalogue",
    "10": "10_post_eruption",        # before/after comparison against same-season placebo onsets
    "10p": "10_pre_eruption",        # the same design applied to the windows before each onset
    "10d": "10_detectability",       # detectability: noise, depth, window length
    "10m": "10_magma_geometry",      # literature values for magma and vent geometry (manually compiled)
    "10s": "10_site_maps",           # study-area maps and satellite imagery
}


def _stage_sub(root: Path, stage, site: str = None, create: bool = True) -> Path:
    p = root / CHECK_STAGES[stage]
    if site:
        p = p / site
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def check_dir(stage, site: str = None, create: bool = True) -> Path:
    """Per-stage folder for visual outputs only (figures, GeoTIFF, GeoJSON).

    Numeric tables and check records go to records_dir().
    """
    return _stage_sub(CHECK_DIR, stage, site, create)


def records_dir(stage, site: str = None, create: bool = True) -> Path:
    """Per-stage folder for numeric records: CSV, JSON, _SUMMARY.md, _manifest.json."""
    return _stage_sub(RECORDS_DIR, stage, site, create)


# ---------------------------------------------------------------------------
# Earth Engine project
# ---------------------------------------------------------------------------
# Read from the EE_PROJECT environment variable. The code in this package reads
# cached inputs and does not call Earth Engine.
EE_PROJECT = os.environ.get("EE_PROJECT")


# Landsat C2 L2 surface temperature (ST_B10) scaling
ST_SCALE, ST_OFFSET = 0.00341802, 149.0


# ΔT outlier removal (robust z based on MAD)
DELTA_T_OUTLIER_Z = 5.0


# Model fitting uses only the quiet baseline; known activity events
# ± BASELINE_EXCLUDE_DAYS are excluded from the fitting sample.
BASELINE_EXCLUDE_DAYS = 180


CSV_ENCODING = "utf-8-sig"      # BOM so that spreadsheet software detects UTF-8
