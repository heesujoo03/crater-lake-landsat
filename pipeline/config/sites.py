# -*- coding: utf-8 -*-
"""
Site definitions: the three crater lakes (Taal, Santa Ana, Ruapehu) and their
temperature references.

Design
------
1. A site consists of one active lake plus either one or more control lakes
   (paired differencing) or, when no suitable control exists, a background
   ring of adjacent land (``default_mode="bg_ref"``). A control lake must be
   (a) close enough and (b) free of geothermal input.
2. A large elevation difference to the control breaks the assumption that
   atmospheric and synoptic effects cancel in the difference. Above
   ELEV_DIFF_WARN_M a lapse-rate covariate is required; above ELEV_DIFF_FAIL_M
   paired differencing is not used.

This module is the single source of truth for sites; all modules read it via
`get_site`. `known_events` holds the GVP eruption onsets used by the analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Lake:
    """A single lake (AOI)."""
    key: str                    # identifier used in file and column names (ASCII, no spaces)
    label: str                  # human-readable name
    lat: float
    lon: float
    radius_m: float             # approximate AOI radius
    elev_m: float               # water-surface elevation (for lapse-rate correction)
    kind: str                   # "active" | "control"
    area_km2_ref: Optional[float] = None   # literature area
    # -- Depth and volume ----------------------------------------------------
    # Area change alters the mixed-pixel water fraction f and can create
    # spurious temperature trends, so a time-varying f is used.
    mean_depth_m: Optional[float] = None    # literature mean depth
    max_depth_m: Optional[float] = None     # literature maximum depth
    hypsometry_p: Optional[float] = None    # V ∝ A^(1+2/p); None uses the settings default
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SitePair:
    """Analysis unit: one active lake plus N control lakes (N may be 0)."""
    key: str
    label: str
    country: str
    active: Lake
    controls: List[Lake]
    utm_epsg: int                     # projected CRS for area and distance calculations
    # Reference design for this site.
    #   paired      : difference against a nearby non-volcanic lake (requires a control)
    #   bg_ref      : difference against a ring of surrounding land (always available)
    #   active_only : no reference (last resort)
    default_mode: str = "paired"
    role: str = ""                    # short description of the site's role in the analysis
    aws_stations: List[Dict] = field(default_factory=list)  # ground weather stations
    # In-situ lake-temperature stations. Agency CSV files converted to the
    # standard schema are read from data/raw/ground_thermal/<site>/.
    ground_thermal: List[Dict] = field(default_factory=list)
    known_events: List[Dict] = field(default_factory=list)  # GVP eruption onsets
    #  Smithsonian GVP volcano number, the source of known_events.
    gvp_number: Optional[int] = None
    # Default observation period. Starts in 1990 because Landsat 5/7 are used.
    start_date: str = "1990-01-01"
    end_date: str = "2026-08-12"
    #  -- Sentinel-2 months (start, end) used to draw the AOI ----------------
    #  The AOI was derived from a multi-year Sentinel-2 composite; listing
    #  snow-free months restricts that composite. Ranges may wrap the year,
    #  e.g. (12, 4). None uses all months. Affects the AOI boundary only.
    s2_months: Optional[List[int]] = None
    #  -- Sentinel-2 period used to draw the AOI ----------------------------
    #  The AOI should represent the lake when full; the actual fill state is
    #  recorded per date by the time-varying water fraction f. A high-water
    #  window can be set here; None uses the default period.
    s2_start: Optional[str] = None
    s2_end: Optional[str] = None
    note: str = ""

    # -- Derived properties ---------------------------------------------
    @property
    def lakes(self) -> List[Lake]:
        return [self.active] + list(self.controls)


# ---------------------------------------------------------------------------
# Site catalogue
# ---------------------------------------------------------------------------
# PRIMARY_SITE is the default of get_site().

SITES: Dict[str, SitePair] = {}


#  -- Taal (Philippines) ------------------------------------------------------
#  Active: Main Crater Lake on Volcano Island.
#  Reference: open water of Taal Lake (caldera lake), 5.2 km away. Both targets
#  fall in the same cell of operational atmospheric-correction grids (NARR,
#  GEOS-5, ERA5), which supports common-mode cancellation in the difference.
#  Coordinates were fitted to the Sentinel-2 water surface.
#  Eruption onsets: Smithsonian GVP, Volcanoes of the World, Holocene Eruptions
#  (Volcano_Number=273070, confirmed eruptions only). The 2024-04-12 to
#  2026-03-16 eruption is a single GVP entry.
SITES["taal"] = SitePair(
    key="taal",
    gvp_number=273070,               # GVP Taal
    label="Taal (Main Crater Lake ↔ Taal Lake main basin)",
    country="PH",
    utm_epsg=32651,                 # UTM 51N
    active=Lake(
        key="main_crater", label="Taal Main Crater Lake",
        lat=14.01035, lon=120.99826, radius_m=503, elev_m=10,
        kind="active", area_km2_ref=1.2,
        mean_depth_m=76.0, max_depth_m=76.0, hypsometry_p=2.0,
        note="Acid crater lake on Volcano Island; large enough for pure water pixels "
             "in Landsat TIR.",
    ),
    controls=[
        Lake(
            key="taal_lake", label="Taal Lake (caldera lake)",
            lat=14.0450, lon=121.0300, radius_m=1500, elev_m=3,
            kind="control", area_km2_ref=234.0,
            mean_depth_m=60.0, max_depth_m=172.0, hypsometry_p=2.0,
            note="Open-water point of the caldera lake, 5.2 km from the active lake, "
                 "in the same atmospheric-correction grid cell (NARR, GEOS-5, ERA5).\n"
                 "  The caldera lake may itself receive hydrothermal input; the paired "
                 "difference is therefore a conservative (lower-bound) estimate.",
        ),
    ],
    known_events=[
        dict(date="2020-01-12", start="2020-01-12", end="2020-01-22",
             label="crater eruption", severity="major", vei=4,
             source="GVP Taal (273070)", verified=True),
        dict(date="2021-07-01", start="2021-07-01", end="2021-07-09",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Taal (273070)", verified=True),
        dict(date="2021-11-15", start="2021-11-15", end="2022-04-01",
             label="Main Crater", severity="moderate", vei=2,
             source="GVP Taal (273070)", verified=True),
        dict(date="2022-10-05", start="2022-10-05", end="2022-10-29",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Taal (273070)", verified=True),
        dict(date="2024-04-12", start="2024-04-12", end="2026-03-16",
             label="crater eruption", severity="moderate", vei=2,
             source="GVP Taal (273070)", verified=True),
    ],
    # Landsat 5 and 7 are included so that the pre-2020 quiet period is long.
    start_date="1990-01-01",
    note="Paired with Taal Lake, whose large thermal inertia differs from the "
         "active lake (β₁ ≠ 1 expected). Period from 1990 (Landsat 5/7 included).",
)


#  -- Ruapehu (New Zealand) ---------------------------------------------------
#  Active: Crater Lake (Te Wai ā-moe), the in-situ validation site: GeoNet
#  monitors lake temperature continuously.
#  Reference: background ring (bg_ref). No control lake is used because the
#  nearest lakes lie more than 1,000 m lower in a different microclimate (snow
#  vs grassland), which lapse-rate correction cannot reconcile.
#  Eruption onsets: Smithsonian GVP, Volcanoes of the World, Holocene Eruptions
#  (Volcano_Number=241100, confirmed eruptions only). Heating episodes that
#  are not eruptions are listed as intervals in thermal_episodes.py["ruapehu"],
#  not here.
SITES["ruapehu"] = SitePair(
    key="ruapehu",
    gvp_number=241100,               # GVP Ruapehu
    label="Ruapehu Crater Lake (New Zealand)",
    country="NZ",
    utm_epsg=32760,                   # UTM 60S
    default_mode="bg_ref",
    role="Absolute validation anchor (continuous in-situ lake temperature)",
    active=Lake(
        key="crater_lake", label="Ruapehu Crater Lake (Te Wai ā-moe)",
        lat=-39.2817, lon=175.5644, radius_m=230, elev_m=2530,
        kind="active", area_km2_ref=0.19,
        mean_depth_m=100.0, hypsometry_p=2.0,
        note="GeoNet monitors lake temperature continuously.",
    ),
    controls=[],
    ground_thermal=[
        dict(id="ruapehu_geonet", name="GeoNet Crater Lake temperature",
             kind="lake_insitu", source="geonet_api",
             lat=-39.2817, lon=175.5644,
             note="GeoNet FITS API is public without authentication. Data are "
                  "supplied as CSV converted to the standard schema."),
    ],
    known_events=[
        dict(date="1990-01-07", start="1990-01-07", end="1990-01-26",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1990-06-17", start="1990-06-17", end="1990-09-08",
             label="crater eruption", severity="minor", vei=0,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1991-07-05", start="1991-07-05", end="1991-07-14",
             label="crater eruption", severity="minor", vei=1,
             note="GVP end date is marked uncertain (?)",
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1992-02-08", start="1992-02-08", end="1992-03-06",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1995-01-11", start="1995-01-11", end="1995-11-09",
             label="1995 major eruption (crater lake ejected)", severity="major", vei=3,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1996-06-16", start="1996-06-16", end="1996-09-01",
             label="1996 major eruption", severity="major", vei=3,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="1997-10-09", start="1997-10-09", end="1997-10-18",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="2006-10-04", start="2006-10-04", end="2006-10-04",
             label="crater eruption", severity="minor", vei=1,
             source="GVP Ruapehu (241100)", verified=True),
        dict(date="2007-09-25", start="2007-09-25", end="2007-09-25",
             label="phreatic eruption", severity="minor", vei=1,
             source="GVP Ruapehu (241100)", verified=True),
    ],
    start_date="1990-01-01",
    note="Background-ring reference (bg_ref); GeoNet lake temperature is used for validation.",
)


#  -- Santa Ana / Ilamatepec (El Salvador) -------------------------------------
#  Active: acid crater lake in the summit crater (2,187 m).
#  Reference: background ring (bg_ref); no nearby lake lies within the
#  elevation-difference limit for paired differencing (ELEV_DIFF_FAIL_M).
#  Eruption onset: Smithsonian GVP (Volcano_Number=343020).
#  GVP records the eruption start as 2005-06-16; the major explosion of
#  2005-10-01 is part of this eruption. Tests use the GVP start date.
SITES["santa_ana"] = SitePair(
    key="santa_ana",
    gvp_number=343020,               # GVP Santa Ana
    label="Santa Ana / Ilamatepec (crater lake ↔ background ring)",
    country="SV",
    utm_epsg=32616,                  # UTM 16N
    default_mode="bg_ref",
    role="Post-eruption warming case",
    active=Lake(
        key="crater_lake", label="Santa Ana Crater Lake (Laguna del cráter)",
        lat=13.84964, lon=-89.62813, radius_m=150, elev_m=2187,
        kind="active", area_km2_ref=0.055,
        note="Coordinates are the centroid of the maximum Sentinel-2 MNDWI water "
             "body. Elevation sampled from the Copernicus DEM (2,187 m).",
    ),
    controls=[],
    known_events=[
        dict(date="2005-06-16", start="2005-06-16", end="2005-10-01",
             label="crater eruption", severity="major", vei=3,
             note="GVP end date is marked uncertain (?)",
             source="GVP Santa Ana (343020)", verified=True),
    ],
    note="Background-ring reference (bg_ref). GVP confirmed eruption "
         "2005-06-16 to 2005-10-01 (VEI 3).",
)


PRIMARY_SITE = "taal"


# ---------------------------------------------------------------------------
# Pair-validity pre-check
# ---------------------------------------------------------------------------

ELEV_DIFF_WARN_M = 300.0     # above this a lapse-rate covariate is required
ELEV_DIFF_FAIL_M = 1000.0    # above this paired differencing is not used


def get_site(key: Optional[str] = None) -> SitePair:
    """Return the SitePair for ``key`` (default PRIMARY_SITE)."""
    k = (key or PRIMARY_SITE).lower()
    if k not in SITES:
        raise KeyError(f"unknown site '{k}'. Available: {sorted(SITES)}")
    return SITES[k]
