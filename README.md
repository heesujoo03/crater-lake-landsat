# Detectability of crater lake temperature changes associated with eruptions in multidecadal Landsat records

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24095186.svg)](https://doi.org/10.5281/zenodo.24095186)

Code and input data for the article

> Joo, H.-S., Jung, H.-S. Detectability of crater lake temperature changes associated with eruptions in
> multidecadal Landsat records. *Submitted.*

Archived version: https://doi.org/10.5281/zenodo.24095186

The repository contains the code and data used in the article for Taal Main Crater Lake (Philippines),
Santa Ana (Ilamatepec) Crater Lake (El Salvador) and Ruapehu Crater Lake (Te Wai ā-moe, New Zealand).
It contains only the code that the analysis executes and the input files that this code reads.

## Contents

```
.
├── README.md
├── LICENSE              MIT licence for the code
├── CITATION.cff         how to cite this repository
├── environment.yml      conda environment (Python 3.10)
├── requirements.txt     the same packages for pip
├── pipeline/            source code (Python package)
│   ├── config/          lake definitions, eruption onsets, paths
│   ├── common/          quiet-period definition and output recorder
│   ├── s2_…s9_*.py      mixing correction, synthetic reference, event statistics, uncertainty
│   ├── s10_*.py         event windows, placebo tests, detectability, figures
│   └── analysis/        analyses reported in the article and figure rendering
├── data/                inputs
├── results/             numerical outputs (written when the code runs)
└── figures/             figures (written when the code runs)
```

`results/` initially holds the per-scene satellite samples that the code reads from its own output
folders (`02_lst`, `03_deltaT`, `10_site_maps`, `11_analysis/taal_reference_spatial`,
`12_diagnostics/samples_buf`).

## Installation

```bash
conda env create -f environment.yml
conda activate crater-lake-landsat
```

Run all commands from the repository root so that `python -m pipeline...` finds the package.

## Usage

Run the steps in this order; later steps read the outputs of earlier ones.
The analysis runs offline in about 30 min on a desktop computer.

```bash
python -m pipeline.s10_post_eruption                        # event windows, contrasts, placebo ranks
python -m pipeline.s10_pre_eruption                         # pre-onset lake state
python -m pipeline.s10_detectability                        # variability and persistence diagnostics
python -m pipeline.s10_magma                                # published source and vent geometry
python -m pipeline.analysis.reference_model                 # synthetic reference and ERA5-Land fits
python -m pipeline.analysis.water_fraction                  # screening counts and water-fraction checks
python -m pipeline.analysis.unmixing_psf                    # mixing correction and sensor consistency
python -m pipeline.analysis.placebo_design                  # placebo design and uniform-step injection
python -m pipeline.analysis.variability_timescale           # short-lag proxy and persistence time
python -m pipeline.analysis.validation_energy               # GeoNet comparison and dimensional scales
python -m pipeline.analysis.taal_reference_spatial --stage analyse
python -m pipeline.analysis.taal_reference_spatial --stage figure
python -m pipeline.analysis.robustness_table                # sensitivity summary
python -m pipeline.analysis.diagnostics thermometry         # heating-episode test
python -m pipeline.analysis.diagnostics diagnostics         # operating diagnostics and alert rates
python -m pipeline.analysis.figures                         # article figures
python -m pipeline.analysis.fig2_workflow                   # workflow figure
```

Numerical outputs are written to `results/` and figures to `figures/analysis/submission/`.

## Data

| File | Content | Source |
|---|---|---|
| `data/aoi/*.geojson` | Lake polygons (EPSG:4326) | This study, from Sentinel-2 |
| `data/raw/ground_thermal/ruapehu/tilde_*.csv` | Lake temperature of GeoNet station RU001 as downloaded from the Tilde service (telemetry and hand samples) | GeoNet |
| `data/raw/ground_thermal/ruapehu/geonet_converted.csv` | The same records in the long format read by the code | GeoNet |
| `data/interim/*_scene_inventory.csv` | Landsat Collection 2 Level-2 scenes over each lake with their clear fraction | Landsat 5, 7, 8, 9 |
| `data/interim/*_lst.csv` | Per-scene surface temperature in the lake polygon and background ring, water fraction | Landsat |
| `data/interim/*_water_extent.csv` | Sentinel-2 water extent of each lake | Sentinel-2 |
| `data/interim/*_donor_lst.csv` | Surface temperature of nearby water bodies used as reference donors | Landsat, JRC Global Surface Water |
| `data/processed/*_deltaT.csv` | Per-scene mixing-corrected lake temperature, reference temperature and screening flag | Derived |
| `data/processed/*_met.csv`, `*_dataset.csv` | ERA5-Land predictors at each overpass, and the scene table joined with them | ERA5-Land |
| `data/processed/*_energy_balance.csv` | Surface energy terms at each overpass | Derived |
| `data/imagery/` | True-colour and surface-temperature rasters, world relief image used in the maps | Sentinel-2, Landsat, ETOPO |
| `results/12_diagnostics/samples_buf/` | Per-sample water fraction and surface temperature (pandas pickles) | Landsat |

The satellite and reanalysis inputs were extracted with Google Earth Engine as described in the Methods of
the article. Times are UTC and temperatures are in kelvin unless a column name says otherwise.

### Licences and attribution of the source data

* Landsat Collection 2 Level-2 products: courtesy of the U.S. Geological Survey.
* Sentinel-2 imagery: contains modified Copernicus Sentinel data.
* Copernicus DEM GLO-30: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
* ERA5-Land: generated using Copernicus Climate Change Service information (Muñoz Sabater, 2019).
* Global Surface Water: Pekel et al. (2016), EC JRC/Google.
* GeoNet: lake temperatures of station RU001 are provided by GeoNet (https://www.geonet.org.nz) under
  the Creative Commons Attribution 3.0 New Zealand licence.
* Eruption onsets follow the Global Volcanism Program database of the Smithsonian Institution.

## Licence

The code is released under the MIT licence (`LICENSE`). Derived data may be reused with citation of
the article.

## Contact

Questions about the code and data: Hee-Su Joo (heesujoo03@gmail.com), or open an issue on this
repository.

Corresponding author of the article: Hyung-Sup Jung, Department of Geoinformatics, University of Seoul
(hsjung@uos.ac.kr).
