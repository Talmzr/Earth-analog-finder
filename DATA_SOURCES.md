# Data sources: what we pull, from where, and what we skipped

Two extraction scripts, one row per site:

| Script | Output | Sites |
|---|---|---|
| `extract.py` | `data/target_features.csv`, `data/earth_features.csv` | terrain, geometric sunlight, Earth climate |
| `extract_extra.py` | `data/target_extra.csv`, `data/earth_extra.csv` | everything below |

Join the two tables on `id`. No login or API key is needed for any source.

## Moon: NASA LRO (from the challenge's PDS / LRO / Diviner links)

| Columns | Dataset | What it adds |
|---|---|---|
| `relief_m`, `slope500_*`, `tri500_mean_m` | LOLA south-pole DEM 80 m (PGDA) | terrain (extract.py) |
| `dvn_tavg_k_*`, `dvn_tmax_k_*` | **Diviner Polar Resource Product (south)**, PDS | measured mean and max surface temperature on ~0.5 km cells |
| `dvn_frac_coldtrap_110k` | same | share of the site cold enough (<110 K max) to trap water ice at the surface |
| `dvn_frac_ice_stable`, `dvn_frac_ice_surface`, `dvn_frac_ice_within_1m`, `dvn_ice_depth_median_m` | same | where subsurface ice is thermally stable, and how deep |
| `dvn_tbol_max_k`, `_min_k`, `_mean_k`, `_noon_k`, `_midnight_k`, `_range_k` | **Diviner Global Cumulative Product**, PDS | full day/night temperature cycle (96 local-time bins), all Moon sites |
| `dvn_rock_abund_*`, `dvn_regolith_temp_anom_k_*` | **Diviner rock abundance + regolith temperature** (Bandfield 2011), PDS | rockiness and regolith compaction (within 60° of the equator, so the Tranquillitatis pit only) |
| `lola_avg_sun_vis_*`, `lola_frac_psr`, `lola_frac_sunlit_gt50pct` | **LOLA average solar visibility** (Mazarico 2011), PGDA | real sunlight **including crater and peak shadows**, plus the permanently shadowed fraction. Fixes the old "illumination ignores topography" limitation |
| terrain columns for `moon_tranquillitatis_pit` | LOLA global DEM 118 m (USGS) | fills the pit's missing terrain |

## Mars: NASA MGS / Odyssey / MRO (from the PDS link)

| Columns | Dataset | What it adds |
|---|---|---|
| `relief_m`, `slope500_*` … | MOLA DEM 463 m (USGS) | terrain (extract.py) |
| `tes_ti_day_*`, `tes_ti_night_*` | **TES thermal inertia** (Putzig & Mellon 2007), PDS | dust vs sand vs rock / ice-cemented ground |
| `tes_albedo_*` | **TES albedo**, USGS | surface brightness / dust cover |
| `swim_ci_0_1m_*`, `swim_ci_1_5m_*`, `swim_ci_gt5m_*` | **SWIM 2.0 subsurface ice consistency**, NASA JPL / PSI | ice likely within 1 m, 1–5 m, >5 m (−1 to +1) |
| `ez_nearest`, `ez_dist_km`, `ez_inside` | **NASA Human Landing Site Exploration Zones** (2015), USGS | whether the target is in, or how far from, a NASA-proposed human landing zone |
| `mcd_tsurf_*`, `mcd_psurf_*` | Mars Climate Database v6 (ESA/CNES/LMD, *complement*) | surface temperature over season and day, surface pressure |

## Earth: NASA Earthdata + partner agencies

| Columns | Dataset | Agency | What it adds |
|---|---|---|---|
| `relief_m`, `slope500_*` … | Copernicus DEM GLO-30 | ESA | terrain (extract.py) |
| `ndvi_*`, `lst_*` | MODIS MOD13Q1, MOD11A2 (ORNL API) | NASA | vegetation, land surface temperature (extract.py) |
| `t_*`, `precip_mm_yr` … | ERA5 via Open-Meteo | ECMWF | 10-yr climate (extract.py) |
| `power_*` | **NASA POWER** climatology 2001–2025 | NASA | air temperature extremes, frost days, humidity, wind, pressure, solar + UV dose, **surface albedo** |
| `modis_pct_bare`, `modis_pct_tree`, … | **MODIS MOD44B** vegetation continuous fields | NASA | % bare ground |
| `modis_lc_*` | **MODIS MCD12Q1** land cover | NASA | IGBP class, % barren, % snow/ice |
| `wc_frac_*` | **ESA WorldCover 10 m** | ESA | fractions of bare, snow/ice, water, moss/lichen … (no coverage south of 60°S) |
| `jaxa_gsmap_*` | **JAXA GSMaP** precipitation normals (GPM) | JAXA/NASA | independent precipitation check (60°S–60°N) |
| `jaxa_amsr2_*` | **JAXA AMSR2** soil-moisture normals | JAXA | how dry the ground is |
| `mstrat_*` | Macrostrat geologic map (*complement*) | — | bedrock unit, lithology, age. Replaces hand-made tags with mapped geology |

## Cross-body columns (`x_*` in the extra tables)

The same physical quantity from each body's best instrument, so it can be compared directly:

| Column | Moon | Mars | Earth |
|---|---|---|---|
| `x_surf_temp_max_k` / `_min_k` / `_mean_k` | Diviner | Mars Climate Database | MODIS LST |
| `x_albedo` | — | TES | NASA POWER |
| `x_sun_visibility` | LOLA (with terrain shadows) | — | geometric daylight fraction |

## Looked at but not used, and why

| Source (challenge resource list) | Why not |
|---|---|
| NASA Earthdata Search (SMAP, GPM IMERG, ASTER …) | Direct download needs an Earthdata Login. Covered instead by the no-login ORNL MODIS API, NASA POWER and JAXA (GPM/AMSR2). If you create an Earthdata account, SMAP L3 soil moisture is the next layer to add. |
| NASA Worldview | Imagery viewer, not a data API. Good for pitch screenshots. |
| CSA RADARSAT-1 / RCM | Needs an EODMS account, and C-band radar isn't directly comparable to the Moon's Mini-RF (S-band). Phase-2 idea. |
| Copernicus Data Space (Sentinel) | Needs a login. ESA products used instead via public buckets (Copernicus DEM, WorldCover). |
| LOLA 10 ppd albedo mosaic (USGS) | Mostly zero-filled. Its values were not trustworthy. |
| LRO LEND / Mini-RF, Odyssey GRS hydrogen | Coarse (tens of km) or heavy SAR processing. Diviner ice stability and SWIM already cover "where is the ice". |
| Diviner north-pole products | All Moon targets are at the south pole or the equator. |

## Practical notes

- `extract_extra.py` caches each site in `data/cache/extra_*.json`. Re-running only fills gaps, and `--refresh` redoes everything.
- Diviner tables are range-read (only the rows each site needs), so nothing large is downloaded. Small files (TES albedo, EZ shapefile) go to `~/.cache/earth_analog_raw`, outside OneDrive.
- `python extract_extra.py --check` tests every source in about a minute.
