# Moon south-polar patch dataset

Built by `tools/build_moon_patches.py` (about 30-65 s on 8 cores; it skips the build if the outputs exist, `--refresh` rebuilds).

## Patches
- **2,901 rows**: 2,892 grid patches + 9 target patches (`is_target=1`, `patch_id` = target id from `config/targets.csv`, body = moon, lat <= -80).
- Grid: 10 km x 10 km squares, stride 10 km, in the LOLA DEM's south polar stereographic metres (`+proj=stere +lat_0=-90 +lat_ts=-90 +R=1737400`). The grid spans x, y in [-300, 300] km, and a patch is kept when its centre lat is <= -80 (centre radius <= ~304 km). `patch_id` = `mp_<x_km>_<y_km>` of the centre, for example `mp_-295_+105`.
- Target patches are centred on the target coordinate with the same 10 km size. The DEM window is snapped to the nearest 80 m pixel edge (shift <= 40 m). Target patches overlap grid patches.
- The DEM is 100% valid in every kept patch, so no patch was dropped by the >= 80% valid rule.
- `x_m`, `y_m` = patch centre (polar stereographic m). `center_lat` and `center_lon` are on a sphere of R = 1737.4 km, lon in -180..180.

## Columns (`moon_patches.csv`)
| group | columns | notes |
|---|---|---|
| terrain (`extract.terrain_metrics`) | native_res_m, valid_frac, elev_mean_m, elev_std_m, relief_m (p95-p5), slope_native_mean_deg, slope500_mean_deg, slope500_p90_deg, frac_slope500_gt10, frac_slope500_gt20, tri500_mean_m | LOLA LDEM_80S_80MPP_ADJ (80 m/px, 125x125 px per patch). Ground spacing is corrected for the stereographic scale factor (79.4-80 m). Elevation is m relative to the 1737.4 km sphere. Slope uses a 500 m baseline, the same as for the Earth sites. |
| LOLA illumination | lola_avg_sun_vis_mean, _p90, _max, lola_frac_psr (vis <= 0), lola_frac_sunlit_gt50pct | AVGVISIB_75S_120M_201608 (Mazarico et al.), the average fraction of the solar disk visible, with topographic shadowing (0..1). Uses every 120 m pixel whose centre is inside the patch. |
| geometric illumination (`extract.illumination(lat,"moon")`) | noon_elev_max_deg, noon_elev_min_deg, noon_elev_mean_deg, frac_polar_day, frac_polar_night, mean_day_elev_deg, frac_daylight | Latitude and obliquity only, no topography. |
| Diviner PRP (south) | dvn_tavg_mean_k, dvn_tavg_min_k, dvn_tmax_mean_k, dvn_tmax_max_k, dvn_tmax_min_k, dvn_frac_coldtrap_110k (tmax < 110 K), dvn_frac_ice_stable, dvn_frac_ice_within_1m, dvn_frac_ice_surface, dvn_ice_depth_median_m (over ice-stable cells; NaN in 182 patches with no stable cells), dvn_n_cells | Triangles whose centre falls in the patch, 760-800 per patch (~240 m triangles). |
| geology (USGS Unified Geologic Map of the Moon, 64 ppd raster) | moon_geo_unit (majority code), moon_geo_unit_name, moon_geo_unit_age, moon_geo_unit_frac, moon_geo_n_units, geo_frac_crater / _basin / _plains / _terra / _mare, geo_frac_age_copernican / _eratosthenian / _imbrian / _nectarian / _prenectarian, geo_valid_frac | Sampled on a 32x32 grid (312 m) per patch. Classes come from the unit names in the raster's DN table. Basin includes the basin "Formation" units. Unit text is in `data/catalogs/moon_geologic_units.csv`. |

**Diviner ice_depth coding.** The PDS label says -999 means ice is not stable within 2.87 m, and 0 means ice is stable at the surface. The table also contains **-1**, which the label does not explain. All 100,319 cells coded -1 have tmax 27-101 K, so they are the coldest cold traps. This dataset treats -1 as ice stable at the surface (depth 0). (`extract_extra.g_diviner_polar` counts -1 as not stable, so it underestimates `dvn_frac_ice_stable` and `dvn_frac_ice_surface` there.)

## Arrays (`moon_patch_arrays.npz`, same row order as the CSV)
Keys and shapes (float32 unless noted):

- `patch_id` (str)
- `center_lat`, `center_lon`
- `elev` (N,64,64): DEM bin-averaged to 156.25 m, minus the patch mean (m).
- `slope` (N,64,64): degrees, computed from the 156.25 m grid.
- `sunvis` (N,64,64): LOLA average sun visibility (0..1), bin-averaged from 120 m pixels.
- `dvn_tmax`, `dvn_tavg` (N,16,16): K, mean of the Diviner triangles in each 625 m bin; NaN where a bin has no triangle (none occur).

Row 0 is the top edge (max y, i.e. towards +y in the projection), and column 0 is the min-x edge. All grids share this orientation, which was checked by correlating sunvis with Diviner tavg per pixel: r = 0.76 as stored, against <= 0.11 for any flipped version.

## Sources (local copies in `~/.cache/earth_analog_raw`)
- LOLA DEM: https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF
- LOLA illumination: https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_75S_120M_201608.TIF
- Diviner PRP south: https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_prp/dlre_prp_south.tab, converted to `dlre_prp_south.npz`
- USGS Unified Geologic Map of the Moon: the raster zip (`unified_geologic_map_of_the_moon_raster.zip`, 64 ppd numerical raster + DN table) and the DMU unit descriptions, range-read from `Unified_Geologic_Map_of_the_Moon_GIS_v2.zip`

## Checks and coverage gaps
- **Pipeline check.** Running the same code on a 30 km box around Shackleton reproduces the earlier target extraction: relief 3,737 m (3,740 before), PSR fraction 0.278 (0.278), mean sun visibility 0.195 (0.195).
- **Shackleton at 10 km.** The 10 km patch lies entirely inside the crater: relief 1.6 km, PSR fraction 1.0, Diviner tavg 53 K, tmax max 108 K. Rim and sunlit-peak sites need neighbouring grid patches or a larger window.
- **Coverage.** All 2,901 patches have DEM, LOLA illumination, Diviner (760-800 cells) and geology data. Patches between -80 and ~-76.3 in the DEM square's corners are excluded by default; `--max-lat -76` adds about 700 more.
- **Geology resolution.** The geology map is 1:5M (64 ppd raster), coarse compared with 10 km patches, and the majority unit covers the whole patch in most of them. geo_frac_mare is 0 everywhere at the pole.
- **Illumination epoch.** LOLA sun visibility is a multi-year average (2016 product), and the geometric illumination ignores topography.
