# Artemis region zones (v2)

Script: `tools/region_zones_v2.py` (about 10 s). The v1 script `tools/region_zones.py` (15 km radius circle on each anchor) is kept. Its last output is saved as `data/patches/artemis_region_zones_v1_radius15km.csv`. Re-running v1 overwrites `artemis_region_zones.csv` with the old schema.

Outputs
- `data/ml/moon_zones.csv`: one row per region x zone (`landing`, `ice`), 20 rows. The features use the same names as `data/ml/moon_tiles.csv`, so the tile models can score zones directly.
- `data/patches/artemis_region_zones.csv`: one summary row per region, including `placement_method`, `source` and `placement_note`.
- `data/patches/artemis_region_boxes.geojson`: the 15 x 15 km boxes used, for maps.
- `data/external/artemis3_regions_2022_moontrek.geojson`: NASA's official region outlines.

## Region placement

Every region is a 15 x 15 km box, axis-aligned in the LOLA south polar stereographic grid. That is the size NASA gives for the Oct 2024 regions.

| placement_method | regions | rule |
|---|---|---|
| `official` | Haworth, Malapert Massif, Nobile Rim 1, Nobile Rim 2, de Gerlache Rim 2 | Box centred on NASA's outline for the region of the same name. The outlines come from the NASA Moon Trek layer "Artemis III Candidate Landing Regions" (`A3_Named_regions`, 13 regions announced 19 Aug 2022), downloaded from https://trek.nasa.gov/moon/TrekWS/rest/cat/data/stream?uuid=82d31599-9957-4d79-b30a-720c918d1515 . These 5 names are also in NASA's Oct 2024 list of 9. The 2022 outlines are 20-21 km squares (Haworth 30 x 30 km). |
| `criteria_fallback` | Peak near Cabeus B, Slater Plain, Shackleton rim | No official outline was found and the anchor is a crater centre. The crater radius R is the rim crest of the azimuthal-median DEM profile. Box centres are searched in a ring 0.6-1.6 R from the crater centre. The chosen box holds the most km2 of sunlit flat ground (slope < 8 deg, visibility >= 0.3) that lies within 2 km of any permanently shadowed pixel. |
| `anchor` | Mons Mouton, Mons Mouton Plateau | Box centred on the anchor in `config/targets.csv` (summit; IM-2 plateau site). |

Notes on placement
- I found no machine-readable outlines for the Oct 2024 regions. I checked the Moon Trek search API (its only Artemis III layer is the 2022 one) and the NASA SVS API (only the 2022 visualization, SVS 5013). Peak near Cabeus B, Mons Mouton, Mons Mouton Plateau and Slater Plain were new in 2024.
- The 2022 "Leibnitz Beta Plateau" outline (Leibnitz Beta is now named Mons Mouton) is centred near 85.42S 31.7E. That is about 20 km from the Mons Mouton Plateau anchor and about 25 km from the Mons Mouton anchor. It is not used, because which 2024 region it became is not documented.
- Shackleton is not an Artemis III region. A 15 x 15 km box on the crater centre would lie entirely inside the crater (rim at R about 10.8 km), so Shackleton uses the rim criteria search instead of the anchor.
- DEM rim radii: Shackleton 10.8 km (literature about 10.5 km), Cabeus B 29.2 km, Slater 22.8 km. The Slater value is larger than expected and may mean the anchor is off-centre.

## Zones and features (80 m)

- `landing`: 80 m slope < 8 deg and average sun visibility >= 0.3. `landing_vis50_km2` gives the area at visibility >= 0.5.
- `ice`: average sun visibility == 0 (permanent shadow) inside the box.
- Terrain columns over the zone pixels:
  - `relief_m` is p95-p5, plus `elev_std_m`.
  - `slope500_*` and `tri500_mean_m` use the same 500 m block baseline as `extract.terrain_metrics`. They are computed on the whole box DEM, then sampled at the zone's 80 m pixels. Masking the DEM first would leave the 500 m blocks of a sparse zone empty.
- Diviner columns use the triangles whose centre falls in a zone pixel:
  - `x_temp_max_k` = max tmax (same definition as the tiles' `dvn_tmax_max_k`).
  - `x_temp_mean_k` = mean tavg.
  - `x_frozen_ground_frac` = ice stable within 1 m (`ice == -1` is treated as surface ice, depth 0).
  - `x_surface_ice_frac` = tmax < 110 K.
  - The mean tmax is also given, as `dvn_tmax_mean_k`.
- Visibility columns: `x_sunlit_frac_terrain` = mean visibility, `x_shadowed_frac` = PSR fraction.
- `x_grazing_sun_frac`, `x_noon_sun_max_deg` and `x_polar_night_frac` depend only on latitude and are computed at the zone centroid.
- `x_temp_min_k` is the pixel-weighted Diviner GCP Tmin of the overlapping 10 km tiles (`moon_patches_tmin.csv`). It is not resolved per zone, so the `x_temp_swing_k` difference between the two zones of a region comes from Tmax.
- Region-level links, repeated on both zone rows:
  - `landing_to_ice_min_km` and `landing_to_ice_median_km`: distance from each landing pixel to the region's own ice zone.
  - `best_landing_vis`.

## Per-region table

Temperatures are Diviner zone means, given as Tmax mean / Tavg in K. "Frozen" = ice-zone `x_frozen_ground_frac`; "cold trap" = ice-zone `x_surface_ice_frac` (tmax < 110 K).

| region | placement | box centre (lat, lon) | offset from anchor km | landing km2 | landing vis>=0.5 km2 | ice km2 | landing->ice min / median km | best vis | landing Tmax/Tavg | ice Tmax/Tavg | frozen | cold trap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Shackleton rim (not an Artemis III region) | criteria_fallback | -89.110, 119.27 | 17.2 | 51.8 | 1.9 | 9.4 | 0.08 / 0.82 | 0.78 | 214 / 106 | 113 / 55 | 0.97 | 0.70 |
| Peak near Cabeus B | criteria_fallback | -83.220, -44.70 | 40.3 | 134.2 | 0.0 | 1.5 | 0.08 / 1.37 | 0.40 | 235 / 125 | 206 / 102 | 0.75 | 0.00 |
| Haworth | official | -86.760, -22.78 | 31.6 | 57.4 | 1.1 | 4.5 | 0.08 / 1.29 | 0.69 | 228 / 119 | 119 / 58 | 0.97 | 0.61 |
| Malapert Massif | official | -85.990, -0.24 | 0.5 | 17.9 | 4.2 | 0.5 | 0.08 / 1.29 | 0.84 | 235 / 134 | 214 / 89 | 1.00 | 0.00 |
| Mons Mouton | anchor | -84.600, 31.00 | 0.0 | 103.2 | 1.8 | 7.2 | 0.08 / 1.83 | 0.54 | 229 / 126 | 175 / 69 | 0.91 | 0.22 |
| Mons Mouton Plateau | anchor | -84.790, 29.20 | 0.0 | 120.8 | 1.2 | 4.2 | 0.08 / 1.24 | 0.58 | 222 / 122 | 192 / 92 | 0.90 | 0.00 |
| Nobile Rim 1 | official | -85.440, 37.37 | 39.2 | 15.0 | 4.8 | 2.6 | 0.08 / 1.29 | 0.86 | 227 / 128 | 191 / 76 | 0.86 | 0.09 |
| Nobile Rim 2 | official | -83.950, 58.82 | 43.2 | 52.0 | 2.7 | 6.6 | 0.11 / 2.25 | 0.54 | 236 / 130 | 173 / 73 | 0.96 | 0.41 |
| de Gerlache Rim 2 | official | -88.230, -64.63 | 20.9 | 38.2 | 2.0 | 0.8 | 0.08 / 2.23 | 0.78 | 232 / 124 | 210 / 90 | 1.00 | 0.00 |
| Slater Plain | criteria_fallback | -88.830, 111.57 | 22.7 | 67.3 | 1.2 | 2.8 | 0.08 / 1.00 | 0.78 | 218 / 110 | 209 / 80 | 0.83 | 0.06 |

## Caveats

- An 8 deg slope at 80 m is a proxy for lander slope limits. Real touchdown limits apply at lander scale (a few metres), and boulders or small craters are invisible at 80 m.
- Visibility is a long-term average fraction of time the Sun is visible (LOLA 120 m). It is not the illumination on a mission date, and it says nothing about continuous-dark periods or Earth visibility.
- Small shadowed pockets inside rim boxes are mostly not true cold traps. Diviner cells are about 240-360 m across, so they mix shadowed pockets with their sunlit surroundings. Ice-zone Tmax means of 170-214 K (and the max-based `x_temp_max_k`) show this mixing. Prefer `x_frozen_ground_frac` and `x_surface_ice_frac`, and note that Diviner zone stats rest on few cells (2-73 per ice zone).
- The ice zones inside the boxes (0.5-9 km2) are far smaller than the v1 15 km circles (up to about 290 km2), which included whole crater floors. The landing-to-ice distances are to the region's own ice zone; larger PSRs just outside a box are not counted.
- Official boxes use 2022 outline centres. NASA refined the regions in 2024 and published no outlines that I could find. The criteria fallback is reproducible but is not NASA's region: for Shackleton it settled near the outer ring limit (1.6 R).
