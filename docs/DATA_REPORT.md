# Data report: Earth analogs for permanent Moon base locations

NASA Space Apps 2026 · Earth ↔ Moon (Mars dropped) · data complete 14 Nov 2026 · **nothing trained yet**

Location: project folder `analog_project/` and the same layout in Google Drive `MyDrive/analog_project/`.
All sources are open, no login. Detail on sources: `DATA_SOURCES.md`. Distribution comparison: `docs/DATA_PROFILE.md`.

---

## 1. The big picture

| Unit of data | What one row is | Count | Why it exists |
|---|---|---|---|
| **Moon tile** | a 10 × 10 km square of the lunar south pole (76°S–90°S) | 3,609 (incl. 9 Artemis anchor tiles) | the "Moon side" sample for learning and for mapping the whole pole |
| **Earth tile** | a 10 × 10 km square on Earth centred on a candidate place | 4,310 | the "Earth side" search space: known analogs **and** never-studied places |
| **Moon zone** | the landing (sunlit, flat) or ice (permanently shadowed) part of one Artemis region | 20 (10 regions × 2) | the actual **targets** we score Earth tiles against |
| **Anchor** | an expert judgement "Moon target X vs Earth site Y, category C = label" | 47 (draft) | the only labels: calibrate what −1…+1 means |
| **Image patch** | 64 × 64 grid of elevation + slope for a tile (~156 m/px) | 3,609 Moon + 4,310 Earth | input for the self-supervised terrain model (DINOv2) |

Both bodies use the **same tile size (10 km)** and **the same terrain definitions** (slope at a 500 m baseline), so terrain numbers are directly comparable.
Columns starting with **`x_`** hold *the same physical quantity on both bodies* (e.g. `x_temp_min_k`), ready for Moon-vs-Earth comparison.

---

## 2. Files, shapes and sizes

| File | Rows × cols | Size | Use |
|---|---|---|---|
| `data/ml/moon_tiles.csv` | 3,609 × 76 | 2.0 MB | **main Moon table** (all Moon features merged) |
| `data/ml/earth_tiles.csv` | 4,310 × 142 | 6.4 MB | **main Earth table** (terrain + environment + soil + geology + traverse) |
| `data/ml/moon_zones.csv` | 20 × 56 | 10 KB | Artemis landing/ice zones with tile-compatible features: **scoring targets** |
| `data/ml/manifest.json` | — | small | row counts, shared columns (27), column→category map, per-column coverage |
| `data/patches/moon_patch_arrays.npz` | elev, slope, sunvis 3,609×64×64; dvn_tmax, dvn_tavg 3,609×16×16 | 154 MB | Moon images for SSL (Drive copy: elev + slope only, float16, 51 MB) |
| `data/patches/earth_patch_arrays.npz` | elev, slope, valid 4,310×64×64 | 121 MB | Earth images for SSL (Drive copy float16, 59 MB) |
| `data/patches/artemis_region_zones.csv` | 10 × 28 | small | one-row-per-region summary (areas, distances, temperatures, placement method) |
| `data/patches/artemis_region_boxes.geojson` | 10 boxes | small | region boxes for maps |
| `data/external/artemis3_regions_2022_moontrek.geojson` | 13 outlines | small | NASA's official Artemis III region outlines (Moon Trek) |
| `data/patches/earth_candidates.csv` | 4,321 × 7 | 1.2 MB | how every Earth tile was chosen (source, stratum, context text) |
| `data/patches/moon_patches.csv`, `moon_patches_tmin.csv`, `earth_patches_terrain.csv`, `earth_patches_env.csv`, `earth_traverse.csv` | — | — | intermediate tables (already merged into `data/ml/`) |
| `data/catalogs/moon_geologic_units.csv` | 49 × 10 | 14 KB | USGS lunar map units with **description text** (geology text model) |
| `data/catalogs/earth_impact_structures.csv` | 200 × 12 | 22 KB | Earth impact structures (candidate source) |
| `data/catalogs/earth_volcanoes_gvp.csv` | 1,214 × 21 | 1.9 MB | Smithsonian Holocene volcanoes incl. rock type and **geological summary text** |
| `config/anchors.csv` | 47 × 7 | 9 KB | draft expert labels (all `needs_review=1`) |
| `config/categories.json` | 8 categories | 41 KB | features, tolerances, labeling functions, caveats per category |
| `config/experiments.json` | 14 experiments | 13 KB | which categories each experiment needs, thresholds, "also account for" |
| `config/targets.csv`, `config/earth_sites.csv` | 10 × 10, 30 × 9 | small | the 10 Moon targets and 30 named Earth analog sites |
| `data/target_*.csv`, `data/earth_*.csv` | 14–30 rows | small | first-pass **site-level** tables (superseded by tiles; kept for reference) |
| `notebooks/earth_analog_ssl_colab.ipynb` | 34 cells | 70 KB | Colab notebook (setup cell + SSL + calibration + export). Not run beyond setup. |

---

## 3. Moon tiles: `data/ml/moon_tiles.csv` (3,609 × 76)

Coverage 100% for every column except `dvn_ice_depth_median_m` (89%: blank where no ice is stable).
Values below: min / median / max.

### Identity and position
| Column | Meaning |
|---|---|
| `patch_id` | `mp_<x km>_<y km>` grid tile, or the target id (`moon_haworth` …) |
| `body` | `moon` |
| `center_lat`, `center_lon` | tile centre (−89.8 … −76.3°) |
| `x_m`, `y_m` | polar-stereographic coordinates (m) |
| `is_target` | 1 for the 9 Artemis anchor tiles |

### Terrain: LOLA 80 m DEM (category: terrain, operations)
| Column | Meaning | Range |
|---|---|---|
| `relief_m` | elevation spread (95th − 5th percentile) | 37 / 925 / 4,321 m |
| `elev_mean_m`, `elev_std_m` | mean height, height variability | |
| `slope500_mean_deg` | mean slope at a 500 m baseline | 0.6 / 7.9 / 24° |
| `slope500_p90_deg` | slope that 90% of the ground is below | 1 / 13.9 / 32° |
| `frac_slope500_gt10`, `_gt20` | share of ground steeper than 10° / 20° | |
| `tri500_mean_m` | roughness (terrain ruggedness index) | 5 / 52 / 169 m |
| `slope_native_mean_deg`, `native_res_m`, `valid_frac` | slope at native 80 m, resolution, valid share | |

### Real sunlight with terrain shadows: LOLA/PGDA (category: illumination, operations)
| Column | Meaning |
|---|---|
| `lola_avg_sun_vis_mean`, `_p90`, `_max` | share of time the sun is visible (0–1), mean / 90th pct / best pixel |
| `lola_frac_psr` | share of the tile in **permanent shadow** |
| `lola_frac_sunlit_gt50pct` | share of the tile lit more than half the time |

### Geometric sunlight (latitude only): category illumination
`noon_elev_max/min/mean_deg`, `frac_polar_day`, `frac_polar_night`, `mean_day_elev_deg`, `frac_daylight`.
⚠️ These are functions of latitude only (several are duplicates, r = 1.0; `frac_daylight` is constant 0.5). Use the LOLA columns instead.

### Temperature and ice: Diviner (category: temperature, water/ice)
| Column | Meaning | Range |
|---|---|---|
| `dvn_tavg_mean_k` | average surface temperature | 38 / 130 / 183 K |
| `dvn_tmax_mean_k`, `dvn_tmax_max_k` | summer maximum temperature (mean / hottest cell) | max 72 / 300 / 349 K |
| `dvn_tavg_min_k`, `dvn_tmax_min_k` | coldest cells in the tile | |
| `dvn_tbol_min_k` | **night minimum** temperature (Diviner global product) | 24 / 60 / 80 K |
| `dvn_tbol_max_k`, `_mean_k`, `_noon_k`, `_midnight_k` | full day/night cycle | |
| `dvn_tbol_ltime_cov`, `_n_cells`, `_nearest_km` | quality: local-time coverage, cells used, distance to nearest cell | |
| `dvn_frac_coldtrap_110k` | share colder than 110 K at its hottest = **surface water ice can survive** | 0 / 0 / 1 |
| `dvn_frac_ice_stable` ≈ `dvn_frac_ice_within_1m` | share where ice is stable in the top ~1 m | 0 / 0.23 / 1 |
| `dvn_frac_ice_surface` | ice stable right at the surface | |
| `dvn_ice_depth_median_m` | median depth to stable ice | 0 / 0.16 / 0.99 m |
| `dvn_n_cells` | Diviner cells averaged (~780) | |

### Geology: USGS Unified Geologic Map of the Moon (category: geology)
| Column | Meaning |
|---|---|
| `moon_geo_unit`, `_name`, `_age` | majority map unit (e.g. `pNb` pre-Nectarian basin material) |
| `moon_geo_unit_frac`, `moon_geo_n_units` | how dominant the unit is; number of units in tile |
| `geo_frac_crater/basin/plains/terra/mare` | share by unit class (mare = 0 everywhere at the pole) |
| `geo_frac_age_*` | share by age (Copernican … pre-Nectarian) |

### Cross-body columns (`x_*`): same quantity as on Earth
| Column | Moon source | Range |
|---|---|---|
| `x_temp_max_k` | Diviner max | 72 / 300 / 349 K |
| `x_temp_mean_k` | Diviner average | 38 / 130 / 183 K |
| `x_temp_min_k` | Diviner night minimum | 24 / 60 / 80 K |
| `x_temp_swing_k` | max − min | 43 / 241 / 297 K |
| `x_grazing_sun_frac` | share of daylight with sun < 12° | 0.70 / 1.0 / 1.0 |
| `x_noon_sun_max_deg` | highest noon sun | 1.8 / 9.4 / 15° |
| `x_polar_night_frac` | share of year in polar night | |
| `x_frozen_ground_frac` | ice stable within 1 m | |
| `x_surface_ice_frac` | cold-trap share | |
| `x_sunlit_frac_terrain`, `x_shadowed_frac` | Moon-only: terrain-aware sunlight / permanent shadow | |

---

## 4. Earth tiles: `data/ml/earth_tiles.csv` (4,310 × 142)

### Where the tiles come from (`source`, `stratum`)
| Source | Tiles | Strata (why chosen) |
|---|---|---|
| `named_site` | 30 | known/proposed analogs from `config/earth_sites.csv` |
| `impact_structure` | 190 | Earth impact craters (crater terrain, shock geology) |
| `volcano` | 1,091 | Holocene volcanoes: strato/dome 691, field/cone 210, shield 107, caldera 78 (basalt, lava tubes, rugged terrain) |
| `landcover_sample` | 2,999 | random barren-type land: bare_hot 450, antarctic_ice_free 450, ice_edge 400, bare_midlat 400, lichen_moss 300, sparse_hot 300, sparse_cold 299, bare_polar 250, ice_sheet 150 |

The random samples are how the project "looks beyond known sites" (challenge requirement).

### Identity
`patch_id`, `body`, `source`, `stratum`, `name`, `context` (free text: tags, crater info, volcano type + rock + summary; 93%), `center_lat/lon`, `lat/lon`, `hw_km` (=5).

### Terrain: Copernicus GLO-30 DEM read at ~78 m (100%)
Same columns and meanings as the Moon: `relief_m` (0 / 243 / 2,630 m), `slope500_mean_deg` (0 / 2.8 / 27°), `slope500_p90_deg`, `frac_slope500_gt10/20`, `tri500_mean_m`, `elev_*`, plus `elev_min/max/range_m`, `frac_elev_zero` (sea/lake-flat share), `dem_fetch`, `n_dem_tiles`.

### Sunlight geometry (100%)
`noon_elev_*`, `frac_polar_day/night`, `mean_day_elev_deg`, `frac_daylight`, and **`noon_elev_m01…m12`**: noon sun height per month, used to find the Earth "light-twin month" closest to lunar-pole light.

### Land cover: ESA WorldCover 10 m (86%; none south of 60°S)
`wc_frac_tree/shrub/grass/crop/built/bare_sparse/snow_ice/water/wetland/mangrove/moss_lichen`, `wc_valid_frac`, `wc_note`.
Use: bare-ground share (geology/dust), vegetation and built-up (biology, ops caps), snow/ice (water/ice).

### Surface temperature: NASA MODIS LST monthly normals via JAXA (99%, 0.05°)
| Column | Meaning | Range |
|---|---|---|
| `lst_day_max_k` | hottest-month daytime surface temp | 238 / 296 / 333 K |
| `lst_night_min_k` | coldest-month night surface temp | 201 / 258 / 301 K |
| `lst_mean_k`, `lst_day_mean_k`, `lst_night_mean_k` | annual means | |
| `lst_diurnal_range_k`, `lst_annual_range_k` | day–night and seasonal swings | |
| `lst_*_months_valid`, `lst_res_deg` | quality | |

### Climate: NASA POWER 2001–2025 climatology (67%: API throttled, nearest-cell values)
`power_t2m_c` (air temp, −51 / −0.6 / 34 °C), `power_t2m_min/max_c`, `_coldest_month_min_c`, `_warmest_month_max_c`, `_annual_range_c`, `_diurnal_range_c`, `power_frost_days` (0 / 239 / 365), `power_rh2m_pct`, `power_ws2m_ms` (wind), `power_ps_kpa` (pressure), `power_precip_mm_yr`, `power_sw_allsky/clrsky_kwh_m2_d` (solar energy), `power_uv_index`, `power_albedo`, `power_cell_lat/lon`.

### Permafrost: ESA CCI Permafrost (42% real values; Northern Hemisphere domain)
`pf_frac_pct` (permafrost share), `pf_gst_c` (ground surface temp), `pf_magt_2m_c`, `pf_magt_10m_c` (ground temp at 2 m / 10 m), `pf_in_cci_domain`, plus `permafrost_proxy`/`permafrost_flag`/`permafrost_src` (air temp < −2 °C proxy where CCI has no data; 67–73%).

### Water and dryness: JAXA (partner agency)
`soil_moist_mean/max` (AMSR2 soil moisture %, 77%), `precip_mm_yr` + `precip_wettest_month_mm` (GSMaP/GPM, 65%, 60°S–60°N), **`precip_best_mm_yr`** (best available of GSMaP/POWER, 93%) + `precip_best_src`, `*_widened_25km` flags.

### Vegetation and dust: MODIS via JAXA (category: biology, dust)
`ndvi_mean/max` (vegetation greenness 0–0.9, 87%), `aot_mean/max` (aerosol optical thickness = dust/haze load, 91%).

### Soil / regolith-like material: ISRIC SoilGrids (81%, 5 km; none in Antarctica)
`sg_sand_pct`, `sg_silt_pct`, `sg_clay_pct` (grain size), `sg_bdod_g_cm3` (bulk density), `sg_cfvo_pct` (coarse fragments).
Use: compare with lunar regolith (fine-grained, ~1.5 g/cm³), dust behaviour, excavation.

### Bedrock geology: Macrostrat (90–96%)
`mstrat_unit`, `mstrat_lith` (e.g. "basalt", "anorthosite"), `mstrat_age`, `mstrat_t_age_ma/b_age_ma`, `mstrat_descrip` (54%), `mstrat_text_all`, `mstrat_scale`.
Use: geology category and the **text-embedding model** (compared with lunar unit descriptions).

### Landing → ice traverse (100%; fast local method, ~1–3 km resolution)
| Column | Meaning |
|---|---|
| `trav_flat_bare_frac` | share of the tile that is flat (< 8°) bare ground (a "landing pad") |
| `trav_ice_frac` | share of the ±10 km window covered by ice/snow |
| `trav_dist_flat_to_ice_km`, `_min_` | median / min distance from flat bare ground to ice (1…10 km; 10 = none) |
| `trav_dist_gt10_flag` | 1 = no ice within 10 km |
| `trav_source`, `trav_method` | C3S 2.8 km land cover, or Antarctic rock mask 1.4 km |

Use: Earth version of the lunar "land on a sunlit ridge, drive a few km to the cold trap" (715 tiles have ice ≤ 3 km from flat bare ground).

### Cross-body columns (`x_*`): Earth side
`x_temp_max_k` (= LST day max), `x_temp_mean_k`, `x_temp_min_k` (= LST night min), `x_temp_swing_k`, `x_grazing_sun_frac` (0.14 / 0.21 / 0.43), `x_noon_sun_max_deg`, `x_polar_night_frac`, `x_frozen_ground_frac` (permafrost or proxy), `x_surface_ice_frac` (WorldCover snow/ice).

---

## 5. Moon zones: `data/ml/moon_zones.csv` (20 × 56) and `artemis_region_zones.csv` (10 × 28)

Each Artemis region is a 15 × 15 km box split at 80 m resolution into:
- **landing zone**: slope < 8° and sunlit ≥ 30% of the time
- **ice zone**: permanently shadowed

Columns: the same terrain / sunlight / Diviner / `x_*` names as the tiles, plus `region_id`, `zone`, `area_km2`, `placement_method`, `source`, `landing_to_ice_min/median_km`, `best_landing_vis`, `landing_vis50_km2`.

| Region | Placement | Landing km² | Ice km² | Landing→ice median |
|---|---|---|---|---|
| Haworth, Malapert Massif, Nobile Rim 1, Nobile Rim 2, de Gerlache Rim 2 | **official NASA outline** (Moon Trek 2022 layer) | 15–57 | 0.5–6.6 | 1.3–2.3 km |
| Shackleton rim, Peak near Cabeus B, Slater Plain | criteria search on the rim | 52–134 | 1.5–9.4 | 0.8–1.4 km |
| Mons Mouton, Mons Mouton Plateau | on the anchor | 103–121 | 4–7 | 1.2–1.8 km |

**Use:** these 20 rows are what each Earth tile is finally scored against: landing experiments vs the landing zone, ice/ISRU experiments vs the ice zone, traverse experiments vs both + the distance.

---

## 6. Labels: `config/anchors.csv` (47 rows, draft)

Columns: `moon_target`, `earth_site`, `category`, `label` (−0.9 … +0.7), `confidence` (13 high / 22 medium / 12 low), `rationale`, `needs_review` (all 1).
Per category: geology 11, terrain 6, thermal 6, water/ice 6, dust 5, biology 5, caves 4, operations 4.
**Use:** calibrate the −1…+1 scale and evaluate the models. **Must be reviewed by the team before training.**

---

## 7. Categories → data → model (from `config/categories.json`)

| Category | Main inputs | Model (planned) | Notes |
|---|---|---|---|
| Terrain | elevation/slope images + slope/relief/roughness | **DINOv2-small** (Hugging Face) fine-tuned self-supervised on Moon+Earth patches | 86–100% value overlap: direct similarity works |
| Thermal & illumination | `x_temp_*`, `x_grazing_sun_frac`, `x_noon_sun_max_deg` | physics kernel with tolerances (e.g. 60 K) + anchor calibration | **no overlap** for mean/min temp and sun angle: "closest approach" score; Earth best ≈ −0.4 |
| Water / ice | `x_frozen_ground_frac`, `x_surface_ice_frac`, traverse, permafrost, soil moisture | physics kernel + anchors | one-sided: more ice on Earth isn't penalised |
| Geology | Moon unit text ↔ Macrostrat/volcano/crater text, SoilGrids | **bge-small** text encoder (Hugging Face), contrastive fine-tune | e.g. Mistastin (anorthosite) ranks top for highland targets |
| Dust / regolith | SoilGrids grain size, aerosol, bare share | kernel vs a fixed lunar-regolith reference | low weight (no per-tile Moon dust data) |
| Biology / sterility | NDVI, precipitation, soil moisture, UV, tree share | Earth-only ranking, capped at 0.6 | Moon is sterile; ranks how lifeless the Earth site is |
| Caves / subsurface | volcano type, tags | simple rule (no model) | only the Tranquillitatis pit target |
| Landing / ops | slopes, roughness, sunlight, temperature swing, traverse | derived from the above | Earth-side caps for forest/towns |

Output (when trained): `results/category_scores.parquet` with `moon_id, earth_id, category, score ∈ [−1, 1]`.

## 8. Experiments (`config/experiments.json`)

14 experiment types, each requiring minimum category scores and listing "also account for" caveats:
landing hazard detection · rover mobility & navigation · ice prospecting instruments · ISRU ice extraction · regolith excavation & construction · thermal/power in extreme cold & long night · illumination & power planning · habitat & human ops/isolation · dust mitigation · spectrometer/mineralogy field tests · GPR/drill subsurface tests · planetary protection / contamination blanks · cave & pit robotics · geology EVA training.
Decision rule: score ≥ 0.5 = sufficient · 0–0.5 = partial (account for …) · < 0 = avoid.
Always attached (Earth can't reproduce): 1/6 gravity, vacuum, radiation, permanently-shadowed 25–110 K temperatures, grazing sun.

---

## 9. Known limitations

1. **No ground truth** for "how good an analog is": the scale is set by 47 draft anchors + stated tolerances.
2. **Temperature and sun angle never overlap** between Moon and Earth (Moon night 24–80 K vs Earth ≥ 201 K; sun always < 15° vs Earth ≥ 24°). Scores there mean "closest Earth gets", not "similar".
3. **Earth climate is coarser** than the tiles (POWER ~50 km, LST 5 km, SoilGrids 5 km); POWER covers only 67% of tiles.
4. **Antarctica gaps**: no WorldCover, SoilGrids or GSMaP south of 60°S (masked, not dropped; Antarctic rock mask used for traverse).
5. **Ice zones are small** (0.5–9 km²) and Diviner cells (~240–360 m) blur them; only Shackleton, Haworth and Nobile Rim 2 are clearly colder than 110 K.
6. **Traverse distances** are ~1–3 km resolution (fast method).
7. **Artemis 2024 regions** have no published outlines; 2022 outlines used where names match; 5 regions placed by criteria or anchor.
8. **Neighbouring tiles are correlated**: evaluate on held-out spatial blocks, not random tiles.
9. Earth cannot reproduce gravity, vacuum or radiation, so these are always reported as caveats, never scored.
