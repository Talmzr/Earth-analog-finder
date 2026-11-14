# Anchors v2 (draft, 151 rows, all `needs_review=1`)

`config/anchors.csv` replaces the 47-row v1, which is kept as `config/anchors_v1.csv`. Nothing is reviewed yet.

**Columns.** `moon_target, moon_zone, earth_site, category, label, confidence, rationale, source, needs_review`.
* `moon_zone = tile`: `moon_target` is a `config/targets.csv` id (`moon_haworth`), i.e. the 10 km target tile.
* `moon_zone = landing | ice`: `moon_target` is a `region_id` from `data/ml/moon_zones.csv` (`haworth`, `nobile_rim_2`, ...).
* `earth_site`: a `patch_id` in `data/ml/earth_tiles.csv`. All 151 ids, targets and zones were checked by script. Labels also respect the score caps, the thermal PSR cap and the water cold-trap cap (0.3).
* `source` is plain text: an agency or program, or author and year. Where no source applies it says "expert judgement" and confidence is lower. "tile stats / tile data" means the check was made against this dataset (GLO-30/LOLA terrain, WorldCover, SoilGrids, MODIS LST, Macrostrat, GVP).

## Counts (label bands: strong neg <= -0.5 < middle < 0.5 <= strong pos)

| category | strong neg | middle | strong pos | total | note |
|---|---|---|---|---|---|
| geology | 7 | 7 | 8 | 22 | |
| terrain | 8 | 4 | 9 | 21 | positives chosen where tile statistics match AND the site is a known analog |
| thermal_illumination | 14 | 5 | **0** | 19 | no positives by design (see below) |
| water_ice | 6 | 8 | 6 | 20 | positives only vs zones/tiles without a surface cold trap |
| dust_regolith | 5 | 8 | 5 | 18 | 4 of the 5 positives are low confidence |
| biology_sterility | 7 | 5 | 6 | 18 | |
| caves_subsurface | 6 | 6 | **3** | 15 | only 3 tiles hold documented, analog-used lava tubes |
| ops_landing_mobility | 7 | 6 | 5 | 18 | |
| **all** | 60 | 49 | 42 | 151 | confidence: 44 high, 25 low, 82 med |

## Thermal scale decision

The scale is **absolute** and uses the tolerances in `categories.json` (60/60/40 K, 10 and 6 deg, 0.15). +0.5 means the same in every category: features within about 0.75 tau on average, e.g. a mean temperature within about 45 K and a noon sun within about 7.5 deg. No Earth surface meets this for any south-pole tile or zone. The best case, the high East Antarctic plateau against a sunlit landing zone, is about 1.5 tau away (kernel prior about -0.58). So **thermal has no positive anchors**, and thermal "sufficient" can only be reached in thermal-vacuum chambers.

Earth sites are still ranked by closest approach:

| Earth site vs Moon side | label |
|---|---|
| plateau or Concordia vs sunlit landing zones | -0.40 to -0.45 |
| other polar sites vs landing zones | -0.5 to -0.6 |
| anything vs PSR tiles or ice zones (the PSR cap also applies) | -0.7 to -0.75 |
| hot or tropical sites | -0.8 to -0.9 |

This resolves the dry-run conflict. The v1 anchors at +0.1 and 0.0 (Concordia vs Mons Mouton, Haughton vs Cabeus B) are now -0.5 and -0.6, and the v1 PSR anchors at -0.2 and -0.3 are now -0.7. Every thermal label now has the same order as tanh(kernel) and lies within about 0.3 of it. tanh(kernel) only spans -0.76 to +0.76, so the fitted slope `a` will be steeper than the prior.

If the team wants "the closest Earth gets = +0.5", publish that as a separate percentile rank. Do not use it to relabel these anchors: thermal >= 0.5 would then mean something different from every other category.

**Caves** are the second exception. The only Moon side is the Tranquillitatis pit. Strong positives are Lava Beds, Craters of the Moon and Lanzarote. Kilauea (0.45) and Undara (0.4) stay below 0.5 until a tube is confirmed inside the tile or the humidity and vegetation caveat is accepted.

## How to review (about 1 h for 2 people)

1. Sort by category. For each row, read the rationale's "Check:" numbers against the tile and either accept the label or edit it.
2. Set `needs_review=0` only after a person has agreed with both the label and the source. Lower confidence instead of deleting a row you doubt.
3. Labels are about the **10 km tile** (or zone), not the famous point site. Where the two differ (simulant quarries, Mauna Kea test pits, La Corona tube), the patch-level value is used and the rationale says so.
4. Re-run the kernel dry run. Each category needs `a > 0`, and the leave-one-out Spearman should be reported.

## Check first (largest label-vs-prior gaps or weakest facts)

1. `malapert_massif/ice x concordia` water -0.4 vs prior +0.70. **This is a kernel bug:** `surface_ice_frac` is NaN for Concordia because the stratum is `named_site` and Antarctica has no WorldCover. Use the `ice_sheet` context tag or `trav_ice_frac` as a fallback.
2. `moon_haworth x dallol` biology 0.2 vs -0.66. The table gives ~372 mm/yr precipitation for Dallol, which looks too high.
3. `haworth/landing x haughton` ops 0.2 vs -0.37. Zone `relief_m` (881 m) spans scattered flat pixels in a 15 km box, which inflates terrain mismatch for every landing zone.
4. `shackleton_rim/landing x concordia` thermal -0.4. This row sets the thermal ceiling; accept or change the scale decision here.
5. `pit x lanzarote` caves 0.5. The La Corona tube is ~35 km from the tile centre.
6. Dust positives on hyperarid basalt fields (`gvp_231060`, `gvp_231070`, `gvp_221080`, `gvp_332020`). These are expert judgement with no analog-program use. Mauna Loa's ~1740 mm/yr is a coarse island value.
7. `peak_near_cabeus_b/landing x mdrs` dust 0.2 vs 0.61, and Qaidam 0.1 vs 0.52. The kernel ranks clay or salt fines top because the tephra feature (tau 1.0) is weak.
8. `mons_mouton_plateau/landing x lava_beds` terrain 0.3 vs 0.71. The slope statistics match, but 35% tree cover is DSM canopy.
9. `moon_mons_mouton x crater_manicouagan` geology 0.6. This is a new strong positive (Macrostrat lithology: anorthosite), but the tile is 91% forest.
10. `haworth/landing x atacama_yungay` water -0.3 vs +0.17. The one-sided ice gap lets dry sites score positive against partly icy zones.

Other data notes:
* The Macrostrat lithology of named tiles can be wrong: Craters of the Moon reads "gneiss, granite" and Black Point reads "andesite, dacite, rhyolite".
* The Sudbury tile is basin-fill wacke, not norite.
