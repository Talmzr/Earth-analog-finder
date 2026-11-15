# Handoff: Earth Analog Finder (NASA Space Apps 2026)

Final state as of **2026-11-15 ~12:00**. Read this first, then `docs/ML_RESULTS.md` (generated, current numbers) and `docs/DATA_REPORT.md`.
Repo: https://github.com/Talmzr/Earth-analog-finder (public, branch `main`). Project root = repo root (the folder was `analog_project/` locally).
**Live website:** https://talmzr.github.io/Earth-analog-finder/ (deployed from `web/` by `.github/workflows/pages.yml` on every push to `main`).
Everything described here is committed and pushed (latest: `5769d5e`).

## 1. What the project is

Challenge: "Identify Earth locations that analog the permanent Moon base locations". **Earth + Moon only (Mars was dropped).**
We score every (Moon target, Earth tile) pair per category on a -1..+1 scale, then turn scores into experiment advice
(>= 0.5 sufficient, 0..0.5 partial, < 0 avoid, plus "also account for" caveats).

- **Moon side:** 30 scoring targets = 20 Artemis zones (10 regions x landing zone / ice zone) + 10 target tiles (9 south-pole anchors + Tranquillitatis pit).
- **Earth side:** 4,310 tiles of 10 x 10 km (30 named analog sites, 190 impact structures, 1,091 volcanoes, ~3,000 random barren / polar / ice-edge / lichen samples).
- **Categories (9):** terrain, geology, thermal (temperature), illumination, water_ice, dust, biology, caves, operations. Illumination was split out of thermal today.
- **User priorities:** MVP first, finetune pretrained models (never train from scratch), do not polish things that are not the MVP,
  heavy compute on Colab T4, light work local.

## 2. Final state (what is DONE)

| Area | State | Where |
|---|---|---|
| Data tables | Done and validated | `data/ml/moon_tiles.csv` (5,610 x 95), `earth_tiles.csv` (4,310 x 143), `moon_zones.csv` (20 x 56), `moon_zone_patch_terrain.csv`, `manifest.json`. Dictionary: `docs/DATA_REPORT.md` |
| Terrain image patches | Done, 128 x 128 (78 m/px), stored as <= 94 MB parts | `data/arrays/` + `python tools/reassemble_arrays.py` |
| Anchors (labels) | 161 drafts, **all `needs_review=1`, no human review yet** (includes 10 geometry-derived illumination drafts) | `config/anchors.csv` (guide: `docs/ANCHORS.md`) |
| Terrain model | Fine-tuned DINOv2-small (Colab T4). **Trained but not used in the score** (adds nothing over measured relief/slope/roughness) | `models/terrain_dinov2_ssl.pt`, `embeddings/*_terrain_384.npy` |
| Geology model | Fine-tuned BGE-small (Colab T4), **not retrained after the anorthosite fix** (see 3.2); 45 geology texts re-embedded with the existing model | `models/geology_bge_ssl.pt`, `embeddings/*_geology_384.npy`, `tools/reembed_geology.py` |
| Scoring + calibration | Done, final run | `ml/score_all.py` (`KEEP_TERMS`, `COMPOSITE_CATS`, `CAPS`, `GEO_MINOR_CAP`), `models/calibration_params.json`, `results/` |
| Website | **Live on GitHub Pages** | `web/index.html`, `web/data/` (from `ml/export_web.py`) |
| Results write-up | Regenerated from final run by a script | `docs/ML_RESULTS.md` via `tools/make_results_doc.py` |

### Changes made today (2026-11-15)

1. **Anorthosite rule.** An Earth tile is anorthositic only if anorthosite/norite/troctolite is its *first-listed* Macrostrat lithology, its name, or a named-site tag (3 tiles: Mistastin, Manicouagan, one Quebec sample). Tiles with only minor anorthosite are capped at geology 0.3 (`GEO_MINOR_CAP`) because the geology model was **not retrained** (Colab GPU quota exhausted; local CPU too slow). `ml/geology_ssl.py` gained `--max-len`, `--eval-every`, `--before` for faster CPU runs.
2. **Per-term audit.** 14 scoring terms dropped (zero weight, worse leave-one-out, or weak data such as SoilGrids, aerosol, soil moisture, DINOv2 cosine). `KEEP_TERMS` in `ml/score_all.py` lists the kept terms and, in the comment above it, why each dropped term went.
3. **Dust, biology, caves** were briefly removed, then restored: they compare Earth with known lunar constants (Apollo regolith, sterile Moon) or a known feature (Tranquillitatis pit).
4. **Light-twin month.** Earth sun terms use the month whose noon sun is closest to the Moon target's. Illumination is its own category (noon sun + grazing share), fitted on 10 drafted anchors (`needs_review=1`, geometry-derived). Thermal (temperature) = night minimum + the same sun terms, shown as closest approach only.
5. **Composite** = experiment-weighted mean of terrain, geology, illumination, water_ice (+ caves for the pit), capped by the ground-cover gate (item 8). Operations, thermal, dust and biology are not averaged in (operations is built from terrain + water_ice; the others have no Moon/Earth overlap or are identical for every target). A missing category takes the target's median.
6. **Fixes for flagged anchors:** ice-sheet surface no longer counts as ice-cemented frozen ground; >= 500 mm/yr rain caps water_ice at 0; a dry Moon target (the pit) penalises icy Earth sites; caves "known" needs documented tubes (region match = plausible, 0.3); mostly-water tiles capped at -0.8 for operations; named sites with documented life (Svalbard springs, Rio Tinto, Lake Salda; `KNOWN_LIFE`) capped at -0.6 sterility; named regolith test sites count as >= 60% bare; geology text spells out map jargon and named sites lead with curated rock tags (45 texts re-embedded, no retraining). **Tried and reverted** (both worse): slope tail in terrain; ice-zone-only water_ice in operations.
7. **Website:** thermal closest-approach percentile, "best light in <month>", per-experiment caveats, reliability table.
8. **Sterility and composite review (afternoon).**
   - *Sterility used raw NDVI*, but bare soil reads ~0.1-0.15 (median 0.116 on >= 90% bare tiles), so deserts counted as lightly vegetated and the category topped out at 0.44 (Atacama 0.28 vs label 0.55). Now `plant_cover = clip((NDVI - 0.10) / 0.70, 0, 1)` (`common.NDVI_SOIL/NDVI_VEG`; 0.10 is the baseline already in `config/categories.json`), plant-cover prior 2.0, `KNOWN_LIFE` anchors reported but not fitted (`calibrate(skip_fit=...)`). Atacama 0.40, Beacon Valley 0.45; LOO 0.134 on the 15 fitted anchors (old formula 0.158). Tried and rejected: soil baseline 0.15, frost-day-scaled "liquid" rainfall (both worse on the anchors).
   - *Operations double-counted terrain and ice in the composite* (fitted as 0.58 terrain + 0.30 water_ice + 0.17 slope hazard; terrain drove ~43% of the composite). Removed from `COMPOSITE_CATS`; still a category and experiment requirement.
   - *Nothing in the composite saw ground cover*: moss/lichen tundra (NW Greenland samples) ranked #1 for most polar targets. Ground-cover gate: composite <= 0.8 - 2 x `cover_nonlunar` (plants incl. moss/lichen, open water, built-up; Antarctic sea from the DEM). Moss in the polar top 20: 46% -> 1%; the gate binds on 5% of top-20 positions; top-20 overlap between targets unchanged (0.13 -> 0.12). Best named site per polar target is now Haughton (11), Dry Valleys (11) or Beacon Valley (5). Tried and rejected: gate on biology (NDVI barely sees moss) and on min(dust, biology) (dust needs ~85% bare ground, so it hit half-ice Antarctic tiles and dominated the top lists).
   - Side effects: best composite per polar target ~0.2-0.3 (was 0.3-0.5); planetary-protection "sufficient" pairs 1,350 -> 30,060 (hyper-arid and polar tiles now reach sterility 0.4); new flagged anchor: Haughton biology (label -0.2, model 0.22: tundra patches the tile-level NDVI dilutes).
9. **Optimization pass (evening, 1-hour budget).** Rule: one change at a time, judged on `ml/evaluate.py`, kept only if it beats the scorecard.
   - *Scorecard* (`results/scorecard.json`): site-grouped 5-fold CV on all 161 anchors (folds fixed by a hash of the site id; every anchor predicted by a fit that never saw its Earth site) + the pre-specified polar-analog check on the overall score (AUC of Dry Valleys / Beacon Valley / Haughton / Svalbard vs the other named sites over the 29 polar targets). Baseline: CV error 0.140, 13 anchors off by > 0.3, AUC 0.914.
   - *Kept:* operations + open bare-ground term: operations CV error 0.232 -> 0.204 (all anchors 0.140 -> 0.137, off by > 0.3: 13 -> 12). Map colour switch (relative rank / absolute score).
   - *Rejected:* DINOv2 terrain term (terrain CV 0.104 -> 0.134; second independent confirmation, so the terrain model stays out); operations without the terrain match (error 0.222 but rank 0.847 -> 0.794); removing `GEO_MINOR_CAP` (no anchor sits on a minor-anorthosite tile, so it cannot be judged: add 2-3 such anchors to decide).
   - *Geology retrain on CPU:* 1 epoch (23 min, max-len 128): held-out mAP 0.556, anchor rank 0.592 vs the current model's 0.756 / 0.883. Rejected; the current model stays. A full retrain needs the Colab T4 (open item 2), now run with `--out-tag _v2` and judged with `ml.score_all --emb-suffix _v2 --out-suffix _v2` + `ml.evaluate --compare results/scorecard.json`.
   - *Consistency checks:* ranges, weights, verdict logic and web export all pass. Fixed: tiles without WorldCover outside Antarctica (> 82.6 N, ocean islands) escaped the cover gate; a North Greenland tile that is 50% sea ranked #5 (now 201). Open: 5 craters are listed twice (named site + impact catalog: Haughton, Ries, Lonar, Meteor Crater, Mistastin).
10. **AI council review of 22 anchors** (the 12 off by > 0.3 + the 10 illumination drafts). Three AI reviewers (planetary geologist, lunar polar environment, analog field-ops / astrobiology) labelled each pair **blind**: they saw the Moon target, the Earth site and its data, not the current label or the model score. Rule: new label = median of the 3, adopted only when they agree within 0.4 and the median differs from the old label by > 0.1. 13 of 22 changed (`docs/council_review_2026-11-15.csv`; each changed row's rationale records the 3 labels and reasons); 10 of the 13 moved away from the model's prediction, so the council did not just echo the model. Main shifts: polar illumination 0.55-0.6 -> 0.3 (Earth's best noon sun 34-38 deg vs 2-15 deg on the Moon); Haughton sterility -0.2 -> 0.25; operations at Dry Valleys -0.5 -> 0.3 and Concordia -0.5 -> 0. The reviewers mostly reasoned from the provided data (2-3 tool calls each), so this is a consistency review, **not** a substitute for a domain expert: `needs_review` stays 1.
    - Scorecard on the new labels (not comparable with earlier ones, the answers changed): CV error 0.144, 13 off by > 0.3, polar-analog AUC 0.906. Operations is now the clear weak spot (CV error 0.252): the council rates polar sites more operable than the model does. The bare-ground term still lowers its error (0.278 without) but lowers rank agreement (0.815 without): revisit operations first.
    - *Operations reworked afterwards:* + `clear` (share of ground free of forest, towns, water, wetland, crops). CV error 0.252 -> 0.187, rank 0.774 -> 0.889, off by > 0.3: 6 -> 4; all anchors 0.144 -> 0.137, 13 -> 11. Best of 6 combinations (selection on 18 anchors: treat the size of the gain with care).
    - Top lists leave out catalog tiles within 2 km of a named site (`common.duplicate_of_named`): Haughton, Ries, Lonar, Meteor Crater, Mistastin no longer appear twice.
11. **Final real-world pass.** A real-world check was added to `ml.evaluate`: 13 documented lunar analog tests (`config/real_world_uses.csv`, dev) and 5 held-out ones locked in a commit before any fix (`config/real_world_uses_holdout.csv`: Apollo geology training at Meteor Crater, Pinacate, Askja, Sudbury; CanMoon at Lanzarote). Before: 11 of 13 dev uses and 5 of 5 held-out uses were mostly "avoid".
    - Fixes (`config/experiments.json`): "avoid" only when a required category is below -0.5 (clearly misleading); rover mobility requires operations + dust (terrain supporting); geology/EVA training requires geology (operations and terrain supporting). Rainfall overrides with sources for Mauna Kea saddle (606 mm, NOAA Hale Pohaku normals) and Dallol (81 mm, CHELSA) in `config/site_climate_overrides.csv`.
    - After: not "avoid" for 85% of dev-use targets (all tiles 57%) and 97% of held-out targets (all tiles 65%); polar-analog AUC 0.906 -> 0.926; anchor CV error unchanged (0.137). Honesty note: the geology-only rule for EVA training was adopted after the first post-fix run, in which held-out Sudbury failed for the same reason as dev-set Ries and Clearwater, so the held-out number is not fully blind for that one rule.
    - Remaining real-world misses: Mauna Kea ice prospecting / ISRU (RESOLVE) stays "avoid" because the site has no lunar-like ice; it reproduces the regolith, not the ice.
    - Globe and map colour the overall match by rank among Earth tiles (best overall scores are ~0.2-0.3: no Earth site matches the lunar pole).

### Final calibration (leave-one-out error / LOO rank agreement; full table in `docs/ML_RESULTS.md`)

| Category | Anchors | LOO error | LOO rank agreement |
|---|---|---|---|
| terrain | 21 | 0.106 | 0.937 |
| geology | 22 | 0.179 | 0.885 |
| thermal | 19 | 0.100 | 0.518 |
| illumination | 10 | 0.097 | 0.728 |
| water_ice | 20 | 0.134 | 0.851 |
| dust | 18 | 0.143 | 0.893 |
| biology | 18 (15 fitted) | 0.134 | 0.784 |
| caves (rule levels) | 15 | 0.123 | 0.923 |
| operations | 18 | 0.285 | 0.688 |

**6 of 161 anchors** are off by > 0.3 (`results/anchor_residuals.csv`, column `flag_gt_0.3`):
geology: pit x Lonar, pit x Pinacate; water_ice: de Gerlache ice zone x Beacon Valley; operations: Black Point, Craters of
the Moon; biology: Haughton (added by the sterility fix, item 8). The first five were judged to be model limits, not anchor errors (an anchor review on 2026-11-15 corrected 7 anchors:
two caves labels above the cap, two contradicting pairs at Concordia and the Dry Valleys, Haughton terrain, and two
illumination rationales; changes are noted in each rationale). All anchors still need a site expert (needs_review=1).

## 3. Open items (priority order)

1. **Human review of `config/anchors.csv`** (161 rows, all drafts): flagged rows and the 10 illumination drafts first. This is the biggest quality lever.
2. **Retrain geology on Colab T4** when the quota resets (pseudo-pairs changed with the anorthosite rule). In the notebook cell: `git clone` the repo (see section 4; `tools/reassemble_arrays.py` is not needed for geology), `python -u -W ignore -m ml.geology_ssl`, download `models/geology_bge_ssl.pt` + `embeddings/*geology*` + `results/geology_ssl_metrics.json` into the repo, then locally
   `python -W ignore -m ml.score_all && python -W ignore -m ml.export_web && python tools/make_results_doc.py`. Then consider removing `GEO_MINOR_CAP`.
3. **Antarctic ice-mask hole:** the Dry Valleys tile reports 0% ice within +-10 km (`trav_ice_frac=0`, `trav_dist` 10 km). It does not change scores (frozen ground overrides the distance), but the raw value is wrong.
4. **Experiment-first website view:** pick an experiment, see the best Earth sites.
5. **Operations** CV error 0.187 after the bare-ground and clear-ground terms; geology (0.184) and dust (0.160) are now the next weakest.
6. **Rotate the NASA Earthdata password** (typed into chat in an earlier session; never written to a file; the repo is public).

Other notes: older planning artefacts are superseded (`docs/ML_PLAN.md`, `notebooks/*.ipynb`, `data/target_*.csv`, `data/earth_*.csv`); `config/categories.json` is still used for text flags and tolerances. Locally there is no `pyarrow`, so tables are written as `*.csv.gz` (`ml/common.py: save_table` falls back automatically).

## 4. How to run things

Local machine: Windows 11 ARM64, venv `.venv` (torch CPU, transformers, scikit-learn, rasterio, pandas; no pyarrow). Local runs are fine for scoring and export, **too slow for training**.

```bash
python tools/reassemble_arrays.py                  # joins data/arrays/*.part* -> data/patches/*_128.npz (checksum-verified)
python -W ignore -m ml.step1_validate              # data checks, writes results/step1_validation.json
python -W ignore -m ml.score_all                   # scores + calibration + outputs (about 1 min local; reads embeddings/)
python -W ignore -m ml.evaluate                    # scorecard (results/scorecard.json); --compare an older one
# experiments without code edits: ml.score_all --out-suffix _x --terms '{"operations": [...]}' --geo-minor-cap 0.8
python -W ignore -m ml.export_web                  # writes web/data/
python tools/make_results_doc.py                   # regenerates docs/ML_RESULTS.md from results/ and models/
python tools/reembed_geology.py                    # re-embeds changed geology texts with the existing model (no training)
python -m http.server 8765 --directory web         # local preview at http://localhost:8765
```
Smoke variants: `--smoke` on `ml.terrain_ssl` / `ml.geology_ssl`, and `ml.score_all --emb-suffix _smoke --out-suffix _smoke` (smoke outputs are git-ignored). CPU geology options: `--max-len`, `--eval-every`, `--before`.
Pushing to `main` redeploys the website automatically.

**Colab (T4) training:** open the notebook (Drive id `1nFV3F5Ams0EULo8nsPS8KiTcqyqDC-V2`, "earth_analog_ssl_colab.ipynb"), one code cell:
```
!rm -rf /content/analog_project; git clone -q https://github.com/Talmzr/Earth-analog-finder /content/analog_project
%cd /content/analog_project
!python tools/reassemble_arrays.py                                                          # terrain only
!python -u -W ignore -m ml.terrain_ssl --coral 25 --epochs 12 2>&1 | grep -v -i warn     # ~12 min (not needed: terrain embedding unused)
!python -u -W ignore -m ml.geology_ssl 2>&1 | grep -v -i warn                              # ~5 min
!zip -q -r /content/earth_analog_ml_outputs.zip models embeddings results
from google.colab import files; files.download('/content/earth_analog_ml_outputs.zip')
```
Unzip into the repo root (`unzip -o ~/Downloads/earth_analog_ml_outputs.zip`). Via the Chrome extension: typing long code freezes the page, so set the cell text with JS
(`colab.global.notebook.cells[2].setText(...)` then `.manualExecute()`); Drive permission popups must be approved by the user.

## 5. Score definition (so you can reason about changes)

- Sub-score of one physical quantity: `1 - |d|/tol` inside one tolerance, then a **soft** floor toward -1 (`-(1 - exp(-(|d|/tol - 1)))`), so far-off Earth sites keep their closest-approach order. One-sided versions only penalise Earth values below the Moon's (ice).
- Per category: `calibrated = clip(intercept + sum_k w_k * term_k, -1, 1)`, **w_k >= 0** fitted on that category's anchors by ridge toward prior weights, using only the terms in `KEEP_TERMS`, then caps (`CAPS`): terrain 0.9, geology 0.8 (0.3 for minor anorthosite), thermal 0.4, illumination 0.7, water_ice 0.7, dust 0.7, biology 0.6, caves 0.5, operations 0.8, plus the rule caps listed in section 2 item 6.
- Kept terms: terrain = relief/slope/roughness kernel; geology = lithology match + BGE text cosine; thermal = night minimum, grazing sun, noon sun; illumination = noon sun, grazing sun (light-twin month); water_ice = frozen ground, traverse to ice, no liquid water; dust = bare ground, tephra; biology = NDVI, precipitation; caves = rule levels (known documented tubes 0.5, region match / basaltic volcano 0.3, 0 else, negative for sedimentary/ice); operations = calibrated terrain + water_ice + slope hazard.
- Moon targets: zones carry their own pixel-level thermal/ice/sun features; terrain features come from the 10 km window at the zone centroid.
- Composite per (Moon target, Earth tile): weighted mean over `COMPOSITE_CATS` (terrain, geology, illumination, water_ice, caves), weights = number of relevant experiments that require (1.0) or support (0.5) the category; missing category = target median; then `min(mean, 0.8 - 2 x cover_nonlunar)` (`COVER_GATE_BASE`, `COVER_GATE_SLOPE`). `results/score_matrices.npz` keeps the ungated mean as `composite_ungated`.

## 6. Known data limits (also in `docs/DATA_REPORT.md`)

Temperature and sun angle do not overlap between Moon and Earth (Moon night 24-80 K vs Earth >= 201 K; Moon noon sun < 15 deg vs Earth >= 24 deg), so thermal means "closest Earth gets" (shown as a percentile). Earth climate layers are coarse (POWER ~50 km, LST 5 km, SoilGrids 5 km, now unused). Antarctica has no WorldCover, SoilGrids or GSMaP (601 tiles; fallbacks in `ml/common.py: prepare_earth`), and the ice mask has a hole at the Dry Valleys (open item 3). Ice zones are tiny (0.5-9 km2) and blurred by Diviner's 240-360 m cells. Neighbouring tiles are correlated: always evaluate on held-out spatial blocks. Earth cannot reproduce 1/6 g, vacuum, radiation, or 25-110 K shadowed temperatures (always caveats, never scored). Artemis III region outlines: official 2022 NASA Moon Trek outlines for 5 regions; others placed by a criteria search or the anchor (`placement_method` in `data/ml/moon_zones.csv`).

## 7. Prompt for a new session (updated)

> Read `docs/HANDOFF.md` in https://github.com/Talmzr/Earth-analog-finder (clone it), then continue from the open items above (section 3, in order). Constraints: MVP first, finetune don't train from scratch, heavy training on Colab T4, small steps locally, commit and push as you go.
