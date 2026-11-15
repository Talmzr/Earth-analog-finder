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
5. **Composite** = experiment-weighted mean of terrain, geology, illumination, water_ice, operations (+ caves for the pit). Thermal, dust and biology stay out (no Moon/Earth overlap, or identical for every target). A missing category takes the target's median.
6. **Fixes for flagged anchors:** ice-sheet surface no longer counts as ice-cemented frozen ground; >= 500 mm/yr rain caps water_ice at 0; a dry Moon target (the pit) penalises icy Earth sites; caves "known" needs documented tubes (region match = plausible, 0.3); mostly-water tiles capped at -0.8 for operations; named sites with documented life (Svalbard springs, Rio Tinto, Lake Salda; `KNOWN_LIFE`) capped at -0.6 sterility; named regolith test sites count as >= 60% bare; geology text spells out map jargon and named sites lead with curated rock tags (45 texts re-embedded, no retraining). **Tried and reverted** (both worse): slope tail in terrain; ice-zone-only water_ice in operations.
7. **Website:** thermal closest-approach percentile, "best light in <month>", per-experiment caveats, reliability table.

### Final calibration (leave-one-out error / LOO rank agreement; full table in `docs/ML_RESULTS.md`)

| Category | Anchors | LOO error | LOO rank agreement |
|---|---|---|---|
| terrain | 21 | 0.106 | 0.937 |
| geology | 22 | 0.179 | 0.885 |
| thermal | 19 | 0.100 | 0.518 |
| illumination | 10 | 0.097 | 0.728 |
| water_ice | 20 | 0.134 | 0.851 |
| dust | 18 | 0.143 | 0.893 |
| biology | 18 | 0.185 | 0.866 |
| caves (rule levels) | 15 | 0.123 | 0.923 |
| operations | 18 | 0.285 | 0.688 |

**5 of 161 anchors** are off by > 0.3 (`results/anchor_residuals.csv`, column `flag_gt_0.3`):
geology: pit x Lonar, pit x Pinacate; water_ice: de Gerlache ice zone x Beacon Valley; operations: Black Point, Craters of
the Moon. All five were judged to be model limits, not anchor errors (an anchor review on 2026-11-15 corrected 7 anchors:
two caves labels above the cap, two contradicting pairs at Concordia and the Dry Valleys, Haughton terrain, and two
illumination rationales; changes are noted in each rationale). All anchors still need a site expert (needs_review=1).

## 3. Open items (priority order)

1. **Human review of `config/anchors.csv`** (161 rows, all drafts): flagged rows and the 10 illumination drafts first. This is the biggest quality lever.
2. **Retrain geology on Colab T4** when the quota resets (pseudo-pairs changed with the anorthosite rule). In the notebook cell: `git clone` the repo (see section 4; `tools/reassemble_arrays.py` is not needed for geology), `python -u -W ignore -m ml.geology_ssl`, download `models/geology_bge_ssl.pt` + `embeddings/*geology*` + `results/geology_ssl_metrics.json` into the repo, then locally
   `python -W ignore -m ml.score_all && python -W ignore -m ml.export_web && python tools/make_results_doc.py`. Then consider removing `GEO_MINOR_CAP`.
3. **Antarctic ice-mask hole:** the Dry Valleys tile reports 0% ice within +-10 km (`trav_ice_frac=0`, `trav_dist` 10 km). It does not change scores (frozen ground overrides the distance), but the raw value is wrong.
4. **Experiment-first website view:** pick an experiment, see the best Earth sites.
5. **Operations** is the weakest category (LOO error 0.285, rank 0.688).
6. **Rotate the NASA Earthdata password** (typed into chat in an earlier session; never written to a file; the repo is public).

Other notes: older planning artefacts are superseded (`docs/ML_PLAN.md`, `notebooks/*.ipynb`, `data/target_*.csv`, `data/earth_*.csv`); `config/categories.json` is still used for text flags and tolerances. Locally there is no `pyarrow`, so tables are written as `*.csv.gz` (`ml/common.py: save_table` falls back automatically).

## 4. How to run things

Local machine: Windows 11 ARM64, venv `.venv` (torch CPU, transformers, scikit-learn, rasterio, pandas; no pyarrow). Local runs are fine for scoring and export, **too slow for training**.

```bash
python tools/reassemble_arrays.py                  # joins data/arrays/*.part* -> data/patches/*_128.npz (checksum-verified)
python -W ignore -m ml.step1_validate              # data checks, writes results/step1_validation.json
python -W ignore -m ml.score_all                   # scores + calibration + outputs (about 2 min local; reads embeddings/)
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
- Composite per (Moon target, Earth tile): weighted mean over `COMPOSITE_CATS` (terrain, geology, illumination, water_ice, operations, caves), weights = number of relevant experiments that require (1.0) or support (0.5) the category; missing category = target median.

## 6. Known data limits (also in `docs/DATA_REPORT.md`)

Temperature and sun angle do not overlap between Moon and Earth (Moon night 24-80 K vs Earth >= 201 K; Moon noon sun < 15 deg vs Earth >= 24 deg), so thermal means "closest Earth gets" (shown as a percentile). Earth climate layers are coarse (POWER ~50 km, LST 5 km, SoilGrids 5 km, now unused). Antarctica has no WorldCover, SoilGrids or GSMaP (601 tiles; fallbacks in `ml/common.py: prepare_earth`), and the ice mask has a hole at the Dry Valleys (open item 3). Ice zones are tiny (0.5-9 km2) and blurred by Diviner's 240-360 m cells. Neighbouring tiles are correlated: always evaluate on held-out spatial blocks. Earth cannot reproduce 1/6 g, vacuum, radiation, or 25-110 K shadowed temperatures (always caveats, never scored). Artemis III region outlines: official 2022 NASA Moon Trek outlines for 5 regions; others placed by a criteria search or the anchor (`placement_method` in `data/ml/moon_zones.csv`).

## 7. Prompt for a new session (updated)

> Read `docs/HANDOFF.md` in https://github.com/Talmzr/Earth-analog-finder (clone it), then continue from the open items above (section 3, in order). Constraints: MVP first, finetune don't train from scratch, heavy training on Colab T4, small steps locally, commit and push as you go.
