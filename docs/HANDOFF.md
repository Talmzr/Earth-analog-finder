# Handoff: Earth Analog Finder (NASA Space Apps 2026)

Written 2026-11-15 for a new session. Read this first, then `docs/ML_RESULTS.md` and `docs/DATA_REPORT.md`.
Repo: https://github.com/Talmzr/Earth-analog-finder (public, branch `main`). Project root = repo root (the folder was `analog_project/` locally).

## 1. What the project is

Challenge: "Identify Earth locations that analog the permanent Moon base locations". **Earth + Moon only (Mars was dropped).**
We score every (Moon target, Earth tile) pair per category on a -1..+1 scale, then turn scores into experiment advice
(>= 0.5 sufficient, 0..0.5 partial, < 0 avoid, plus "also account for" caveats).

- **Moon side:** 30 scoring targets = 20 Artemis zones (10 regions x landing zone / ice zone) + 10 target tiles (9 south-pole anchors + Tranquillitatis pit).
- **Earth side:** 4,310 tiles of 10 x 10 km (30 named analog sites, 190 impact structures, 1,091 volcanoes, ~3,000 random barren / polar / ice-edge / lichen samples).
- **Categories (8):** terrain, geology, thermal, water_ice, dust, biology, caves, operations.
- **User priorities:** MVP first, finetune pretrained models (never train from scratch), do not polish things that are not the MVP,
  heavy compute on Colab T4, light work local. The deadline was **09:00 on 2026-11-15** (hackathon). Final goal: the calibrated data usable in the team's website.

## 2. State at handoff (what is DONE)

| Area | State | Where |
|---|---|---|
| Data tables | Done and validated | `data/ml/moon_tiles.csv` (5,610 x 95: 3,609 polar + 2,001 mid-latitude), `earth_tiles.csv` (4,310 x 143), `moon_zones.csv` (20 x 56), `moon_zone_patch_terrain.csv`, `manifest.json`. Dictionary: `docs/DATA_REPORT.md` |
| Terrain image patches | Done, 128 x 128 (78 m/px), stored as <= 94 MB parts | `data/arrays/` + `python tools/reassemble_arrays.py`; zone patches in `data/patches/moon_zone_arrays_128.npz` |
| Anchors (labels) | 151 drafts, **all `needs_review=1`, no human review yet** | `config/anchors.csv` (guide: `docs/ANCHORS.md`) |
| Terrain model | Fine-tuned (Colab T4) | `models/terrain_dinov2_ssl.pt`, `embeddings/*_terrain_384.npy`, metrics `results/terrain_ssl_metrics.json` |
| Geology model | Fine-tuned (Colab T4) | `models/geology_bge_ssl.pt`, `embeddings/*_geology_384.npy`, metrics `results/geology_ssl_metrics.json` |
| Parametric scorers + calibration | Done and run on real embeddings | `ml/score_all.py`, `models/calibration_params.json`, `results/` |
| Website bundle | Exported (from the pre-fix scores) | `web/data/` (34 JSON files, 8.9 MB), demo page `web/index.html` |
| Results write-up | Done (pre-fix numbers) | `docs/ML_RESULTS.md` |

Last pushed commit: `9ce83bf`. Commits are in `git log`; the pipeline was built in this order: data -> `ml/step1_validate.py` -> `ml/terrain_ssl.py` -> `ml/geology_ssl.py` -> `ml/score_all.py` -> `ml/export_web.py`.

### Model results (held-out spatial blocks, no anchors used in training)

- **Terrain (DINOv2-small, VICReg + CORAL, 12 epochs, best epoch 4 kept by early stopping):** two-view retrieval top-1 0.486 -> 0.836;
  cross-body terrain agreement (1.0 = random, lower better; relief / slope / roughness) 1.034 / 0.801 / 1.002 -> 0.569 / 0.514 / 0.476.
  Moon-vs-Earth AUC is still ~0.9999 (embeddings are fully separable by body), so `score_all.py` compares post-hoc alignments and picked
  `inlp8` (iterative null-space projection of 8 body directions), agreement 0.462 (pooled 0.515, per-body 0.604).
  A longer 40-epoch run was *worse* on cross-body agreement than epoch 4, which is why early stopping on that metric exists.
- **Geology (BGE-small, InfoNCE on lithology pseudo-pairs, 20 epochs):** retrieval mAP 0.171 -> 0.756; Spearman with the 22 geology anchors 0.588 -> 0.883 (anchors were not used in training; anchor sites are excluded from the training pairs).

### Calibration against anchors (`results/calibration_summary.csv`, run before the anorthosite fix below)

| Category | Anchors | MAE | LOO MAE | Rank agreement | Off by > 0.3 |
|---|---|---|---|---|---|
| terrain | 21 | 0.148 | 0.162 | 0.939 | 2 |
| geology | 22 | 0.182 | 0.197 | 0.902 | 2-3 |
| thermal | 19 | 0.120 | 0.139 | 0.640 (raw prior -0.373) | 0 |
| water_ice | 20 | 0.136 | 0.180 | 0.954 | 2 |
| dust | 18 | 0.147 | 0.200 | 0.910 | 0 |
| biology | 18 | 0.176 | 0.196 | 0.875 | 1 |
| caves (rule levels, not rescaled) | 15 | 0.140 | 0.140 | 0.911 | 1 |
| operations | 18 | 0.233 | 0.281 | 0.743 | 4 |

12 of 151 anchors disagree with the model by more than 0.3: `results/anchor_residuals.csv` (flag column `flag_gt_0.3`).

## 3. UNFINISHED / next steps (in priority order)

### 3.1 Uncommitted geology fix needs re-scoring and a geology retrain (do first, ~15 min)
Found while reviewing: tiles whose bedrock is **gneiss/granite with anorthosite only as a minor component** of a long unit description
(e.g. `sparse_cold sample 49`, Rae Craton) were classed as anorthositic and scored 0.72 geology against Haworth. 39 tiles scored >= 0.5, only a handful are real.
The fix is in the working tree of this commit (`ml/geo_text.py` adds `earth_primary_text()` and an `anorth_minor` class; call sites updated in `ml/geology_ssl.py` and `ml/score_all.py`). It reduced "anorthosite" tiles from 42 to 25 strong + 17 minor, **but it has not been re-scored or re-validated.**
Still to check: many remaining "anorth" tiles are granite/gneiss units that list anorthosite *early* in the text (the primary-text check uses name + `mstrat_lith` + context only). Inspect the 25 listed tiles; consider requiring anorthosite/norite/troctolite in `mstrat_lith` itself, and keep `mafic_plut` as a partial match.
Then:
1. `python -m ml.geology_ssl` on Colab (the pseudo-pairs changed, so retrain; ~5 min on T4).
2. Unzip outputs into the repo, `python -W ignore -m ml.score_all`, `python -W ignore -m ml.export_web`.
3. Regenerate `docs/ML_RESULTS.md` (see 3.4), commit, push.

### 3.2 Review and optimise results (the user asked for this if budget remained; only started)
- **Polar landing-zone analogs look weak and scattered:** top Earth tiles for the south-pole zones are Antarctic ice-free samples and hot-desert/Andean volcanic fields with composite only 0.05-0.2. Check whether the composite weighting (`score_all.readiness` / `Wm`, weights by how many experiments need each category) is sensible, and whether the terrain kernel/SSL blend (weights kernel 0.73, ssl_cos 0.46) is dominated by one term.
- **Thermal:** every polar zone scores <= -0.6 against every Earth tile (best Concordia/plateau ~ -0.4 by the anchors' own convention). This is physically honest (Earth is never lunar-cold), but the website needs a "closest approach" view next to the absolute score (idea: percentile rank among Earth tiles).
- **Nothing is "sufficient" for ISRU ice extraction or illumination power planning** (0 pairs), by design of the caps (water_ice <= 0.3 vs cold traps, thermal capped at 0 for permanently shadowed zones). Decide whether that is the story to present or whether requirements in `config/experiments.json` need relaxing.
- **Operations anchors** have the worst calibration (rank agreement 0.74): 4 flagged. Operations is derived from calibrated terrain + thermal + water_ice + slope-hazard + traverse; consider fitting fewer terms or reviewing the anchors.
- **Tranquillitatis pit:** best composite is 0.44 (Harrat volcanic fields in Arabia), but thermal terms for it are extrapolated (all thermal anchors are polar).
- Open anchor issues listed in `docs/ANCHORS.md` (e.g. Malapert ice zone x Concordia, Haworth x Dallol rainfall value looks wrong, some Macrostrat lithologies wrong: Craters of the Moon reads "gneiss, granite", Sudbury tile is basin fill).

### 3.3 Website deployment (not done)
- Only a **minimal demo** exists: `web/index.html` (Leaflet map + category bars + experiment tags) reading `web/data/`. It works over HTTP (`python -m http.server 8765 --directory web`); it was checked in the browser with no console errors.
- The data contract is documented at the top of `ml/export_web.py`: `meta.json`, `moon_targets.json`, `earth_tiles.json` (parallel arrays), `scores/<moon_id>.json` (all category scores x 100 as ints, `null` = n/a, composite, readiness codes 3/2/1/0/-1, top 25), `top_analogs.json`.
- **Not deployed anywhere** (no GitHub Pages yet). The team's real site was never inspected. A new session should ask where the site lives and adapt the export (or enable GitHub Pages on `web/`).
- The page does not yet show: caveats per experiment (data is in `meta.json` -> `experiments[].also_account_for` and `earth_cannot_reproduce`), the model/calibration quality (`meta.json` -> `categories[]`), or a "closest approach" thermal view.

### 3.4 Smaller to-dos
- `docs/ML_RESULTS.md` was generated by an **inline Python snippet that was not saved to a file**. Recreate it as `tools/make_results_doc.py` (inputs: `results/*.json|csv`, `models/calibration_params.json`, `results/score_matrices.npz`); the doc currently reflects the pre-fix run.
- Parquet outputs: the local ARM64 Python has no `pyarrow` wheel, so `category_scores`, `experiment_readiness` and the per-category score files are `*.csv.gz` locally (`ml/common.py: save_table` falls back automatically). Running `python -m ml.score_all` on Colab writes real `.parquet`.
- Human review of `config/anchors.csv` (151 rows, all unreviewed) is the single biggest quality lever; flagged rows first.
- Older planning artefacts are superseded and can confuse: `docs/ML_PLAN.md`, `config/categories.json` (still used for text flags and tolerances), `notebooks/*.ipynb` (older notebook, **not used** by the `ml/` pipeline), `data/target_*.csv` / `data/earth_*.csv` (first-pass site-level tables).
- Tell the user to **rotate their NASA Earthdata password** (it was typed into chat earlier; it was never written to any file). The repo is public and commits carry the user's git author email.

## 4. How to run things

Local machine: Windows 11 ARM64, venv `.venv` (torch CPU, transformers 5.23, scikit-learn, rasterio, pandas; no pyarrow). Local runs are fine for scoring and export, **too slow for training** (1-epoch terrain smoke test ~5 min, full embedding pass ~20 min on CPU).

```bash
python tools/reassemble_arrays.py                  # joins data/arrays/*.part* -> data/patches/*_128.npz (checksum-verified)
python -W ignore -m ml.step1_validate              # data checks, writes results/step1_validation.json
python -W ignore -m ml.score_all                   # scores + calibration + outputs (about 2 min local; reads embeddings/)
python -W ignore -m ml.export_web                  # writes web/data/
```
Smoke variants exist: `--smoke` on `ml.terrain_ssl` / `ml.geology_ssl`, and `ml.score_all --emb-suffix _smoke --out-suffix _smoke` (smoke outputs are git-ignored; delete them before real runs).

**Colab (T4) training, the way it was done:** open the notebook (Drive id `1nFV3F5Ams0EULo8nsPS8KiTcqyqDC-V2`, "earth_analog_ssl_colab.ipynb"), put this in one code cell and run it:
```
!rm -rf /content/analog_project; git clone -q https://github.com/Talmzr/Earth-analog-finder /content/analog_project
%cd /content/analog_project
!python tools/reassemble_arrays.py
!python -u -W ignore -m ml.terrain_ssl --coral 25 --epochs 12 2>&1 | grep -v -i warn     # ~12 min (best-epoch checkpoint kept)
!python -u -W ignore -m ml.geology_ssl 2>&1 | grep -v -i warn                              # ~5 min
!zip -q -r /content/earth_analog_ml_outputs.zip models embeddings results
from google.colab import files; files.download('/content/earth_analog_ml_outputs.zip')
```
Then unzip into the repo root (`unzip -o ~/Downloads/earth_analog_ml_outputs.zip`). Notes from doing this through the Chrome extension: typing long code freezes the page, so paste via clipboard or set the cell text with JS
(`colab.global.notebook.cells[2].setText(...)` then `.manualExecute()`); Drive permission popups must be approved by the user; read progress from `[...document.querySelectorAll('.cell.code')][1].querySelector('.output').innerText`.
The terrain run in the zip that is committed used `--coral 25 --epochs 12`; `terrain_dinov2_ssl.pt` is that run (the earlier 40-epoch non-CORAL and CORAL checkpoints were overwritten/ignored).

## 5. Score definition (so you can reason about changes)

- Sub-score of one physical quantity: `1 - |d|/tol` inside one tolerance, then a **soft** floor toward -1 (`-(1 - exp(-(|d|/tol - 1)))`), so Earth sites that are many tolerances away still keep their closest-approach order. (The original brief's hard floor made all Earth sites tie at -1 for temperature.) One-sided versions only penalise Earth values *below* the Moon's (ice).
- Per category: `calibrated = clip(intercept + sum_k w_k * term_k, -1, 1)`, **w_k >= 0** fitted on that category's anchors by ridge toward prior weights (`C.fit_terms`), then caps: terrain 0.9 (and <= 0 when tree + built + water >= 0.5), geology 0.8, thermal 0.4 (and <= 0 for permanently shadowed Moon targets), water_ice 0.7 (and <= 0.3 against cold traps), dust 0.7, biology 0.6, caves 0.5, operations 0.8 (-0.5 for forest/towns, <= -0.3 if obstacles >= 0.5).
- Terms: terrain = SSL cosine (best of pooled / per-body / INLP alignment) + measured relief/slope/roughness kernel; geology = BGE text cosine + discrete lithology match; thermal = swing, min, max temperature, grazing-sun share, noon sun, polar night; water_ice = frozen ground, surface ice, flat-bare-ground-to-ice distance vs the Moon's landing-to-ice distance, no liquid water; dust = Earth soil vs a fixed lunar-regolith reference; biology = 1 - (NDVI, precipitation, soil moisture, tree cover); caves = rule levels (+0.5 known lava-tube region/site, +0.3 basaltic volcano, 0 else, -0.5 sedimentary/ice) used as the score itself; operations = calibrated terrain + thermal + water_ice, slope hazard, traverse.
- Moon targets: zones carry their own pixel-level thermal/ice/sun features; their terrain features come from the 10 km window at the zone centroid (zones have no tile image).
- Composite per (Moon target, Earth tile): weighted mean of category scores, weights = number of experiments (relevant to that Moon target) that require (1.0) or support (0.5) the category.

## 6. Known data limits (also in `docs/DATA_REPORT.md`)

Temperature and sun angle do not overlap between Moon and Earth (Moon night 24-80 K vs Earth >= 201 K; Moon noon sun < 15 deg vs Earth >= 24 deg), so thermal means "closest Earth gets". Earth climate layers are coarse (POWER ~50 km, LST 5 km, SoilGrids 5 km). Antarctica has no WorldCover, SoilGrids or GSMaP (601 Antarctic tiles; fallbacks in `ml/common.py: prepare_earth`). Ice zones are tiny (0.5-9 km2) and blurred by Diviner's 240-360 m cells. Neighbouring tiles are correlated: always evaluate on held-out spatial blocks (`C.spatial_blocks`, `C.holdout_mask`). Earth cannot reproduce 1/6 g, vacuum, radiation, or 25-110 K shadowed temperatures (always caveats, never scored). Artemis III region outlines: official 2022 NASA Moon Trek outlines for 5 regions; others placed by a criteria search or the anchor (`placement_method` in `data/ml/moon_zones.csv`).

## 7. Prompt to give the new session

> Read `docs/HANDOFF.md` in https://github.com/Talmzr/Earth-analog-finder (clone it), then continue: (1) finish and verify the uncommitted anorthosite fix, retrain geology on Colab, re-score, re-export `web/data`, regenerate `docs/ML_RESULTS.md` as a script; (2) review and optimise the results (section 3.2); (3) wire `web/data` into our website (ask me where it lives). Constraints: MVP first, finetune don't train from scratch, heavy training on Colab T4, small steps locally, commit and push as you go.
