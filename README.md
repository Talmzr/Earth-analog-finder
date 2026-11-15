# TerraMatch: Earth analogs for permanent Moon base locations

NASA Space Apps 2026 challenge: *Identify Earth locations that analog the permanent Moon base locations*.

**Live site:** https://talmzr.github.io/Earth-analog-finder/ (3-D globe; 2-D map at `/map.html`)

Pick a Moon target, see which Earth places come closest to it, and read the verdict of each field-test experiment with its caveats. The site is built from `web/` and deployed by `.github/workflows/pages.yml` on every push to `main`.

## Top picks

No place on Earth reproduces the lunar south pole: vacuum, 1/6 g, 25-110 K permanent shadow and a sun that stays within a few degrees of the horizon are always missing. TerraMatch therefore finds the **best partial analogs**, aspect by aspect:

- **Artemis III south-pole regions:** ice-free ground in Antarctica (around the McMurdo Dry Valleys, the Royal Society Range and Dronning Maud Land nunataks) and High Arctic polar deserts (Canadian Arctic Archipelago, northern Greenland, Svalbard) lead almost every list. They match low sun in their best month, frozen ground and dry, bare surfaces. Their rocks do **not** match lunar highland anorthosite: geology is negative for nearly all of them, and anorthosite sites such as Mistastin are the geology analogs instead.
- **Tranquillitatis pit (mare lava-tube skylight):** young basaltic volcanic fields with documented lava tubes in hot deserts (Arabian harrats such as Harrat Kishb, and Afar). They match lava-tube basalt, not the pit's vacuum or depth.
- **Best overall scores are about 0.2-0.3 ("partial")** on a -1..+1 scale. The globe and map colour every score on the same absolute scale: any positive score is green (light green just above 0, full green from 0.5) and negative scores are pink-red, so the best analogs show green while places with no positive match stay red. The 2-D map can switch to a relative view (rank among Earth tiles).

## What it does

For each of 30 Moon targets, the tool scores 4,310 Earth tiles in 9 categories. It then says which Earth sites suit each of 14 experiments (landing hazard detection, rover mobility, ice prospecting, ISRU, dust mitigation, cave robotics, EVA geology training and others).

| | Count | Detail |
|---|---|---|
| Moon targets | 30 | 20 Artemis III landing and ice zones (10 regions x 2) + 10 target tiles, including the Tranquillitatis pit |
| Earth tiles (10 x 10 km) | 4,310 | 30 named analog sites, 190 impact structures, 1,091 volcanoes, about 3,000 random barren and polar samples (shown by land cover and region) |
| Categories | 9 | see table below |
| Experiments | 14 | `config/experiments.json` |

### Scores and verdicts

Every category score runs from -1 to +1:

| Score | Meaning |
|---|---|
| >= 0.5 | **sufficient**: the site reproduces this aspect |
| 0 to 0.5 | **partial** |
| < 0 | **poor match** for this aspect |

Each experiment gets a verdict plus an "also account for" list: **sufficient** when every required category meets its minimum, **avoid** only when a required category is clearly misleading (below -0.5), **partial** otherwise. Real analog campaigns (Desert RATS, PANGAEA, Apollo geology training) use sites that match the tested aspect and only partly match the rest, so "partial" is the normal verdict for a good analog. Gravity, vacuum, radiation and the 25-110 K temperatures of permanent shadow can never be reproduced on Earth, so they always appear as caveats.

**Place verdict (the badge on the globe):** a place (Moon target x Earth tile) is **sufficient** if at least one experiment that depends on that Moon target is sufficient there (dust mitigation and planetary protection, which are the same for every Moon target, can only make it partial), **partial** if none is sufficient but at least one is partial, and **avoid** only if no experiment is even partly possible. The overall score (colour) shows how close the place is overall; the badge shows whether it is usable for at least one test.

### Categories

CV = site-grouped 5-fold cross-validation against the 161 draft anchors (mean absolute error / rank agreement); naive = predicting the average label.

| Category | What is compared | CV error / rank | naive error |
|---|---|---|---|
| Terrain | Measured relief, slope, roughness, share of steep ground (LOLA 80 m vs Copernicus DEM) | .104 / .944 | .551 |
| Geology | Lunar map-unit text vs Earth bedrock text (fine-tuned BGE-small) + rock-type match | .184 / .926 | .572 |
| Temperature (thermal) | Night minimum + light-twin sun terms; closest approach only, Earth never gets lunar-cold | .094 / .700 | .163 |
| Illumination | Noon sun height + share of grazing sun in the Earth "light-twin month" (the month whose sun is closest to the Moon target's) | .089 / .728 \* | .426 |
| Water / ice | Frozen ground (ice sheets do not count as ice-cemented soil), landing-to-ice distance, dryness | .146 / .873 | .513 |
| Dust / regolith | Bare ground + volcanic tephra vs known Apollo regolith (same for every Moon target) | .160 / .911 | .464 |
| Biological sterility | Plant cover (vegetation index above the bare-soil level) and rainfall vs the sterile Moon (same for every Moon target) | .117 / .802 | .560 |
| Caves | Lava-tube rule; Tranquillitatis pit only | .123 / .923 | .499 |
| Operations | Terrain match + slope hazard + open bare ground + ground free of forest, towns, water and crops (+ ice access) | .187 / .877 | .481 |

\* The 10 illumination anchors are drafts derived from the same sun geometry, so this number is optimistic.

### Overall (composite) score

The overall score for a Moon target is the experiment-weighted mean of the categories that compare with that specific Moon site: terrain, geology, illumination and water/ice (plus caves for the pit). Operations stays out because it is built from the terrain and ice scores (it would count them twice). Temperature, dust and sterility are scored and drive experiment verdicts, but are not averaged in because they are identical across targets or have no real overlap. Missing category data counts as the target's median.

The mean is then capped by a **ground-cover gate**: overall <= 0.6 - 2 x the share of the tile under plants (including moss and lichen), open water or towns, so a tile with 30% or more such cover is at most 0.

### Models

| Model | Base | Fine-tuning | Used in score? |
|---|---|---|---|
| Geology text | `BAAI/bge-small-en-v1.5` | Contrastive (InfoNCE) on Moon/Earth lithology pseudo-pairs | Yes |
| Terrain image | `facebook/dinov2-small` | Self-supervised (VICReg + CORAL) on 128 x 128 elevation patches | No: trained and evaluated, but added nothing over measured relief and slope |

Fine-tuning ran on a Colab T4 GPU. Scoring runs on a laptop CPU in about 1 minute.

## How accurate is it?

All numbers come from `python -W ignore -m ml.evaluate` (`results/scorecard.json`).

| Check | What it compares with | Result |
|---|---|---|
| Category accuracy | 161 **draft** anchors (team-written labels; 22 revised by a blind, model-assisted review), site-grouped 5-fold CV | mean error **0.137** on the -1..+1 scale vs **0.474** for predicting the average label; 11 of 161 off by more than 0.3 |
| Overall ranking (sanity check) | The 4 pre-named polar analogs (Dry Valleys, Beacon Valley, Haughton, Svalbard) vs the other 26 named sites, 29 south-pole targets | AUC **0.944** (0.5 = chance) |
| Real-world site choices (consistency check) | 13 documented lunar analog tests (Desert RATS, RESOLVE, PANGAEA, BRAILLE, FINESSE, ARCHES, K10, Concordia) and 5 held-out ones (Apollo geology training at Meteor Crater, Pinacate, Askja, Sudbury; CanMoon at Lanzarote); sources in `config/real_world_uses*.csv` | not rated "avoid" for **85%** (dev) and **97%** (held-out) of their Moon targets, vs 74% / 89% for the other named sites and 57% / 65% for an average tile |

How to read these numbers honestly:

- The anchors are unreviewed drafts written from the same data layers, and the model structure (terms, gate, operations formula) was chosen on them, so 0.137 is an optimistic estimate.
- The AUC uses sites that also carry category anchors: it checks consistency, it does not prove accuracy.
- "Not avoid" is a low bar. The model rates these real test sites "sufficient" for only 16% (dev) / 5% (held-out) of their targets, the avoid threshold was relaxed to -0.5 after the dev failures, and 4 of the 5 held-out uses test one rule (geology/EVA training). That rule (geology only) was adopted after held-out Sudbury failed for the same reason as dev-set Ries and Clearwater, so the held-out number is not fully blind.
- Known miss: Mauna Kea's ice-prospecting and ISRU tests (RESOLVE) stay "avoid" because the site reproduces lunar regolith but has no lunar-like ice.

Data checks: all 30 named Earth sites and 10 Moon targets sit at their published coordinates. Rainfall comes from 25-50 km grids and is replaced by documented local values where it was clearly wrong for a small or steep site (Mauna Kea saddle, Dallol: `config/site_climate_overrides.csv`). Random-sample tiles are named by their land cover and an OpenStreetMap region (`config/tile_place_names.csv`) or the nearest catalogued site.

## Limits

- All 161 anchors (`config/anchors.csv`) are drafts (`needs_review=1`); 11 disagree with the cross-validated model by more than 0.3 (`results/anchor_residuals.csv`, column `cv_pred`). Scores will move when domain experts review them.
- The geology model was trained before a stricter anorthosite rule; tiles with only minor anorthosite are capped at 0.3 until it is retrained.
- Earth climate layers are 5-50 km, coarser than the tiles, and 10 km tiles dilute small test sites (a quarry or a single lava tube is a small part of its tile).
- Antarctica lacks land-cover, soil, rain and bedrock layers; fallbacks are used there, and that is where many top analogs are.
- Sun-height overlap with the lunar pole exists only in an Earth site's best month and only for high-latitude sites.
- Thermal power survival (extreme cold, long night) has no qualifying Earth site. This is physically correct, not a bug.

## How to run

From the repo root, in a Python virtual environment with `pip install -r requirements.txt`:

```bash
python tools/reassemble_arrays.py          # rebuild image arrays from their parts
python -W ignore -m ml.step1_validate      # data checks
python -W ignore -m ml.score_all           # scores + calibration (~1 min, CPU)
python -W ignore -m ml.evaluate            # scorecard: site-grouped CV per category + polar-analog check
python -W ignore -m ml.export_web          # web/data bundle for the site
python tools/build_globe.py                # web/globe.html: 3-D globe with all final scores embedded (site root redirects to it)
python tools/make_results_doc.py           # regenerate docs/ML_RESULTS.md
python tools/reembed_geology.py            # re-embed changed geology texts (no training)
python -m http.server 8765 --directory web # local preview at http://localhost:8765 (2-D map: /map.html)
```

Training (Colab T4, optional; the trained models are in `models/`):

```bash
python -m ml.terrain_ssl --coral 25 --epochs 12
python -m ml.geology_ssl
```

## Repo map

| Path | Contents |
|---|---|
| `data/ml/` | Main Moon / Earth / zone tables |
| `data/arrays/` | Image arrays, split into parts (rebuild with `tools/reassemble_arrays.py`) |
| `config/` | `anchors.csv` (expert anchors), `experiments.json`, `categories.json` |
| `ml/` | Pipeline: validation, scoring, model training, web export |
| `tools/` | Data building, array reassembly, results doc, geology re-embedding |
| `models/`, `embeddings/` | Fine-tuned models and their embeddings |
| `results/` | Scores, calibration, anchor residuals |
| `web/` | Static map site (GitHub Pages); `web/globe.html` is the 3-D globe (built by `tools/build_globe.py`; the site root opens it), `web/map.html` the 2-D map |
| `frontend/` | Globe source: `globe.template.html` (page code), `assess.js` (Similarity Assessment page), `moon_textures.json` |
| `docs/` | `ML_RESULTS.md` (generated results), `DATA_REPORT.md` (data dictionary), `ANCHORS.md`, `HANDOFF.md` |

**Superseded (kept for history, not used by the pipeline):** `docs/ML_PLAN.md`, `notebooks/*.ipynb`, `data/target_*.csv`, `data/earth_*.csv`. The root scripts `extract.py`, `extract_extra.py` and `compare.py` come from the first data-collection phase, when Mars was still in scope; Mars has since been dropped.

## Documentation

- `docs/ML_RESULTS.md`: generated results (model evaluation, anchor calibration, experiment readiness, best analog per Moon target, known limits)
- `docs/DATA_REPORT.md`: data dictionary (every table, column and coverage)
- `DATA_SOURCES.md`: where each data layer comes from and what was skipped
- `docs/ANCHORS.md`: anchor columns and how the draft labels were set

## Data credits

All sources are open data.

- **Moon:** LOLA DEM and average solar visibility (Barker et al.; Mazarico et al.; NASA GSFC PGDA), LOLA global DEM (USGS), Diviner polar and global products (NASA PDS), USGS Unified Geologic Map of the Moon, NASA Moon Trek Artemis III region outlines.
- **Earth:** Copernicus DEM GLO-30 (ESA/Airbus, AWS Open Data), ESA WorldCover, ESA CCI Permafrost, NASA MODIS (via JAXA), NASA POWER, JAXA GSMaP, ISRIC SoilGrids, Macrostrat, Smithsonian Global Volcanism Program, Earth impact structure catalog.
- **Models:** BAAI bge-small-en-v1.5 and Meta DINOv2-small (Hugging Face).
