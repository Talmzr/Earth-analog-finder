# Earth Analog Finder: Earth sites for permanent Moon base locations

NASA Space Apps 2026 challenge: *Identify Earth locations that analog the permanent Moon base locations*.

**Live site:** https://talmzr.github.io/Earth-analog-finder/

Pick a Moon target, colour Earth sites by category, and read the verdict of each field-test experiment with its caveats. The site is built from `web/` and deployed by `.github/workflows/pages.yml` on every push to `main`.

## What it does

For each of 30 Moon targets, the tool scores 4,310 Earth tiles in 9 categories. It then says which Earth sites are good enough for each of 14 experiments (landing hazard detection, rover mobility, ice prospecting, ISRU, dust mitigation, cave robotics, EVA geology training and others).

| | Count | Detail |
|---|---|---|
| Moon targets | 30 | 20 Artemis III landing and ice zones (10 regions x 2) + 10 target tiles, including the Tranquillitatis pit |
| Earth tiles (10 x 10 km) | 4,310 | 30 named analog sites, 190 impact structures, 1,091 volcanoes, about 3,000 random barren and polar samples |
| Categories | 9 | see table below |
| Experiments | 14 | `config/experiments.json` |

### Scores and verdicts

Every score runs from -1 to +1:

| Score | Meaning |
|---|---|
| >= 0.5 | **sufficient**: the site reproduces this aspect |
| 0 to 0.5 | **partial** |
| < 0 | **avoid** |

Each experiment gets a verdict (sufficient / partial / avoid) plus an "also account for" list. Gravity, vacuum, radiation and the 25-110 K temperatures of permanent shadow can never be reproduced on Earth, so they always appear as caveats.

### Categories

LOO = leave-one-out check against the expert anchors (mean error / rank agreement).

| Category | What is compared | LOO error / rank |
|---|---|---|
| Terrain | Measured relief, slope, roughness, share of steep ground (LOLA 80 m vs Copernicus DEM) | .141 / .943 |
| Geology | Lunar map-unit text vs Earth bedrock text (fine-tuned BGE-small) + rock-type match | .179 / .885 |
| Temperature (thermal) | Night minimum + light-twin sun terms; closest approach only, Earth never gets lunar-cold | .100 / .518 |
| Illumination | Noon sun height + share of grazing sun in the Earth "light-twin month" (the month whose sun is closest to the Moon target's); real overlap for about 35-40% of Earth sites | .097 / .728 * |
| Water / ice | Frozen ground (ice sheets do not count as ice-cemented soil), landing-to-ice distance, dryness | .150 / .846 |
| Dust / regolith | Bare ground + volcanic tephra vs known Apollo regolith (same for every Moon target) | .143 / .893 |
| Biological sterility | Vegetation and rainfall vs the sterile Moon (same for every Moon target) | .185 / .866 |
| Caves | Lava-tube rule; Tranquillitatis pit only | .133 / .919 |
| Operations | Terrain + ice scores + slope hazard, with penalties for forest, towns and open water | .283 / .686 |

\* The 10 illumination anchors are drafts derived from the same sun geometry, so this number is optimistic.

### Overall (composite) score

The overall score for a Moon target is the experiment-weighted mean of the categories that actually compare with that specific Moon site: terrain, geology, illumination, water/ice and operations (plus caves for the pit). Temperature, dust and sterility are scored and drive experiment verdicts, but stay out of the composite because they are identical across targets or have no real overlap. Missing category data counts as the target's median.

### Models

| Model | Base | Fine-tuning | Used in score? |
|---|---|---|---|
| Geology text | `BAAI/bge-small-en-v1.5` | Contrastive (InfoNCE) on Moon/Earth lithology pseudo-pairs | Yes |
| Terrain image | `facebook/dinov2-small` | Self-supervised (VICReg + CORAL) on 128 x 128 elevation patches | No: trained and evaluated, but added nothing over measured relief and slope |

Fine-tuning ran on a Colab T4 GPU. Scoring runs on a laptop CPU in about 2 minutes.

## Status: MVP with draft expert calibration

- All 161 expert anchors (`config/anchors.csv`) are drafts (`needs_review=1`). They set the scale of every category, so scores will move when experts review them.
- 8 anchors disagree with the model by more than 0.3 (`results/anchor_residuals.csv`, column `flag_gt_0.3`).
- The geology model was trained before a stricter anorthosite rule. Tiles with only minor anorthosite are capped at 0.3 until the model is retrained.
- Earth climate layers are 5-50 km, coarser than the tiles.
- 10 km tiles dilute small test sites (a quarry or a single lava tube is a small part of its tile).
- Antarctica lacks land-cover, soil, rain and bedrock layers; fallbacks are used there.
- Thermal power survival (extreme cold, long night) has no qualifying Earth site. This is physically correct, not a bug.

## How to run

From the repo root, in a Python virtual environment with `pip install -r requirements.txt`:

```bash
python tools/reassemble_arrays.py          # rebuild image arrays from their parts
python -W ignore -m ml.step1_validate      # data checks
python -W ignore -m ml.score_all           # scores + calibration (~2 min, CPU)
python -W ignore -m ml.export_web          # web/data bundle for the site
python tools/make_results_doc.py           # regenerate docs/ML_RESULTS.md
python tools/reembed_geology.py            # re-embed changed geology texts (no training)
python -m http.server 8765 --directory web # local preview at http://localhost:8765
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
| `web/` | Static map site (GitHub Pages) |
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
