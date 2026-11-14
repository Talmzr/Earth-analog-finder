# ML plan: Moon <-> Earth analog scores (Space Apps 2026)

**Goal.** Give a score in [-1, 1] for every (Moon patch, Earth patch) pair and every analog category. Turn the scores into "sufficient / partial / avoid for experiment X, and also account for A, B, C".
**Files.** `config/categories.json` (inputs, tolerances, labeling functions, caps), `config/experiments.json` (rules), `config/anchors.csv` (draft expert labels), `notebooks/ssl_colab.ipynb` (Colab T4). Training runs only on Colab; nothing has been trained yet.
**Data (as of writing).** Moon: 3,609 patches of 10 km (3,600 grid + 9 Artemis targets, -76 to -90 deg). Earth: 4,321 candidates (30 named sites, 192 impact structures, ~1,100 volcanoes, ~2,500 land-cover samples). Environment and terrain builds are still filling in.

## 1. Why patches fix the input side but not the label side

* **Inputs.** Patches turn 40 site rows into ~8k samples. That is enough to learn feature distributions, train encoders and check them on held-out blocks.
* **Labels.** The target is "how good is Earth patch e as an analog for Moon patch m, in category c". No dataset measures it. 3,609 x 4,321 is ~15.6 M pairs with **zero** labels. Adding patches adds pairs, not labels.
* **The target depends on purpose.** The same pair can be fine for rover traction and useless for cryogenic drilling, and a tolerance has to say how close is close enough. No sensor records that.
* **For many quantities the two bodies do not overlap.** South-pole surfaces average 26-210 K; no Earth surface is below ~180 K. Ice sits at 25-110 K in vacuum, and the Moon has no life at all. No Moon-Earth pair is "truly similar" here, so even a perfect similarity learner has nothing to copy.
* **Fewer independent samples than it looks.** Neighbouring patches (10 km stride) are strongly correlated, so validation must use spatial blocks.

## 2. Fixing the label side (four parts)

| Part | What it gives | What it cannot give |
|---|---|---|
| **SSL representation** (DINOv2 on DEMs, bge on geology text) | which patches *look alike* across and within bodies; crater bowls vs ridges vs plains, lithology semantics | how many kelvin, degrees or % ice count as "close enough"; any absolute meaning of a score |
| **Weak supervision**: physics labeling functions (LFs) with units (`categories.json`) | noisy +1/-1 votes on ~30k sampled pairs; coverage 37-79%, conflict < 3% in the dry run | truth. LFs encode our assumptions, so `eval_only` LFs are held out to check agreement |
| **Expert anchors** (`anchors.csv`, 47 drafts, `needs_review=1`) | the meaning of the scale: they fit a 2-parameter monotone calibration per category and act as the final test | statistical power. With ~5 per category they are a sanity check; target is >= 15 reviewed per category |
| **Label-free evaluation** | augmentation invariance, domain-probe AUC, retrieval of known analogs, physically impossible pairs, held-out LF agreement, spatial-block CV, seed stability | proof of usefulness. It only rules out broken models |

**Stated tolerances.** The physics kernel uses tolerances `tau` with units: 60 K, 10 deg of noon sun, 0.25 ice fraction, 3 deg of slope. They are the honest answer to "is 230 K close to 100 K?". With no anchors, `score = tanh(kernel)`, and a pair is *sufficient* when its features are on average within ~0.75 tau. Anchors can move that scale, but a ridge pulls it back towards the tolerances, so a few unreviewed anchors cannot overrule the physics.

## 3. Cross-domain (Moon vs Earth) shift

1. **Inputs that mean the same on both bodies.** The DEM channels are computed identically on both bodies: elevation anomaly/1000 m, slope/30 deg, curvature, all at ~156 m pixels with slopes on a common 500 m baseline. There is **no `sunvis` channel**, because a zero channel for Earth would reveal the body. Body words ("lunar", "Earth") and country fields are stripped from text.
2. **Normalisation.** Features stay in pooled physical units. **No per-body quantile normalisation for features whose ranges do not overlap.** It would pair a "warm" 180 K Moon patch with a median Earth site, when physically the coldest Earth site is always the closest. Per-patch normalisation (elevation anomaly) and pooled-mean centring of embeddings are fine.
3. **Alignment.** CORAL (lambda = 1) between the Moon and Earth halves of each balanced batch matches second-order statistics. Domain-adversarial training (DANN) is not used: it is unstable in a 1-day build and can force matches where Earth simply has no such landform.
4. **Guardrails.** A linear Moon-vs-Earth probe AUC is measured on held-out blocks for every embedding. If AUC > 0.9, set `alpha_ssl` to 0.3 or ship the kernel. Caps cover unreachable conditions (a PSR-dominated patch caps the thermal score at 0). A one-sided gap is used where "more on Earth" is harmless (ground ice).

## 4. Pair score

`raw = alpha * z(cos(e_M, e_E)) + (1 - alpha) * z(kernel)`, where `kernel = 2 exp(-0.5 mean_i((x_M - x_E)/tau_i)^2) - 1` and `z()` standardises over 20k random pairs.
`score = min(capMoon, capEarth, score_cap, tanh(a (raw - b)))`. Here `a > 0` and `b` are fitted per category on anchors (confidence weights 1 / 0.6 / 0.3) plus weak labels (total weight 0.25), with ridge to the kernel prior.
Bands: **>= 0.5 sufficient, 0 to 0.5 partial, < 0 avoid**, NaN = not applicable. A NaN kernel (no data) is never filled in by text or images alone.

## 5. Per-category models

| # | Category | Moon inputs | Earth inputs | Model (HF checkpoint) | alpha | Cap | Status |
|---|---|---|---|---|---|---|---|
| 1 | geology | `moon_geo_unit_name` + catalog description, `geo_frac_*` | `mstrat_lith`, `mstrat_descrip`, `context` (GVP rock type, impact row, tags) | `BAAI/bge-small-en-v1.5` (fallback `all-MiniLM-L6-v2`), MNRL fine-tune on ontology-mined triplets + lithology-class kernel | 0.5 | 0.8 | learned |
| 2 | terrain | `relief_m, slope500_mean/p90_deg, frac_slope500_gt10/gt20, tri500_mean_m, elev_std_m`; arrays `elev, slope` | same + `wc_frac_tree/built/water` | `facebook/dinov2-small`, last 4 blocks, VICReg + CORAL; scalar VICReg MLP (0.7 image / 0.3 scalar) | 0.6 | 0.9 | learned |
| 3 | thermal_illumination | `dvn_tavg_mean_k, dvn_tbol_min_k, dvn_tmax_max_k, noon_elev_max_deg, mean_day_elev_deg, frac_polar_day/night`, PSR/coldtrap for caps | `lst_mean/night_min/day_max_k` (POWER fallback), same geometry | physics kernel (tau 60/60/40 K, 10/6 deg, 0.15); tabular SSL only as an ablation | 0 | 0.6, PSR 0 | kernel |
| 4 | water_ice | `dvn_frac_ice_within_1m, dvn_frac_ice_stable, dvn_ice_depth_median_m, dvn_frac_coldtrap_110k` | `pf_frac_pct` -> `permafrost_flag` -> MAAT proxy, `wc_frac_snow_ice`, precipitation, `soil_moist_mean` | kernel, one-sided ice gap, wetness penalty | 0 | 0.7, coldtrap 0.3 | kernel |
| 5 | dust_regolith | constant lunar-regolith prototype (fines 45%, bulk density 1.5 g/cm3, dry, glassy; *verify*) | `sg_silt+clay, sg_cfvo, sg_bdod, soil_moist, ndvi, wc_bare, tephra flag` | prototype kernel (no Moon-side variation) | 0 | 0.7 | kernel |
| 6 | biology_sterility | sterile constant | NDVI above bare-soil baseline, vegetated WorldCover share, precipitation, soil moisture | prototype kernel | 0 | 0.6 | kernel |
| 7 | caves_subsurface | the Tranquillitatis pit only (off the polar grid) | lava-tube/basalt flags from tags and GVP text, tree cover | rule table | - | 0.7 | **rule only** |
| 8 | ops_landing_mobility | hazard fractions | hazard fractions, obstacle land cover | derived: 0.45 terrain + 0.2 dust + 0.15 thermal + 0.2 hazard kernel, then calibrated | - | 0.8 | derived |

TabPFN (`Prior-Labs/TabPFN-v2-reg`) is an optional calibration head once a category has >= 30 reviewed anchors (check the licence). Until then the 2-parameter tanh fit is used.

**Keep / drop.** *Caves:* drop the learned model. There is one Moon target, it is outside the patch grid, and there are ~5 Earth examples. Keep a transparent rule for the pit. *Dust:* keep, but as a prototype kernel with low weight. At the pole the Moon side has no per-patch dust data (Diviner rock abundance stops at +-60 deg), so the score ranks Earth sites against fixed lunar-regolith values. *Wind:* dropped (no air). *Biology:* keep (Earth-only sterility ranking, capped at 0.6 because no Earth surface is sterile).

## 6. Weak-supervision labeling functions (examples; full list with units in `categories.json`)

| Category | +1 when | -1 when |
|---|---|---|
| terrain | relief within x1.4, mean slope within 3 deg, steep share within 15 pp | p90 slope off by > 12 deg; relief off by > x4; tree+built+water > 30% (DSM artefacts) |
| thermal | Earth night LST <= 213 K and Moon mean <= 150 K; Earth noon sun <= 35 deg | Earth day LST >= 320 K vs a cold Moon patch; Earth noon sun >= 60 deg |
| water_ice | Moon ice within 1 m >= 0.3 and Earth permafrost >= 0.9 | the same Moon patch with Earth permafrost <= 0.05; wetness index >= 0.6 |
| dust | silt+clay >= 30%, coarse <= 15%, precipitation <= 250 mm/yr, NDVI <= 0.1; bare tephra | NDVI >= 0.3 or precipitation >= 800 mm/yr; snow/ice >= 50% |
| biology | excess NDVI <= 0.05, precipitation <= 50 mm/yr, vegetated <= 2% | excess NDVI >= 0.2 or vegetated >= 20%; moss/lichen/wetland present |
| geology | highland unit x anorthosite; crater unit x impact structure | carbonate/evaporite/clay (non-impact); ice-sheet surface |
| ops | both landable (steep10 <= 0.1); both hazard-rich | steep-share gap >= 0.4; forest/fields/towns/water >= 30% |

## 7. Training and evaluation protocol

* **Spatial-block CV.** Moon: 50 km blocks in polar-stereographic x/y (144 blocks). Earth: ~5 deg cells widened with latitude (588 blocks). 5 folds, fold 0 held out. SSL trains on folds != 0. All diagnostics run on fold 0. Weak labels never train the encoders, so LF agreement stays a fair check.
* **Pass criteria** (in `results/ml_eval.json`):

| Check | Pass |
|---|---|
| terrain augmentation invariance, cos(x, aug x), held-out | >= 0.9 and well above random-pair cos |
| domain-probe AUC per embedding | <= 0.8 good; > 0.9 = lower alpha or use the kernel |
| positive anchors (>= 0.5): percentile among all Earth patches for their Moon target | median >= 90th |
| physically impossible pairs (e.g. forest vs biology, wet tropics vs icy patch) | < 5% scoring >= 0, < 1% scoring >= 0.5 |
| held-out (`eval_only`) LF sign agreement | >= 0.7 |
| anchor leave-one-out MAE / Spearman | report; ship SSL only where it beats the kernel's anchor Spearman |
| seed stability: top-20 Jaccard over 3 seeds | >= 0.6 |

* **Dry run, kernel only, real data, no training** (run locally; the POWER pull was ~62% complete). Geology: Mistastin ranks at the 100th percentile for Haworth and Malapert, and Spearman vs anchors is 0.90 (n = 9). Biology: Dry Valleys, Beacon Valley and Concordia rank top, Undara and Ries bottom (Spearman 0.73). The impossible-pair checks pass for every category. Two findings matter for training. (a) The crater-unit x impact-structure LF agrees only 7%: the scalar terrain kernel cannot see crater morphology, which is exactly what the DINOv2 encoder is for. (b) Thermal stays at or below -0.4 for every Earth site (best: Concordia). With 60 K / 10 deg tolerances no Earth site comes close, and 6 unreviewed anchors are not allowed to overrule that. If the team decides the coldest Earth class should count as "partial", it must raise `tau` or add reviewed anchors. **That is the label-side decision SSL cannot make.**

## 8. Compute budget (Colab T4, 16 GB)

| Step | Full run | SMALL_RUN |
|---|---|---|
| DINOv2-small, 112 px, batch 128 x 2 views, fp16, 30 epochs (~1.9k steps) | 10-20 min | 20 steps, ~1 min |
| tabular VICReg (3k steps) | ~1 min | seconds |
| bge-small zero-shot + fine-tune (<= 3k triplets, 3 epochs) | 2-4 min | ~1 min |
| kernels, LFs, calibration, evaluation, export (~1.6 M rows), inference | ~1-2 min (40 s on a laptop CPU) | seconds |
| **Total** (excluding Drive upload of ~270 MB of arrays) | **~20-30 min** | **~5 min** |

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Biggest: the meaning of the scale rests on 47 unreviewed draft anchors (~5 per category; none >= 0.5 yet for terrain, thermal or water) and on hand-set tolerances.** "Sufficient" is a stated convention, not a measurement. | Review anchors first (section 10). Grow to >= 15 per category. Show LOO error. Present scores as "agreement with stated tolerances". |
| The SSL embedding encodes the body (Earth erosion networks and DSM canopy vs lunar craters) | Body-invariant channels, CORAL, domain-AUC gate, kernel fallback |
| Alignment invents matches | Small lambda; impossible-pair rate; one-sided gaps and caps instead of aggressive alignment |
| Anchors are point sites, features are 10 km means (Mauna Kea tephra pits vs a wet 10 km patch; the 1.2 km Meteor Crater inside a flat window) | Say so in rationales; add finer patches for named sites later |
| Data gaps: Antarctica has no WorldCover, SoilGrids or AMSR2; POWER still filling; permafrost proxy | NaN -> `not_applicable`, never a guess; coverage table printed in notebook section 2 |
| Weak labels and calibration can be circular | `eval_only` LFs held out; encoders never see weak labels |

## 10. One-day schedule

| Time | Who | Task |
|---|---|---|
| 0:00-0:30 | ML | Copy the project to Drive, run the notebook with `SMALL_RUN=True` (kernel path already verified) |
| 0:00-1:30 | domain experts | Review `anchors.csv` (fix labels, set `needs_review=0`); add positives for terrain, thermal and water |
| 0:30-1:30 | ML | Full run; read invariance and domain AUC; choose `alpha` per category |
| 1:30-2:30 | ML | Re-run with reviewed anchors; seeds 0/1/2; freeze `categories.json` |
| 2:30-4:00 | all | Merge `experiment_recommendations.csv` into `results/report.md`, heatmaps, per-target cards |
| 4:00-5:00 | all | Limitations slide (gravity, vacuum, radiation, PSR 25-110 K), demo, pitch |
