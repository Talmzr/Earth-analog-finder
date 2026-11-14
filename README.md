# Earth Analog Finder: Moon base sites & Mars

NASA Space Apps 2026 challenge: *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*.

The tool scores 30 Earth sites against 14 Moon and Mars targets. Each pair gets four separate category scores and five mission-task scores. It also writes an inference report: which **new candidate** sites beat the established analogs, and where **no Earth site is a good match**.

## Run it (about 10–20 minutes, mostly API calls)

```bash
conda create -n analog -c conda-forge python=3.11 rasterio pandas requests matplotlib -y
conda activate analog
python extract.py --check        # ~10 s: is every data source reachable?
python extract.py                # fills data/*.csv (cached per site, safe to re-run)
python compare.py                # writes results/
```

**If `rasterio` won't install** (common on Windows ARM), use Google Colab:
upload this folder, then run `!pip install rasterio` followed by the three commands above, each with a leading `!`.

Useful flags: `--only haughton,moon_haworth` (just those sites), `--skip-env` (terrain + light only), `--refresh` (ignore cache), `-v` (full error traces).

## What is compared

| Category | Moon / Mars side | Earth side | Score |
|---|---|---|---|
| Terrain | LOLA south-pole DEM 80 m (NASA PGDA COG); MOLA 463 m (USGS COG) | Copernicus GLO-30 (AWS COGs, global incl. polar) | Distance in standardized space over relief, slope **at a common 500 m baseline**, % steep ground, roughness (TRI) |
| Illumination | Solar geometry from latitude + obliquity (Moon 1.54°, Mars 25.19°) | Same, Earth 23.44° | Distance over noon sun max/min, polar day/night fractions, mean daytime sun height. Also a **"light-twin month"**: the Earth month whose noon sun best matches the target |
| Environment | — | MODIS NDVI & day/night LST (ORNL API), ERA5 temperature & precipitation (Open-Meteo) | "Closest approach" percentile: drier, colder, barer, bigger day/night swing = higher |
| Geology | Curated tags (crater, anorthosite, ground ice, lava tube…) | Curated tags | 80% share of the target's traits present + 20% Jaccard |

Mission-task weights live in `config/scoring.json` (overall, landing_rover, ice_isru, cave_shelter, habitat_ops). Change them to fit your pitch.

## Outputs (`results/`)
- `report.md`: best analog per target, new candidates that beat every established analog, category gaps, per-body ranking
- `top5_<task>.csv`, `best_by_category.csv`, `scores_long.csv` (all pairs, all scores)
- `heatmap_<task>.png`: targets × Earth sites, with the best match outlined

## Honest limitations (say these to the judges; it builds trust)
- **Artemis region coordinates are anchors, not official centres.** NASA publishes the region names, not their centres. Each region is placed on its named crater or peak (see `coord_note` in `config/targets.csv`); swap in real outlines if you find them.
- **Illumination ignores topography.** At the lunar poles, crater rims and peaks catch far more sun than flat-ground geometry predicts. Next step: NASA PGDA illumination maps.
- **Slope depends on resolution.** Everything is compared at a 500 m baseline. Mars data (463 m) can't show smaller features, so Mars terrain is coarser than Moon/Earth.
- **Environment can only rank Earth sites against each other.** No Earth site gets close to vacuum or −200 °C, so this category measures "most extreme on Earth", not similarity.
- Tags and some Earth coordinates were curated by hand and are approximate (window ±4–12 km). The Tranquillitatis pit and Deuteronilus entries were added from memory; verify them.
- Scores are relative to this set of sites. Add sites to `config/earth_sites.csv` and re-run.

## Data credits
LOLA: Barker et al., NASA GSFC PGDA. MOLA: NASA/USGS Astrogeology. Copernicus DEM GLO-30: ESA/Airbus (AWS Open Data). MODIS: NASA LP DAAC via ORNL DAAC TESViS web service. ERA5: ECMWF via Open-Meteo.
