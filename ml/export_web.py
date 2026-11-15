"""Website bundle: compact JSON the front end can load directly.   python -m ml.export_web

web/data/
  meta.json               categories, score bands, experiments (requirements + caveats), calibration quality,
                          conditions Earth cannot reproduce, known limitations
  moon_targets.json       the 30 Moon targets (20 Artemis landing / ice zones + 10 target tiles) with key numbers
  earth_tiles.json        the Earth tiles as parallel arrays (id, name, type, lat, lon + a few measured values)
  scores/<moon_id>.json   for one Moon target: every category score for every Earth tile (integers = score x 100,
                          null = not available), composite, experiment readiness codes, top 25 tiles,
                          percentile.thermal = closest-approach rank among Earth tiles (0-100, 100 = closest),
                          twin_month = Earth month (1-12, 0 = none) whose sun is closest to this Moon target's
  top_analogs.json        top 20 Earth tiles per Moon target with names, scores and the experiments they suit
Scores are in [-1, 1]: >= 0.5 sufficient, 0..0.5 partial, < 0 avoid. Readiness codes: 3 sufficient, 2 partial,
1 avoid, 0 not available, -1 experiment not relevant for that Moon target.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import common as C

LABELS = {"terrain": "Terrain & topography", "geology": "Geology & surface materials",
          "thermal": "Temperature", "illumination": "Sunlight & illumination", "water_ice": "Water, ice & permafrost",
          "dust": "Dust & regolith", "biology": "Biological sterility", "caves": "Caves & lava tubes",
          "operations": "Landing & operations"}
HOW = {"terrain": "Measured relief, slope, roughness and steep-ground share, same 80 m elevation processing on both bodies (the fine-tuned DINOv2 model is evaluated but does not improve the score, so it is not used)",
       "geology": "BGE-small text encoder fine-tuned contrastively on lunar map units vs Earth lithology, plus a lithology-match term",
       "thermal": "Night minimum temperature plus the light-twin sun terms. Closest approach only: no Earth site is lunar-cold, so this ranks how cold and polar the site is. Not part of the overall score",
       "illumination": "Noon sun height and grazing-sun share in the Earth light-twin month (the month whose sun is closest to the Moon target's). A real overlap for ~35-40% of Earth sites; Earth still has a 24 h day and a scattering sky. Calibrated on draft anchors",
       "water_ice": "Frozen ground within ~1 m, distance from flat bare ground to ice, absence of liquid water",
       "dust": "Bare ground share and volcanic tephra, compared with the known lunar regolith (Apollo samples: bare, dry, fine). Same for every Moon target, so not part of the overall score",
       "biology": "How lifeless the Earth site is (vegetation greenness, rainfall), compared with the sterile Moon. Same for every Moon target, capped at 0.6, not part of the overall score",
       "caves": "Rule: known lava-tube region / basaltic volcano / no tubes, against the Tranquillitatis pit (an LRO-imaged skylight). Only for that target",
       "operations": "Calibrated terrain and ice scores plus slope hazard, penalty for forest / towns / water"}


def ints(a):
    """float array -> list of ints (x100) with None for NaN (compact JSON)."""
    a = np.asarray(a, float)
    out = np.round(a * 100).astype(object)
    out[~np.isfinite(a)] = None
    return [None if v is None else int(v) for v in out]


def percentile(a):
    """Rank of each Earth tile among all Earth tiles for one Moon target, 0..100 (100 = best); None for NaN."""
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    out = np.full(a.shape, np.nan)
    if ok.sum() > 1:
        out[ok] = pd.Series(a[ok]).rank(pct=True, method="average").to_numpy()
    return out


def r(v, n=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, n) if np.isfinite(v) else None


def main(sfx=""):
    out = C.ROOT / "web" / "data"
    (out / "scores").mkdir(parents=True, exist_ok=True)
    Z = np.load(C.RES / f"score_matrices{sfx}.npz", allow_pickle=False)
    cats, ent_ids, tile_ids = Z["categories"].tolist(), Z["entity_id"].tolist(), Z["earth_tile_id"].tolist()
    cal, comp, ready, exps_ids, Wm = Z["calibrated"], Z["composite"], Z["readiness"], Z["experiments"].tolist(), Z["category_weights"]
    twin = Z["light_twin_month"] if "light_twin_month" in Z.files else None
    moon, earth, zones = C.load_tables()
    ent = C.build_entities(moon, zones).set_index("entity_id").loc[ent_ids].reset_index()
    earth = earth.set_index("patch_id").loc[tile_ids].reset_index()
    exps_cfg = json.loads((C.CFG / "experiments.json").read_text(encoding="utf-8"))
    params = json.loads((C.MODELS / f"calibration_params{sfx}.json").read_text())
    summ = pd.read_csv(C.RES / f"calibration_summary{sfx}.csv").set_index("category")

    # ---- earth tiles (parallel arrays)
    E = dict(id=tile_ids, name=earth["name"].fillna("").tolist(), source=earth["source"].tolist(), stratum=earth["stratum"].tolist(),
             lat=[r(v, 4) for v in earth["center_lat"]], lon=[r(v, 4) for v in earth["center_lon"]],
             relief_m=[r(v, 0) for v in earth["relief_m"]], slope_deg=[r(v, 1) for v in earth["slope500_mean_deg"]],
             temp_min_k=[r(v, 0) for v in earth["x_temp_min_k"]], temp_max_k=[r(v, 0) for v in earth["x_temp_max_k"]],
             frozen_ground=[r(v, 2) for v in earth["x_frozen_ground_frac"]],
             ice_dist_km=[r(v, 1) for v in earth["trav_dist_flat_to_ice_km"]],
             rock=[("" if pd.isna(v) else str(v)[:60]) for v in earth["mstrat_lith"]])
    (out / "earth_tiles.json").write_text(json.dumps(E, separators=(",", ":")), encoding="utf-8")

    # ---- per Moon target
    exp_name = {e["id"]: e["name"] for e in exps_cfg["experiments"]}
    targets, tops_all = [], {}
    for i, m in enumerate(ent_ids):
        e = ent.iloc[i]
        order = np.argsort(-np.nan_to_num(comp[i], nan=-9))
        th_pct = percentile(cal[cats.index("thermal"), i]) if "thermal" in cats else None
        top = []
        for rank, j in enumerate(order[:25], 1):
            suff = [exps_ids[x] for x in range(len(exps_ids)) if ready[x, i, j] == 3]
            part = [exps_ids[x] for x in range(len(exps_ids)) if ready[x, i, j] == 2]
            top.append(dict(rank=rank, tile=int(j), id=tile_ids[j], name=E["name"][j], stratum=E["stratum"][j],
                            lat=E["lat"][j], lon=E["lon"][j], composite=r(comp[i, j]),
                            scores={c: r(cal[k, i, j]) for k, c in enumerate(cats)},
                            thermal_percentile=None if th_pct is None else r(th_pct[j] * 100, 0),
                            twin_month=None if twin is None or not twin[i, j] else int(twin[i, j]),
                            sufficient_for=suff, partial_for=part))
        tops_all[m] = top[:20]
        relevant = [exps_ids[x] for x in range(len(exps_ids)) if (ready[x, i] >= 0).any()]
        (out / "scores" / f"{m}.json").write_text(json.dumps(dict(
            moon_id=m, categories=cats, scores={c: ints(cal[k, i]) for k, c in enumerate(cats)}, composite=ints(comp[i]),
            percentile={} if th_pct is None else {"thermal": ints(th_pct)},
            twin_month=None if twin is None else twin[i].astype(int).tolist(),
            readiness={exps_ids[x]: ready[x, i].astype(int).tolist() for x in range(len(exps_ids)) if exps_ids[x] in relevant},
            top=top), separators=(",", ":")), encoding="utf-8")
        best = {c: (None if not np.isfinite(cal[k, i]).any() else r(float(np.nanmax(cal[k, i])))) for k, c in enumerate(cats)}
        n_suff = {exps_ids[x]: int((ready[x, i] == 3).sum()) for x in range(len(exps_ids)) if exps_ids[x] in relevant}
        targets.append(dict(
            id=m, name=e["name"], kind=e["kind"], zone=e["zone"], region=e["region_id"], target=e["target_id"],
            lat=r(e["lat"], 4), lon=r(e["lon"], 4), placement=e["placement_method"], geology_unit=e["moon_geo_unit"],
            slope_deg=r(e["slope500_mean_deg"], 1), relief_m=r(e["relief_m"], 0), temp_max_k=r(e["x_temp_max_k"], 0),
            temp_mean_k=r(e["x_temp_mean_k"], 0), temp_min_k=r(e["x_temp_min_k"], 0), shadowed_frac=r(e["lola_frac_psr"], 2),
            ice_within_1m_frac=r(e["x_frozen_ground_frac"], 2), cold_trap_frac=r(e["x_surface_ice_frac"], 2),
            landing_to_ice_km=r(e["landing_to_ice_median_km"], 2), best_score=best, relevant_experiments=relevant,
            earth_tiles_sufficient=n_suff, composite_weights={c: r(Wm[k, i] / max(Wm[:, i].sum(), 1e-9), 3) for k, c in enumerate(cats)},
            top3=[dict(name=t["name"], id=t["id"], composite=t["composite"]) for t in top[:3]]))
    (out / "moon_targets.json").write_text(json.dumps(targets, indent=1), encoding="utf-8")
    (out / "top_analogs.json").write_text(json.dumps(tops_all, separators=(",", ":")), encoding="utf-8")

    # ---- meta
    cat_meta = []
    for c in cats:
        st = params["categories"].get(c, {})
        row = summ.loc[c] if c in summ.index else {}
        cat_meta.append(dict(id=c, config_id=C.CATS[c], label=LABELS[c], how=HOW[c], cap=params["caps"].get(c),
                             n_anchors=int(st.get("n_anchors", 0)), terms=st.get("weights"), intercept=st.get("intercept"),
                             anchor_mae=r(row.get("mae")), anchor_mae_leave_one_out=r(row.get("mae_loo")),
                             anchor_rank_agreement=r(row.get("spearman")), anchors_off_by_0_3=int(row.get("n_flagged", 0))))
    meta = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_moon_targets=len(ent_ids), n_earth_tiles=len(tile_ids),
        score=dict(range=[-1, 1], bands=dict(sufficient=">= 0.5", partial="0 to 0.5", avoid="< 0"), stored_as="integer = score x 100"),
        readiness_codes={"3": "sufficient", "2": "partial", "1": "avoid", "0": "not available", "-1": "experiment not relevant for this Moon target"},
        categories=cat_meta,
        experiments=[dict(id=e["id"], name=e["name"], required={C.CFG2OUT[k]: v for k, v in e["required"].items()},
                          supporting={C.CFG2OUT[k]: v for k, v in (e.get("supporting") or {}).items()},
                          also_account_for=e.get("also_account_for", [])) for e in exps_cfg["experiments"]],
        earth_cannot_reproduce=exps_cfg.get("global_non_reproducible", []),
        models=params.get("info", {}), composite_categories=params.get("composite_categories"), kept_terms=params.get("kept_terms"),
        limitations=[
            "No ground truth exists for analog quality: the -1..+1 scale is set by draft expert anchors (all still marked needs_review).",
            "Temperature and sun angle never overlap between Moon and Earth: thermal scores are closest-approach (a latitude ranking), not similarity, and are left out of the overall score.",
            "Dust/regolith and sterility compare Earth sites with known lunar constants (Apollo regolith, sterile Moon), so they rank Earth sites the same way for every Moon target and are left out of the overall score.",
            "Earth climate layers are coarser (5-50 km) than the 10 km tiles; Antarctica lacks land-cover, soil and rainfall layers.",
            "Lunar geology comes from a 1:5M map plus the known composition of mare vs highland crust, not per-tile mineralogy.",
            "Thermal scores for the Tranquillitatis pit are extrapolated (thermal anchors are all polar).",
            "Gravity, vacuum, radiation and 25-110 K shadowed temperatures cannot be reproduced on Earth and are never scored."])
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*.json")) / 1e6
    print(f"web bundle -> {out}  ({len(list(out.rglob('*.json')))} files, {size:.1f} MB)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
