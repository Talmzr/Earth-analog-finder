"""Step 1: load and validate the data before any model touches it.  python -m ml.step1_validate"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C


def main():
    C.ensure_dirs()
    issues, info = [], {}
    moon, earth, zones = C.load_tables()
    man = json.loads((C.ML / "manifest.json").read_text())
    info["rows"] = dict(moon_tiles=len(moon), earth_tiles=len(earth), moon_zones=len(zones),
                        moon_by_region=moon["region"].value_counts().to_dict())
    for k, n in (("moon_rows", len(moon)), ("earth_rows", len(earth))):
        if man.get(k) != n:
            issues.append(f"manifest {k}={man.get(k)} but table has {n}")
    for name, df in (("moon", moon), ("earth", earth)):
        if df["patch_id"].duplicated().any():
            issues.append(f"{name}: {int(df['patch_id'].duplicated().sum())} duplicate patch_id")

    # cross-body columns: present on both sides, comparable ranges
    xm = sorted(c for c in moon.columns if c.startswith("x_"))
    xe = sorted(c for c in earth.columns if c.startswith("x_"))
    shared = sorted(set(xm) & set(xe))
    info["x_shared"] = shared
    info["x_moon_only"] = sorted(set(xm) - set(xe))
    info["x_earth_only"] = sorted(set(xe) - set(xm))
    polar = moon[moon["region"] == "south_pole"]
    rng = {}
    for c in shared:
        if not pd.api.types.is_numeric_dtype(moon[c]) or not pd.api.types.is_numeric_dtype(earth[c]):
            continue
        m, e = polar[c].dropna(), earth[c].dropna()
        if not len(m) or not len(e):
            issues.append(f"{c}: empty on one body")
            continue
        overlap = float(((m >= e.quantile(.01)) & (m <= e.quantile(.99))).mean())
        rng[c] = dict(moon_polar=[round(float(m.min()), 2), round(float(m.median()), 2), round(float(m.max()), 2)],
                      earth=[round(float(e.min()), 2), round(float(e.median()), 2), round(float(e.max()), 2)],
                      moon_cov=round(float(polar[c].notna().mean()), 3), earth_cov=round(float(earth[c].notna().mean()), 3),
                      moon_in_earth_range=round(overlap, 3))
    info["x_ranges"] = rng

    # scoring entities and anchors
    ent = C.build_entities(moon, zones)
    info["entities"] = dict(total=len(ent), zones=int((ent.kind == "zone").sum()), target_tiles=int((ent.kind == "tile").sum()))
    need = C.X_COLS + C.TERRAIN_COLS + ["moon_geo_unit"]
    miss = {c: ent.loc[ent[c].isna(), "entity_id"].tolist() for c in need if ent[c].isna().any()}
    info["entity_missing"] = miss
    a = C.load_anchors()
    bad_m = sorted(set(a["entity_id"]) - set(ent["entity_id"]))
    bad_e = sorted(set(a["earth_site"]) - set(earth["patch_id"]))
    if bad_m:
        issues.append(f"anchors reference unknown Moon entities: {bad_m}")
    if bad_e:
        issues.append(f"anchors reference unknown Earth tiles: {bad_e}")
    info["anchors"] = dict(total=len(a), by_category=a["cat"].value_counts().to_dict(),
                           strong_pos=a[a.label >= .5]["cat"].value_counts().to_dict(),
                           strong_neg=a[a.label <= -.5]["cat"].value_counts().to_dict(),
                           unreviewed=int((a.get("needs_review", 1) == 1).sum()))

    # image arrays
    ids, body, X, vf = C.load_terrain_arrays()
    info["arrays"] = dict(total=len(ids), shape=list(X.shape[1:]), by_body=pd.Series(body).value_counts().to_dict())
    tile_ids = set(moon["patch_id"]) | set(earth["patch_id"]) | set(ent.loc[ent.kind == "zone", "entity_id"])
    have = set(ids)
    no_arr_moon = sorted(set(moon["patch_id"]) - have)
    no_arr_earth = sorted(set(earth["patch_id"]) - have)
    no_arr_zone = sorted(set(ent.loc[ent.kind == "zone", "entity_id"]) - have)
    info["arrays_missing"] = dict(moon=len(no_arr_moon), earth=len(no_arr_earth), zones=len(no_arr_zone),
                                  not_in_tables=len(have - tile_ids))
    if no_arr_moon or no_arr_earth or no_arr_zone:
        issues.append(f"tiles without a 128x128 patch: moon {len(no_arr_moon)}, earth {len(no_arr_earth)}, zones {len(no_arr_zone)}")
    if not np.isfinite(X.astype(np.float32)).all():
        issues.append("non-finite values in terrain arrays")
    info["earth_low_valid_patches"] = int((vf[body == "earth"] < 0.7).sum())

    # coverage of the Earth columns the scorers read
    cols = ["lst_day_max_k", "lst_night_min_k", "power_t2m_c", "pf_frac_pct", "soil_moist_mean", "precip_best_mm_yr",
            "ndvi_mean", "aot_mean", "sg_sand_pct", "sg_bdod_g_cm3", "wc_frac_bare_sparse", "mstrat_text_all",
            "trav_dist_flat_to_ice_km", "trav_flat_bare_frac"]
    info["earth_coverage"] = {c: round(float(earth[c].notna().mean()), 3) for c in cols if c in earth}
    ant = earth["center_lat"] < -60
    info["antarctic_tiles"] = int(ant.sum())
    info["antarctic_coverage"] = {c: round(float(earth.loc[ant, c].notna().mean()), 3) for c in
                                  ("wc_frac_bare_sparse", "sg_sand_pct", "ndvi_mean", "soil_moist_mean", "aot_mean") if c in earth}

    # differences from the brief
    info["brief_vs_data"] = [
        f"Moon tiles: brief says 3,609 x 76; table has {len(moon)} x {moon.shape[1]} "
        f"({info['rows']['moon_by_region']}): mid-latitude tiles were added for encoder variety and the pit target.",
        f"Earth tiles: brief says 4,310 x 142; table has {len(earth)} x {earth.shape[1]}.",
        f"Anchors: brief says 47; file has {len(a)} draft anchors (all needs_review).",
        f"Image patches: brief says 64x64; using {X.shape[-1]}x{X.shape[-1]} (78 m/px).",
        f"Scoring targets: brief says 29 locations; data has {int((ent.kind == 'zone').sum())} zones (10 regions x landing/ice) "
        f"+ {int((ent.kind == 'tile').sum())} target tiles (9 polar + Tranquillitatis pit) = {len(ent)} Moon entities.",
    ]
    info["issues"] = issues
    (C.RES / "step1_validation.json").write_text(json.dumps(info, indent=1, default=str))

    print("== STEP 1: data validation ==")
    print("rows:", info["rows"])
    print("entities:", info["entities"], "| anchors:", info["anchors"]["total"], info["anchors"]["by_category"])
    print("arrays:", info["arrays"], "| missing:", info["arrays_missing"], "| low-valid Earth patches:", info["earth_low_valid_patches"])
    print(f"shared x_ columns ({len(shared)}):", shared)
    print("moon-only x_:", info["x_moon_only"], "| earth-only x_:", info["x_earth_only"])
    print("\ncross-body ranges (min / median / max) and share of Moon polar tiles inside Earth's 1-99% range:")
    for c, r in rng.items():
        print(f"  {c:<24} moon {r['moon_polar']}  earth {r['earth']}  overlap {r['moon_in_earth_range']:.0%}  cov M/E {r['moon_cov']:.0%}/{r['earth_cov']:.0%}")
    print("\nEarth coverage:", info["earth_coverage"])
    print("Antarctic tiles:", info["antarctic_tiles"], info["antarctic_coverage"])
    print("entity columns with gaps:", {k: len(v) for k, v in miss.items()} or "none")
    print("\nbrief vs data:")
    for s in info["brief_vs_data"]:
        print("  -", s)
    print("\nISSUES:", issues or "none")


if __name__ == "__main__":
    main()
