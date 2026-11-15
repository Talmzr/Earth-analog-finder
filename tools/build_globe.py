"""Build web/globe.html: the 3-D globe front end with the final scoring data embedded (one self-contained file).

    python tools/build_globe.py            (run after ml.score_all and ml.export_web)

Inputs
  frontend/globe.template.html   page code; /*@DATA@*/, /*@MOON_TEX@*/ and /*@ASSESS@*/ are filled in here
  frontend/assess.js             Similarity Assessment page code
  frontend/moon_textures.json    Moon colour / DEM / south-polar textures (data URIs)
  data/ml/*.csv, config/*        Earth tiles, Moon tiles and Artemis zones, named sites, targets, experiments
  results/score_matrices.npz     calibrated category scores, composite, experiment readiness, light-twin month
  results/calibration_summary.csv, web/data/meta.json   calibration quality, caveats, limitations
Output
  web/globe.html                 TILE_DATA, EARTH_X, EARTH_SUN, MOON_DB, META, SCORE_PACK embedded as JS constants
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import common as C  # noqa: E402
from ml import geo_text as G  # noqa: E402
from ml.export_web import HOW, LABELS, percentile  # noqa: E402
from ml.score_all import COMPOSITE_CATS, KNOWN_LIFE, earth_monthly_sun  # noqa: E402

FRONT = ROOT / "frontend"
SHORT = {"terrain": "Terrain", "geology": "Geology", "thermal": "Temperature", "illumination": "Sunlight",
         "water_ice": "Water & ice", "dust": "Dust & regolith", "biology": "Sterility", "caves": "Caves",
         "operations": "Operations"}
TARGET_INVARIANT = {"dust", "biology"}          # compared with lunar constants: same ranking for every Moon target


def num(v, nd=1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    v = round(v, nd)
    return int(v) if nd == 0 else v


def pctv(v, nd=1):
    """fraction 0-1 -> percent"""
    v = num(v, 6)
    return None if v is None else num(v * 100, nd)


def txt(v, n=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    return s[:n] if n else s


def kv(context, key):
    m = re.search(rf"(?:^|;\s*){key}=([^;]*)", context or "")
    return m.group(1).strip() if m else ""


def js_const(name, obj):
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"const {name} = {s.replace('</', '<' + chr(92) + '/')};"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.int8).tobytes()).decode("ascii")


def nibbles(codes):
    """readiness codes -1..3 -> 4 bits each (code + 1), two per byte, low nibble first"""
    f = (np.asarray(codes, np.int16).ravel() + 1).astype(np.uint8)
    if f.size % 2:
        f = np.append(f, 0)
    return base64.b64encode((f[0::2] | (f[1::2] << 4)).astype(np.uint8).tobytes()).decode("ascii")


def to_i8(a, scale=100.0):
    a = np.asarray(a, float)
    out = np.full(a.shape, -128, np.int16)
    ok = np.isfinite(a)
    out[ok] = np.clip(np.round(a[ok] * scale), -127, 127)
    return out.astype(np.int8)


# ----------------------------------------------------------------------------- Earth
def earth_blocks(earth, cave_rule):
    E = C.prepare_earth(earth)
    sites = pd.read_csv(C.CFG / "earth_sites.csv").set_index("id")
    gvp_path = C.DATA / "catalogs" / "earth_volcanoes_gvp.csv"
    gvp = pd.read_csv(gvp_path).drop_duplicates("Volcano_Number").set_index("Volcano_Number") if gvp_path.exists() else None
    src_list = ["named_site", "impact_structure", "volcano", "landcover_sample"]
    strata = sorted(earth["stratum"].astype(str).unique())
    ice = earth["x_surface_ice_frac"].to_numpy(float)
    et = [G.earth_text(r) for r in earth.to_dict("records")]
    ec = [G.earth_classes(t, s, i, G.earth_primary_text(r)) for t, s, i, r in zip(et, earth["source"], ice, earth.to_dict("records"))]
    anorth = [2 if e.get("anorth") else 1 if e.get("anorth_minor") else 0 for e in ec]

    rows, xrows = [], []
    for i, r in enumerate(earth.to_dict("records")):
        e = E.iloc[i]
        ctx = str(r.get("context") or "")
        src = r["source"]
        country, xinfo = "", ""
        if src == "named_site":
            country = kv(ctx, "country") or (sites.loc[r["patch_id"], "country"] if r["patch_id"] in sites.index else "")
        elif src == "volcano":
            country = kv(ctx, "country")
            vtype, summ = kv(ctx, "type"), kv(ctx, "summary")
            if gvp is not None:
                try:
                    g = gvp.loc[int(str(r["patch_id"]).split("_")[1])]
                    vtype = txt(g.get("Primary_Volcano_Type")) or vtype
                    summ = txt(g.get("Geological_Summary")) or summ
                except (ValueError, KeyError, IndexError):
                    pass
            summ = (summ or "").replace("�", "")
            if len(summ) > 240:                               # cut at a sentence end
                cut = summ.rfind(". ", 0, 240)
                summ = summ[:cut + 1] if cut > 80 else summ[:240].rsplit(" ", 1)[0] + "…"
            xinfo = f"{vtype}|{summ}" if (vtype or summ) else ""
        elif src == "impact_structure":
            loc = kv(ctx, "location")
            country = loc.split(",")[-1].strip() if loc else ""
            d = num(kv(ctx, "diameter_km"), 1)
            xinfo = f"D={d:g} km" if d is not None else ""
        rows.append([num(r["center_lat"], 2), num(r["center_lon"], 2), src_list.index(src), strata.index(str(r["stratum"])),
                     txt(r["name"]) or "", country or "",
                     num(r["slope500_mean_deg"], 1), num(r["relief_m"], 0), num(r["elev_mean_m"], 0), num(r["tri500_mean_m"], 1),
                     txt(r["mstrat_lith"], 90) or "", num(r["sg_sand_pct"], 0), num(r["sg_silt_pct"], 0), num(r["sg_clay_pct"], 0),
                     num(r["sg_bdod_g_cm3"], 2), txt(r["mstrat_age"], 40) or "", int(r["permafrost_flag"] or 0) if pd.notna(r["permafrost_flag"]) else 0,
                     num(r["x_surface_ice_frac"], 3) or 0, num(r["soil_moist_mean"], 1), num(e["precip"], 0),
                     num(r["x_temp_max_k"], 0), num(r["x_temp_min_k"], 0), num(e["bare_f"], 2) or 0, num(r["x_surface_ice_frac"], 2) or 0,
                     xinfo])
        tags = kv(ctx, "tags") if src == "named_site" else ""
        notes = kv(ctx, "notes") if src == "named_site" else ""
        xrows.append([
            num(r["slope500_p90_deg"], 1), num(r["frac_slope500_gt10"], 3), num(r["frac_slope500_gt20"], 3), num(r["frac_elev_zero"], 3),
            num(r["mstrat_t_age_ma"], 0), num(r["mstrat_b_age_ma"], 0), num(r["sg_cfvo_pct"], 0), num(r["pf_frac_pct"], 0),
            num(r["pf_magt_10m_c"], 1), num(r["x_frozen_ground_frac"], 3), num(r["x_surface_ice_frac"], 3),
            num(r["trav_dist_flat_to_ice_km"], 1), int(r["trav_dist_gt10_flag"]) if pd.notna(r["trav_dist_gt10_flag"]) else None,
            r["patch_id"], num(r["x_temp_mean_k"], 0), num(r["x_noon_sun_max_deg"], 1), num(r["x_grazing_sun_frac"], 3),
            num(r["x_polar_night_frac"], 3), num(e["ndvi_f"], 3), num(r["wc_frac_tree"], 3), num(r["wc_frac_built"], 3),
            num(r["wc_frac_water"], 3), num(r["wc_frac_crop"], 3), num(r["wc_frac_moss_lichen"], 3), num(r["wc_frac_bare_sparse"], 3),
            num(e["bare_f_site"], 3), num(r["trav_flat_bare_frac"], 3), int(bool(e["flag_tephra"])), KNOWN_LIFE.get(r["patch_id"]),
            anorth[i], num(cave_rule[i], 1), txt(r.get("precip_best_src")), tags or None, notes or None])
    xcols = ["slope500_p90_deg", "frac_slope500_gt10", "frac_slope500_gt20", "frac_elev_zero", "mstrat_t_age_ma", "mstrat_b_age_ma",
             "sg_cfvo_pct", "pf_frac_pct", "pf_magt_10m_c", "x_frozen_ground_frac", "x_surface_ice_frac", "trav_dist_flat_to_ice_km",
             "trav_dist_gt10_flag", "patch_id", "x_temp_mean_k", "x_noon_sun_max_deg", "x_grazing_sun_frac", "x_polar_night_frac",
             "ndvi_mean", "wc_frac_tree", "wc_frac_built", "wc_frac_water", "wc_frac_crop", "wc_frac_moss_lichen", "wc_frac_bare_sparse",
             "bare_f_site", "trav_flat_bare_frac", "flag_tephra", "known_life", "anorth", "cave_rule", "precip_src", "site_tags", "site_notes"]
    assert len(xcols) == len(xrows[0])
    noon, graz = earth_monthly_sun(earth)
    sun = dict(noon=[num(v, 1) for v in noon.ravel()], graz=[num(v, 2) for v in graz.ravel()])
    return dict(s=src_list, t=strata, d=rows), dict(c=xcols, d=xrows), sun


# ----------------------------------------------------------------------------- Moon
def moon_db(moon, zones, ent, cats, comp, mt_by_id):
    units = pd.read_csv(C.DATA / "catalogs" / "moon_geologic_units.csv").drop_duplicates("unit").set_index("unit")
    tg = pd.read_csv(C.CFG / "targets.csv").set_index("id")
    mt = moon.set_index("patch_id")
    zi = zones.set_index(["region_id", "zone"])
    targets = {}
    for i, e in enumerate(ent.to_dict("records")):
        tid = e["entity_id"]
        geo = mt.loc[e["geo_tile"]] if e["geo_tile"] in mt.index else None
        if e["kind"] == "zone":
            z = zi.loc[(e["region_id"], e["zone"])]
            src = z
            region, region_name = e["region_id"], str(z["region_name"]).replace(" (not an Artemis III region)", "")
            label = "Landing zone" if e["zone"] == "landing" else "Ice zone"
            area = num(z["area_km2"], 2)
            l2i_min, vis50, best_vis = num(z["landing_to_ice_min_km"], 2), num(z["landing_vis50_km2"], 2), pctv(z["best_landing_vis"], 0)
        else:
            src = mt.loc[e["target_id"]]
            regs = zones.loc[zones["anchor_target"] == e["target_id"], "region_id"].unique().tolist()
            region = regs[0] if regs else "tranquillitatis_pit"
            region_name = str(tg.loc[e["target_id"], "name"]) if e["target_id"] in tg.index else e["target_id"]
            label, area, l2i_min, vis50, best_vis = "10 km target tile", None, None, None, None
        unit = txt(geo["moon_geo_unit"]) if geo is not None else None
        u = units.loc[unit] if unit in units.index else None
        g = lambda c: (geo[c] if geo is not None and c in geo.index else np.nan)  # noqa: E731
        s = lambda c: (src[c] if c in src.index else np.nan)  # noqa: E731
        m = mt_by_id.get(tid, {})
        order = C.top_order(comp[i], DUP)[:25]                          # duplicates of named sites left out
        targets[tid] = dict(
            id=tid, idx=i, region=region, regionName=region_name, kind=e["kind"], zone=e["zone"], label=label,
            name=f"{region_name} · {label.lower()}", lat=num(e["lat"], 4), lon=num(e["lon"], 4), placement=e["placement_method"],
            area_km2=area, pit=bool(e["is_pit"]),
            slope=num(e["slope500_mean_deg"], 1), slope_p90=num(e["slope500_p90_deg"], 1), steep10=pctv(e["frac_slope500_gt10"], 0),
            steep20=pctv(e["frac_slope500_gt20"], 0), tri=num(e["tri500_mean_m"], 0), relief=num(e["relief_m"], 0),
            elev=num(s("elev_mean_m"), 0), elev_sd=num(e["elev_std_m"], 0),
            haz_gt10=pctv(e["haz_gt10"], 0), haz_gt20=pctv(e["haz_gt20"], 0), haz_p90=num(e["haz_p90"], 1),
            geo_unit=unit, geo_name=txt(g("moon_geo_unit_name")) or (txt(u["name"]) if u is not None else None),
            geo_age=txt(g("moon_geo_unit_age")) or (txt(u["age"]) if u is not None else None),
            geo_desc=txt(u["description"]) if u is not None else None, geo_interp=txt(u["interpretation"]) if u is not None else None,
            mix_crater=pctv(g("geo_frac_crater"), 0), mix_basin=pctv(g("geo_frac_basin"), 0), mix_plains=pctv(g("geo_frac_plains"), 0),
            mix_terra=pctv(g("geo_frac_terra"), 0), mix_mare=pctv(g("geo_frac_mare"), 0),
            ice1m=pctv(s("dvn_frac_ice_within_1m"), 1), ice_stable=pctv(s("dvn_frac_ice_stable"), 1),
            coldtrap=pctv(s("dvn_frac_coldtrap_110k"), 1), surf_ice=pctv(s("dvn_frac_ice_surface"), 1), psr=pctv(s("lola_frac_psr"), 1),
            ice_depth=num(s("dvn_ice_depth_median_m"), 2), landing_to_ice_km=num(e["landing_to_ice_median_km"], 2),
            landing_to_ice_min_km=l2i_min, landing_vis50_km2=vis50, best_landing_vis=best_vis,
            sun=pctv(s("lola_avg_sun_vis_mean"), 0), sun_p90=pctv(s("lola_avg_sun_vis_p90"), 0), sun_max=pctv(s("lola_avg_sun_vis_max"), 0),
            sunlit50=pctv(s("lola_frac_sunlit_gt50pct"), 0), noon_max=num(e["x_noon_sun_max_deg"], 1),
            grazing=pctv(e["x_grazing_sun_frac"], 0), polar_night=pctv(e["x_polar_night_frac"], 0),
            t_min=num(e["x_temp_min_k"], 0), t_mean=num(e["x_temp_mean_k"], 0), t_max=num(e["x_temp_max_k"], 0),
            t_swing=num(e["x_temp_swing_k"], 0),
            composite_weights=m.get("composite_weights"), relevant=m.get("relevant_experiments", []),
            n_sufficient=m.get("earth_tiles_sufficient", {}),
            top=[dict(tile=int(j), composite=num(comp[i, j], 2)) for j in order])
    regions = []
    for rid, z in zones.groupby("region_id", sort=False):
        z0 = z.iloc[0]
        other = "not an Artemis III region" in str(z0["region_name"])
        regions.append(dict(
            id=rid, name=str(z0["region_name"]).replace(" (not an Artemis III region)", ""),
            kind="other" if other else "a2024", lat=num(z0["center_lat"], 4), lon=num(z0["center_lon"], 4),
            placement=z0["placement_method"], source=txt(z0["source"]),
            targets=[f"{rid}__landing", f"{rid}__ice", f"{z0['anchor_target']}__tile"]))
    pit = tg.loc["moon_tranquillitatis_pit"]
    regions.append(dict(id="tranquillitatis_pit", name="Mare Tranquillitatis pit", kind="pit", lat=num(pit["lat"], 4),
                        lon=num(pit["lon"], 4), placement="anchor_tile", source=txt(pit["source"]),
                        targets=["moon_tranquillitatis_pit__tile"]))
    for k, r in enumerate(regions):
        r["idx"] = k
        assert all(t in targets for t in r["targets"]), r
    return dict(regions=regions, targets=targets)


def main():
    Z = np.load(C.RES / "score_matrices.npz", allow_pickle=False)
    cats, ent_ids, tile_ids = Z["categories"].tolist(), Z["entity_id"].tolist(), Z["earth_tile_id"].tolist()
    cal, comp, ready, exps = Z["calibrated"], Z["composite"], Z["readiness"], Z["experiments"].tolist()
    twin = Z["light_twin_month"]
    moon, earth, zones = C.load_tables()
    earth = earth.set_index("patch_id").loc[tile_ids].reset_index()
    global DUP
    dup = DUP = C.duplicate_of_named(earth)
    earth["name"] = C.display_names(earth)                              # readable names for random samples                                   # same place as a named site
    ent = C.build_entities(moon, zones).set_index("entity_id").loc[ent_ids].reset_index()
    meta_web = json.loads((ROOT / "web" / "data" / "meta.json").read_text(encoding="utf-8"))
    mt_by_id = {t["id"]: t for t in json.loads((ROOT / "web" / "data" / "moon_targets.json").read_text(encoding="utf-8"))}
    exps_cfg = json.loads((C.CFG / "experiments.json").read_text(encoding="utf-8"))
    summ = pd.read_csv(C.RES / "calibration_summary.csv").set_index("category")

    pit_i = [i for i, e in enumerate(ent_ids) if e == "moon_tranquillitatis_pit__tile"][0]
    cave_rule = cal[cats.index("caves"), pit_i]
    tile_data, earth_x, earth_sun = earth_blocks(earth, cave_rule)
    mdb = moon_db(moon, zones, ent, cats, comp, mt_by_id)

    cat_meta = []
    for c in cats:
        row = summ.loc[c] if c in summ.index else {}
        wc = next((x for x in meta_web["categories"] if x["id"] == c), {})
        cat_meta.append(dict(id=c, label=LABELS[c], short=SHORT[c], how=HOW[c], cap=wc.get("cap"),
                             n_anchors=int(row.get("n_anchors", 0)), mae=num(row.get("mae"), 3), loo_mae=num(row.get("mae_loo"), 3),
                             rank=num(row.get("spearman"), 3), loo_rank=num(row.get("spearman_loo"), 3),
                             flagged=int(row.get("n_flagged", 0)), in_composite=c in COMPOSITE_CATS,
                             target_invariant=c in TARGET_INVARIANT))
    pre = {e["id"]: e.get("moon_precondition") for e in exps_cfg["experiments"]}
    meta = dict(
        generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), scores_generated=meta_web.get("generated_utc"),
        n_targets=len(ent_ids), n_tiles=len(tile_ids), categories=cat_meta, composite_categories=list(COMPOSITE_CATS),
        experiments=[dict(id=e["id"], name=e["name"], required=e["required"], supporting=e["supporting"],
                          also=e.get("also_account_for", []), precondition=pre.get(e["id"])) for e in meta_web["experiments"]],
        cannot=meta_web["earth_cannot_reproduce"], limitations=meta_web["limitations"],
        dust_ref=meta_web.get("models", {}).get("dust_reference", dict(fines_pct=45.0, bulk_density=1.5, coarse_pct=5.0)),
        n_anchors=int(summ["n_anchors"].sum()), anchors_flagged=int(summ["n_flagged"].sum()))

    S = np.concatenate([cal.transpose(1, 0, 2), comp[:, None, :]], axis=1)          # (M, cats + composite, E)
    th = np.stack([percentile(cal[cats.index("thermal"), i]) for i in range(len(ent_ids))])
    pack = dict(ids=ent_ids, cats=cats, exps=exps, n=len(tile_ids),
                s=b64(to_i8(S)), r=nibbles(ready.transpose(1, 0, 2)), p=b64(to_i8(th)), m=b64(twin.astype(np.int8)))

    data = "\n".join([
        "// ═══ Final data, generated by tools/build_globe.py on " + meta["generated"] + " (do not edit by hand) ═══",
        "// TILE_DATA row: [0]lat [1]lon [2]src_idx [3]stratum_idx [4]name [5]country [6]slope [7]relief [8]elev [9]rough [10]lith",
        "//   [11]sand [12]silt [13]clay [14]bdod [15]geo_age [16]pf_flag [17]ice_frac [18]soil_moist [19]precip [20]tmax [21]tmin",
        "//   [22]bare [23]snow [24]extra_info",
        js_const("TILE_DATA", tile_data), js_const("EARTH_X", earth_x), js_const("EARTH_SUN", earth_sun),
        js_const("MOON_DB", mdb), js_const("META", meta), js_const("SCORE_PACK", pack)])
    tpl = (FRONT / "globe.template.html").read_text(encoding="utf-8")
    assess = (FRONT / "assess.js").read_text(encoding="utf-8")
    tex = (FRONT / "moon_textures.json").read_text(encoding="utf-8").strip()
    for mark in ("/*@DATA@*/", "/*@MOON_TEX@*/null", "/*@ASSESS@*/"):
        assert tpl.count(mark) == 1, mark
    html = tpl.replace("/*@DATA@*/", data).replace("/*@MOON_TEX@*/null", tex).replace("/*@ASSESS@*/", assess)
    out = ROOT / "web" / "globe.html"                                   # the globe; the 2-D map is web/map.html
    out.write_text(html, encoding="utf-8")
    # site root opens the globe
    (ROOT / "web" / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Earth Analog Finder</title>'
                                             '<meta http-equiv="refresh" content="0; url=globe.html"><a href="globe.html">Open the globe</a>' + chr(10),
                                             encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB): {len(tile_ids)} Earth tiles, "
          f"{len(mdb['regions'])} Moon regions, {len(ent_ids)} Moon targets, {len(cats)} categories, {len(exps)} experiments")



if __name__ == "__main__":
    main()
