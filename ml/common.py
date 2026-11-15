"""Shared loading, Moon scoring entities, Earth derived columns, spatial blocks, anchors, calibration helpers."""
from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, CFG = ROOT / "data", ROOT / "config"
ML, PATCH = DATA / "ml", DATA / "patches"
MODELS, EMB, RES = ROOT / "models", ROOT / "embeddings", ROOT / "results"
R_MOON = 1737400.0

# output category name -> name used in config/categories.json, experiments.json and anchors.csv
# (dust_regolith, biology_sterility, caves_subsurface were removed 2026-11-15: no Moon data; their anchors are ignored)
CATS = {"terrain": "terrain", "geology": "geology", "thermal": "thermal_illumination", "water_ice": "water_ice",
        "operations": "ops_landing_mobility"}
CFG2OUT = {v: k for k, v in CATS.items()}

TERRAIN_COLS = ["relief_m", "elev_std_m", "slope500_mean_deg", "slope500_p90_deg", "frac_slope500_gt10",
                "frac_slope500_gt20", "tri500_mean_m"]
X_COLS = ["x_temp_max_k", "x_temp_mean_k", "x_temp_min_k", "x_temp_swing_k", "x_grazing_sun_frac",
          "x_noon_sun_max_deg", "x_polar_night_frac", "x_frozen_ground_frac", "x_surface_ice_frac"]
MOON_EXTRA = ["dvn_frac_coldtrap_110k", "lola_frac_psr", "dvn_frac_ice_within_1m", "dvn_frac_ice_stable"]
ARRAY_FILES = [("moon_patch_arrays_128.npz", "moon"), ("moon_midlat_arrays_128.npz", "moon"),
               ("moon_zone_arrays_128.npz", "zone"), ("earth_patch_arrays_128.npz", "earth")]


def log(*a):
    import time
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_dirs():
    for d in (MODELS, EMB, RES):
        d.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Parquet when an engine is available (Colab), otherwise gzip CSV with the same stem."""
    path = Path(path)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as ex:  # noqa: BLE001  (no pyarrow on this machine)
        alt = path.with_suffix(".csv.gz")
        df.to_csv(alt, index=False)
        log(f"parquet unavailable ({type(ex).__name__}); wrote {alt.name}")
        return alt


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            pass
    alt = path.with_suffix(".csv.gz")
    return pd.read_csv(alt)


# ----------------------------------------------------------------------------- tables
def load_tables():
    moon = pd.read_csv(ML / "moon_tiles.csv", low_memory=False)
    earth = pd.read_csv(ML / "earth_tiles.csv", low_memory=False)
    zones = pd.read_csv(ML / "moon_zones.csv")
    if "region" not in moon:
        moon["region"] = "south_pole"
    earth = fill_volcano_context(earth)
    return moon, earth, zones


def fill_volcano_context(earth: pd.DataFrame) -> pd.DataFrame:
    """Some volcano tiles lost their catalog text upstream (a missing field blanked the whole string).
    Rebuild it from the Smithsonian GVP catalog so geology text, flags and the cave rule can use it."""
    gvp_path = DATA / "catalogs" / "earth_volcanoes_gvp.csv"
    miss = earth["context"].isna() & earth["source"].eq("volcano")
    if not miss.any() or not gvp_path.exists():
        return earth
    g = pd.read_csv(gvp_path).drop_duplicates("Volcano_Number").set_index("Volcano_Number")

    def build(pid):
        try:
            r = g.loc[int(str(pid).split("_")[1])]
        except Exception:  # noqa: BLE001
            return np.nan
        f = lambda k: "" if pd.isna(r.get(k)) else str(r.get(k))  # noqa: E731
        return (f"volcano; type={f('Primary_Volcano_Type')}; landform={f('Volcanic_Landform')}; rock={f('Major_Rock_Type')}; "
                f"tectonic={f('Tectonic_Setting')}; last_eruption={f('Last_Eruption_Year')}; elev_m={f('Elevation')}; "
                f"country={f('Country')}; summary={f('Geological_Summary')}")
    earth = earth.copy()
    earth.loc[miss, "context"] = earth.loc[miss, "patch_id"].map(build)
    return earth


def stereo_south(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    rho = 2 * R_MOON * np.tan(np.radians(90 + lat) / 2)
    return rho * np.sin(np.radians(lon)), rho * np.cos(np.radians(lon))


def build_entities(moon: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """The Moon things every Earth tile is scored against:
       20 Artemis zones (region landing / ice) + the 10 target tiles (9 polar anchors + Tranquillitatis pit).
       Terrain columns of a zone describe the 10 km window at its centroid (same definition as tiles);
       haz_* columns describe the zone pixels themselves (used by the operations category)."""
    targets = pd.read_csv(CFG / "targets.csv")
    tags = dict(zip(targets["id"], targets["tags"].fillna("")))
    zp_path = ML / "moon_zone_patch_terrain.csv"
    zp = pd.read_csv(zp_path).set_index("entity_id") if zp_path.exists() else None
    grid = moon[(moon["region"] == "south_pole") & (moon["is_target"] == 0)]
    gx, gy = grid["x_m"].to_numpy(float), grid["y_m"].to_numpy(float)
    rows = []
    for z in zones.to_dict("records"):
        eid = f"{z['region_id']}__{z['zone']}"
        d = dict(entity_id=eid, kind="zone", zone=z["zone"], region_id=z["region_id"], target_id=z["anchor_target"],
                 name=f"{z['region_name']} ({z['zone']} zone)", lat=z["zone_centroid_lat"], lon=z["zone_centroid_lon"],
                 placement_method=z.get("placement_method", ""), tags=tags.get(z["anchor_target"], ""))
        for c in X_COLS + MOON_EXTRA:
            d[c] = z.get(c, np.nan)
        d.update(haz_gt10=z["frac_slope500_gt10"], haz_gt20=z["frac_slope500_gt20"], haz_p90=z["slope500_p90_deg"])
        for c in TERRAIN_COLS:
            d[c] = zp.loc[eid, f"patch_{c}"] if zp is not None and eid in zp.index else z.get(c, np.nan)
        d["landing_to_ice_median_km"] = z.get("landing_to_ice_median_km", np.nan)
        zx, zy = stereo_south(z["zone_centroid_lat"], z["zone_centroid_lon"])
        near = grid.iloc[int(np.argmin(np.hypot(gx - zx, gy - zy)))]
        d.update(moon_geo_unit=near["moon_geo_unit"], geo_frac_mare=near.get("geo_frac_mare", 0.0),
                 geo_tile=near["patch_id"])
        rows.append(d)
    l2i = zones.groupby("anchor_target")["landing_to_ice_median_km"].mean().to_dict()
    for t in moon[moon["is_target"] == 1].to_dict("records"):
        d = dict(entity_id=f"{t['patch_id']}__tile", kind="tile", zone="tile", region_id="", target_id=t["patch_id"],
                 name=f"{t['patch_id']} (10 km target tile)", lat=t["center_lat"], lon=t["center_lon"],
                 placement_method="anchor_tile", tags=tags.get(t["patch_id"], ""))
        for c in X_COLS + MOON_EXTRA + TERRAIN_COLS:
            d[c] = t.get(c, np.nan)
        d.update(haz_gt10=t["frac_slope500_gt10"], haz_gt20=t["frac_slope500_gt20"], haz_p90=t["slope500_p90_deg"])
        d["landing_to_ice_median_km"] = l2i.get(t["patch_id"], np.nan)
        d.update(moon_geo_unit=t["moon_geo_unit"], geo_frac_mare=t.get("geo_frac_mare", 0.0), geo_tile=t["patch_id"])
        rows.append(d)
    ent = pd.DataFrame(rows)
    ent["is_pit"] = ent["target_id"].eq("moon_tranquillitatis_pit") & ent["kind"].eq("tile")
    return ent


# ----------------------------------------------------------------------------- Earth derived columns
def _nsum(df, cols):
    cols = [c for c in cols if c in df]
    out = df[cols].sum(axis=1, min_count=1) if cols else pd.Series(np.nan, index=df.index)
    return out


def text_blob(df, cols):
    cols = [c for c in cols if c in df]
    return df[cols].fillna("").astype(str).agg(" ; ".join, axis=1).str.lower() if cols else pd.Series("", index=df.index)


def prepare_earth(earth: pd.DataFrame) -> pd.DataFrame:
    """Adds unit-converted columns, documented fallbacks for Antarctica, and text flags from categories.json."""
    cfg = json.loads((CFG / "categories.json").read_text(encoding="utf-8"))
    e = earth.copy()
    lat = e["center_lat"]
    e["soil_moist_frac"] = e["soil_moist_mean"] / 100.0                      # AMSR2 % -> m3/m3
    e["precip"] = e["precip_best_mm_yr"] if "precip_best_mm_yr" in e else e.get("precip_mm_yr")
    e["fines_pct"] = e["sg_silt_pct"] + e["sg_clay_pct"]
    veg = _nsum(e, ["wc_frac_tree", "wc_frac_shrub", "wc_frac_grass", "wc_frac_crop", "wc_frac_moss_lichen",
                    "wc_frac_wetland", "wc_frac_mangrove"])
    e["veg_frac"] = veg.where(veg.notna(), np.where(lat < -60, 0.0, np.nan))  # no WorldCover south of 60S
    e["ndvi_f"] = e["ndvi_mean"].where(e["ndvi_mean"].notna(), np.where(lat < -60, 0.0, np.nan))
    e["tree_f"] = e["wc_frac_tree"].where(e["wc_frac_tree"].notna(), np.where(lat < -60, 0.0, np.nan))
    ice = e["x_surface_ice_frac"] if "x_surface_ice_frac" in e else e["wc_frac_snow_ice"]
    e["bare_f"] = e["wc_frac_bare_sparse"].where(e["wc_frac_bare_sparse"].notna(),
                                                  np.where(lat < -60, (1 - ice).clip(0, 1), np.nan))
    e["obstacle_frac"] = _nsum(e, ["wc_frac_tree", "wc_frac_built", "wc_frac_water"])
    e["obstacle_frac_ops"] = _nsum(e, ["wc_frac_tree", "wc_frac_built", "wc_frac_water", "wc_frac_wetland", "wc_frac_crop"])
    fields = {"context": ["context"], "lith_text": ["mstrat_lith"], "lith_desc": ["mstrat_text_all", "mstrat_descrip"],
              "source": ["source"], "stratum": ["stratum"]}
    for name, spec in cfg["text_flags"].items():
        if "earth" not in spec.get("bodies", []):
            continue
        cols = [c for f in spec["fields"] for c in fields.get(f, [])]
        e[name] = text_blob(e, cols).str.contains(spec["regex"], regex=True)
    e["flag_ice_surface"] = e["flag_ice_surface"] | (ice >= 0.7)
    e["has_geo_text"] = e["mstrat_text_all"].notna() | e["source"].isin(["named_site", "impact_structure", "volcano"])
    return e


# ----------------------------------------------------------------------------- spatial blocks (no leakage)
def spatial_blocks(lat, lon, body, polar=None):
    """Moon south pole: 100 km blocks in polar-stereographic x/y. Elsewhere: 5-degree lat/lon cells."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    body = np.asarray(body).astype(str)
    polar = (lat < -70) & (body != "earth") if polar is None else np.asarray(polar, bool)
    x, y = stereo_south(np.where(polar, lat, -80.0), lon)
    out = np.empty(len(lat), dtype=object)
    for i in range(len(lat)):
        if polar[i]:
            out[i] = f"MP_{int(np.floor(x[i] / 1e5))}_{int(np.floor(y[i] / 1e5))}"
        else:
            out[i] = f"{'E' if body[i] == 'earth' else 'MM'}_{int(np.floor(lat[i] / 5))}_{int(np.floor(lon[i] / 5))}"
    return out


def holdout_mask(blocks, frac=0.15, seed=0):
    """True for rows whose spatial block is held out (deterministic hash of the block id)."""
    return np.array([(zlib.crc32(f"{seed}:{b}".encode()) % 1000) < frac * 1000 for b in blocks])


# ----------------------------------------------------------------------------- terrain arrays
def load_terrain_arrays():
    """ids, body ('moon' | 'zone' | 'earth'), X float16 [N, 2, 128, 128] (elev minus mean in m, slope in deg),
    valid_frac per patch. 128 x 128 arrays come from tools/reassemble_arrays.py / tools/build_zone_patches.py."""
    ids, body, xs, vf = [], [], [], []
    for fname, b in ARRAY_FILES:
        p = PATCH / fname
        if not p.exists():
            log(f"WARNING missing {fname}")
            continue
        z = np.load(p, allow_pickle=False)
        n = len(z["patch_id"])
        ids += z["patch_id"].astype(str).tolist()
        body += [b] * n
        xs.append(np.stack([z["elev"].astype(np.float16), z["slope"].astype(np.float16)], axis=1))
        vf.append(z["valid"].reshape(n, -1).mean(1) if "valid" in z.files else np.ones(n))
    return np.array(ids), np.array(body), np.concatenate(xs), np.concatenate(vf)


# ----------------------------------------------------------------------------- anchors
def load_anchors() -> pd.DataFrame:
    a = pd.read_csv(CFG / "anchors.csv")
    if "moon_zone" not in a:
        a["moon_zone"] = "tile"
    a["moon_zone"] = a["moon_zone"].fillna("tile")
    a["entity_id"] = np.where(a["moon_zone"].isin(["landing", "ice"]), a["moon_target"] + "__" + a["moon_zone"],
                              a["moon_target"] + "__tile")
    a["cat"] = a["category"].map(CFG2OUT)
    a["w"] = a["confidence"].map({"high": 1.0, "med": 0.7, "low": 0.4}).fillna(0.7)
    return a


# ----------------------------------------------------------------------------- scoring + calibration maths
def subscore(m, e, tol, one_sided=False):
    """Similarity of one physical quantity: 1 at equal values, 0 at one tolerance (linear, as in the brief),
    then a SOFT floor towards -1 instead of a hard clip, so sites many tolerances away still keep their
    closest-approach order. m: (M, 1) Moon values, e: (1, E) Earth values. one_sided: only Earth BELOW Moon counts."""
    d = np.asarray(m, float) - np.asarray(e, float)
    x = (np.maximum(d, 0.0) if one_sided else np.abs(d)) / tol
    return np.where(x <= 1.0, 1.0 - x, -(1.0 - np.exp(-(x - 1.0))))


def impute_rowmean(T):
    """T: (n, k) term matrix. NaN terms take the mean of that row's available terms; all-NaN rows stay NaN."""
    T = np.array(T, float, copy=True)
    with np.errstate(all="ignore"):
        m = np.nanmean(T, axis=1)
    idx = np.where(np.isnan(T))
    T[idx] = m[idx[0]]
    return T


def fit_terms(T, y, w=None, prior=None, lam=1.0):
    """Non-negative weights + intercept: y ~ b + T @ c, c >= 0, ridge towards `prior`. Returns (c, b)."""
    from scipy.optimize import lsq_linear
    T, y = np.asarray(T, float), np.asarray(y, float)
    n, k = T.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    prior = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, float)
    sw = np.sqrt(w)[:, None]
    A = np.hstack([T, np.ones((n, 1))]) * sw
    b = y * sw[:, 0]
    A = np.vstack([A, np.sqrt(lam) * np.hstack([np.eye(k), np.zeros((k, 1))])])
    b = np.concatenate([b, np.sqrt(lam) * prior])
    res = lsq_linear(A, b, bounds=(np.r_[np.zeros(k), -np.inf], np.full(k + 1, np.inf)))
    return res.x[:k], float(res.x[k])


def spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    ok = a.notna() & b.notna()
    if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
