#!/usr/bin/env python
"""
Merge everything into two ML-ready tables + one manifest, and (optionally) a Colab upload bundle.

  data/ml/moon_tiles.csv   moon_patches + Diviner day/night (tmin) + cross-body x_* columns
  data/ml/earth_tiles.csv  earth candidates + terrain + environment + cross-body x_* columns
  data/ml/manifest.json    row counts, column -> category map (from config/categories.json when it lists
                           features), per-column coverage, array files and their keys
  --bundle PATH            zip data/ml + the two *_arrays.npz + configs + anchors + region zones for Colab

x_* columns are the SAME physical quantity on both bodies (directly comparable):
  x_temp_max_k, x_temp_mean_k, x_temp_min_k, x_temp_swing_k, x_grazing_sun_frac, x_noon_sun_max_deg,
  x_polar_night_frac, x_frozen_ground_frac, x_surface_ice_frac, plus the terrain columns (same names).

Optional inputs, merged when present (each is skipped silently when missing):
  data/patches/moon_midlat_patches.csv      extra Moon tiles 60S-60N (region='midlat'), appended to moon_tiles
                                            (union of columns, NaN where a part lacks a column)
  data/patches/earth_traverse_precise.csv   ~111 m traverse for the shortlisted Earth tiles: overrides the coarse
                                            trav_* values for those patch_ids (trav_method='precise_111m')
  data/patches/earth_patches_env_powerfill.csv  NASA POWER top-up: fills power_* only where still NaN, then the
                                            permafrost proxy / precip_best / x_frozen_ground_frac are recomputed
  data/patches/*_arrays_128.npz             128 x 128 terrain arrays: recorded in the manifest (shapes from headers)
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data" / "patches", ROOT / "data" / "ml"
TERRAIN = ["relief_m", "elev_std_m", "slope500_mean_deg", "slope500_p90_deg", "frac_slope500_gt10",
           "frac_slope500_gt20", "tri500_mean_m"]
MIDLAT_CSV = P / "moon_midlat_patches.csv"
TRAV_PRECISE_CSV = P / "earth_traverse_precise.csv"
POWERFILL_CSV = P / "earth_patches_env_powerfill.csv"
ARRAY_FILES = ("moon_patch_arrays.npz", "earth_patch_arrays.npz", "moon_patch_arrays_128.npz",
               "earth_patch_arrays_128.npz", "moon_midlat_arrays_128.npz")
MOON_OBLIQUITY, EARTH_OBLIQUITY = 1.54, 23.44
INPUTS = {}   # what was merged -> manifest["merge_inputs"]


def _read_optional(path, **kw):
    """Read an optional CSV that another process may still be writing; None when missing/unreadable."""
    if not path.exists():
        INPUTS[path.name] = {"present": False}
        return None
    try:
        df = pd.read_csv(path, low_memory=False, **kw)
    except Exception as ex:  # noqa: BLE001
        INPUTS[path.name] = {"present": True, "used": False, "error": str(ex)[:200]}
        return None
    INPUTS[path.name] = {"present": True, "used": True, "rows": int(len(df)),
                         "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}
    return df


def grazing_sun_frac(lat, obliquity, low=12.0):
    eps, phi = np.radians(obliquity), np.radians(np.asarray(lat, float))[:, None, None]
    t = np.linspace(0, 1, 120, endpoint=False)[None, :, None]
    h = np.linspace(-np.pi, np.pi, 96, endpoint=False)[None, None, :]
    dec = np.arcsin(np.sin(eps) * np.sin(2 * np.pi * t))
    el = np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)))
    day = el > 0
    return ((day & (el < low)).sum(axis=(1, 2)) / np.maximum(day.sum(axis=(1, 2)), 1)).astype(float)


def _put(df, col, val):
    """Set an x_* column, but keep values a part already provides (e.g. the midlat builder's own x_*)."""
    val = pd.Series(np.nan, index=df.index) if val is None else pd.Series(val, index=df.index)
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val) if col in df else val


def moon_x(m):
    """Cross-body x_* columns for one part of the Moon table (south pole or midlat)."""
    g = m.get
    lat = pd.to_numeric(m["center_lat"], errors="coerce")
    _put(m, "x_temp_max_k", g("dvn_tmax_max_k"))
    _put(m, "x_temp_mean_k", g("dvn_tavg_mean_k"))
    _put(m, "x_temp_min_k", g("dvn_tbol_min_k"))
    _put(m, "x_temp_swing_k", m["x_temp_max_k"] - m["x_temp_min_k"])
    _put(m, "x_grazing_sun_frac", grazing_sun_frac(lat, MOON_OBLIQUITY))
    # geometric fallbacks (latitude only) when a part has no noon/polar-night columns
    _put(m, "x_noon_sun_max_deg", g("noon_elev_max_deg"))
    _put(m, "x_noon_sun_max_deg", 90 - np.maximum(lat.abs() - MOON_OBLIQUITY, 0))
    _put(m, "x_polar_night_frac", g("frac_polar_night"))
    _put(m, "x_polar_night_frac", np.where(lat.abs() < 90 - MOON_OBLIQUITY, 0.0, np.nan))
    _put(m, "x_frozen_ground_frac", g("dvn_frac_ice_within_1m"))
    _put(m, "x_surface_ice_frac", g("dvn_frac_coldtrap_110k"))
    _put(m, "x_sunlit_frac_terrain", g("lola_avg_sun_vis_mean"))  # Moon only (terrain-aware)
    _put(m, "x_shadowed_frac", g("lola_frac_psr"))               # Moon only
    return m


def moon_table():
    m = pd.read_csv(P / "moon_patches.csv")
    if (P / "moon_patches_tmin.csv").exists():
        m = m.merge(pd.read_csv(P / "moon_patches_tmin.csv"), on="patch_id", how="left")
    m.insert(1, "body", "moon")
    m.insert(2, "region", "south_pole")
    m = moon_x(m)
    mid = _read_optional(MIDLAT_CSV)
    if mid is not None and len(mid):
        # duplicate headers (e.g. 'region' twice, read back as 'region.1') -> drop the copy when it is identical
        dups = [c for c in mid.columns if "." in c and c.rsplit(".", 1)[1].isdigit() and c.rsplit(".", 1)[0] in mid
                and mid[c].astype(str).equals(mid[c.rsplit(".", 1)[0]].astype(str))]
        mid = mid.drop(columns=dups + [c for c in ("body",) if c in mid])
        mid.insert(1, "body", "moon")
        if "region" in mid:
            mid["region"] = mid["region"].fillna("midlat").replace("", "midlat")
        else:
            mid.insert(2, "region", "midlat")
        dup = mid["patch_id"].isin(m["patch_id"]) | mid["patch_id"].duplicated()
        if dup.any():
            print(f"midlat: dropping {int(dup.sum())} rows whose patch_id is already in the table")
            mid = mid[~dup]
        if "is_target" not in mid:
            mid["is_target"] = np.nan
        tgt = set(pd.read_csv(ROOT / "config" / "targets.csv")["id"]) if (ROOT / "config/targets.csv").exists() else set()
        mid["is_target"] = mid["is_target"].fillna(mid["patch_id"].isin(tgt).astype(int))
        mid = moon_x(mid.copy())
        # the column means terrain-aware sunlight only; the midlat builder filled it with geometric daylight
        mid["x_sunlit_frac_terrain"] = np.nan
        INPUTS[MIDLAT_CSV.name].update(appended=int(len(mid)), columns_only_in_midlat=sorted(set(mid.columns) - set(m.columns)),
                                       polar_columns_missing_in_midlat=sorted(set(m.columns) - set(mid.columns)))
        m = pd.concat([m, mid], ignore_index=True, sort=False)   # union of columns, NaN where absent
    m["is_target"] = pd.to_numeric(m["is_target"], errors="coerce").fillna(0).astype(int)
    return m


def traverse_table():
    """Coarse traverse for all tiles, overridden by the precise (~111 m) run where it has a valid result."""
    tr = pd.read_csv(P / "earth_traverse.csv") if (P / "earth_traverse.csv").exists() else None
    if tr is not None:
        tr = tr[[c for c in tr.columns if c == "patch_id" or c.startswith("trav_")]].drop_duplicates("patch_id", keep="last")
    pr = _read_optional(TRAV_PRECISE_CSV)
    if pr is None or not len(pr):
        return tr
    ok = pr["patch_id"].notna()
    if "trav_source" in pr:
        ok &= pr["trav_source"].fillna("none").astype(str).ne("none")
    if "error" in pr:
        ok &= pr["error"].isna() | pr["error"].astype(str).str.strip().isin(["", "nan"])
    if "trav_flat_bare_frac" in pr:
        ok &= pr["trav_flat_bare_frac"].notna()
    pr = pr[ok]
    pr = pr[[c for c in pr.columns if c == "patch_id" or c.startswith("trav_")]].drop_duplicates("patch_id", keep="last").copy()
    pr["trav_method"] = "precise_111m"
    INPUTS[TRAV_PRECISE_CSV.name].update(valid_rows=int(len(pr)))
    if tr is None:
        return pr
    tr = tr.set_index("patch_id")
    pr = pr.set_index("patch_id")
    for c in pr.columns:
        if c not in tr:
            tr[c] = np.nan
    new = pr.index.difference(tr.index)
    if len(new):
        tr = pd.concat([tr, pr.loc[new]], sort=False)
    old = pr.index.intersection(tr.index)
    for c in pr.columns:        # replace whole rows (incl. NaN) so coarse and precise values are never mixed
        if pr[c].dtype == object:
            tr[c] = tr[c].astype(object)
        tr.loc[old, c] = pr.loc[old, c]
    INPUTS[TRAV_PRECISE_CSV.name].update(overrode=int(len(old)), added=int(len(new)))
    return tr.reset_index()


def power_fill(df):
    """Fill power_* from the POWER top-up only where still NaN; recompute the derived proxy columns."""
    pf = _read_optional(POWERFILL_CSV)
    if pf is None or "power_t2m_c" not in pf:
        return df
    cols = [c for c in pf.columns if c.startswith("power_")]
    pf = pf.drop_duplicates("patch_id", keep="last").set_index("patch_id")[cols]
    before = int(df["power_t2m_c"].notna().sum()) if "power_t2m_c" in df else 0
    df = df.set_index("patch_id")
    for c in cols:
        v = pd.to_numeric(pf[c], errors="coerce").reindex(df.index)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(v) if c in df else v
    df = df.copy().reset_index()
    t2m = df["power_t2m_c"].astype(float)
    cci = df["pf_frac_pct"].astype(float) if "pf_frac_pct" in df else pd.Series(np.nan, index=df.index)
    # same definitions as tools/build_earth_env.py
    df["permafrost_proxy"] = np.where(t2m.notna(), (t2m < -2.0).astype(float), np.nan)
    df["permafrost_flag"] = np.where(cci.notna(), (cci >= 50).astype(float), df["permafrost_proxy"])
    df["permafrost_src"] = np.where(cci.notna(), "esa_cci_pfr_2023", np.where(t2m.notna(), "proxy_power_t2m_lt_-2C", ""))
    if "precip_best_mm_yr" in df and "power_precip_mm_yr" in df:
        miss = df["precip_best_mm_yr"].isna() & df["power_precip_mm_yr"].notna()
        df.loc[miss, "precip_best_mm_yr"] = df.loc[miss, "power_precip_mm_yr"]
        if "precip_best_src" in df:
            df.loc[miss, "precip_best_src"] = "power_merra2"
        INPUTS[POWERFILL_CSV.name].update(precip_best_filled=int(miss.sum()))
    after = int(df["power_t2m_c"].notna().sum())
    INPUTS[POWERFILL_CSV.name].update(power_t2m_c_rows_before=before, power_t2m_c_rows_after=after, filled=after - before)
    return df


def earth_table():
    c = pd.read_csv(P / "earth_candidates.csv")
    t = pd.read_csv(P / "earth_patches_terrain.csv")
    e = pd.read_csv(P / "earth_patches_env.csv") if (P / "earth_patches_env.csv").exists() else pd.DataFrame({"patch_id": []})
    df = t.merge(c[["patch_id"] + [k for k in ("context",) if k in c]], on="patch_id", how="left")
    e = e.drop(columns=[k for k in e.columns if k in df.columns and k != "patch_id"])
    df = df.merge(e, on="patch_id", how="left")
    df = power_fill(df)
    tr = traverse_table()
    if tr is not None:   # landing -> ice traverse feature (Earth counterpart of Moon zones)
        df = df.merge(tr, on="patch_id", how="left")
    df = df.copy()
    df.insert(1, "body", "earth")
    df["x_temp_max_k"] = df.get("lst_day_max_k")
    df["x_temp_mean_k"] = df.get("lst_mean_k")
    df["x_temp_min_k"] = df.get("lst_night_min_k")
    df["x_temp_swing_k"] = df["x_temp_max_k"] - df["x_temp_min_k"]
    df["x_grazing_sun_frac"] = grazing_sun_frac(df["center_lat"], 23.44)
    df["x_noon_sun_max_deg"] = df["noon_elev_max_deg"]
    df["x_polar_night_frac"] = df["frac_polar_night"]
    proxy = (df["power_t2m_c"] < -2).astype(float).where(df["power_t2m_c"].notna()) if "power_t2m_c" in df else np.nan
    df["x_frozen_ground_frac"] = (df["pf_frac_pct"] / 100).fillna(proxy) if "pf_frac_pct" in df else proxy
    # WorldCover snow/ice; where WorldCover is NaN (south of 60S) fall back to the traverse ice share
    # (precise ~111 m where available, else coarse), e.g. Concordia
    wc = pd.to_numeric(df["wc_frac_snow_ice"], errors="coerce") if "wc_frac_snow_ice" in df else pd.Series(np.nan, index=df.index)
    ti = pd.to_numeric(df["trav_ice_frac"], errors="coerce") if "trav_ice_frac" in df else pd.Series(np.nan, index=df.index)
    df["x_surface_ice_frac"] = wc.fillna(ti)
    df["x_surface_ice_src"] = np.where(wc.notna(), "worldcover", np.where(ti.notna(), "traverse_ice_frac", ""))
    return df


def coverage(df):
    return {c: round(float(df[c].notna().mean()), 3) for c in df.columns}


def category_map():
    """Column -> category from config/categories.json if it names features; else a built-in default."""
    default = {
        "terrain": TERRAIN,
        "thermal_illumination": ["x_temp_max_k", "x_temp_mean_k", "x_temp_min_k", "x_temp_swing_k", "x_grazing_sun_frac",
                                 "x_noon_sun_max_deg", "x_polar_night_frac", "x_sunlit_frac_terrain"],
        "water_ice": ["x_frozen_ground_frac", "x_surface_ice_frac", "x_shadowed_frac", "dvn_ice_depth_median_m",
                      "pf_magt_2m_c", "soil_moist_mean", "power_frost_days"],
        "geology": ["moon_geo_unit", "moon_geo_unit_name", "geo_frac_crater", "geo_frac_basin", "geo_frac_terra",
                    "geo_frac_plains", "mstrat_lith", "mstrat_descrip", "mstrat_text_all", "sg_sand_pct", "sg_silt_pct",
                    "sg_clay_pct", "sg_bdod_g_cm3", "sg_cfvo_pct", "context"],
        "dust_regolith": ["aot_mean", "aot_max", "sg_silt_pct", "sg_clay_pct", "wc_frac_bare_sparse"],
        "biology_sterility": ["ndvi_mean", "ndvi_max", "precip_best_mm_yr", "soil_moist_mean", "power_uv_index",
                              "wc_frac_tree", "wc_frac_grass", "wc_frac_crop", "wc_frac_moss_lichen"],
        "ops_landing_mobility": ["slope500_p90_deg", "frac_slope500_gt10", "frac_slope500_gt20", "x_sunlit_frac_terrain",
                                 "x_shadowed_frac", "x_temp_min_k", "x_temp_swing_k"],
    }
    try:
        cats = json.loads((ROOT / "config" / "categories.json").read_text())
        cats = cats.get("categories", cats)
        found = {}
        for name, spec in cats.items():
            cols = []
            for key in ("features", "columns", "moon_features", "earth_features", "inputs"):
                v = spec.get(key) if isinstance(spec, dict) else None
                if isinstance(v, list):
                    cols += [x for x in v if isinstance(x, str)]
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, list):
                            cols += [x for x in vv if isinstance(x, str)]
            if cols:
                found[name] = sorted(set(cols))
        return {"from_config": found, "default": default}
    except Exception as ex:  # noqa: BLE001
        return {"from_config": {}, "default": default, "note": str(ex)}


def npz_shapes(path):
    """{key: shape} and {key: dtype} from the .npy headers only (no decompression of the arrays)."""
    shapes, dtypes = {}, {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with zf.open(info) as f:
                ver = np.lib.format.read_magic(f)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dt = rd(f)
            shapes[info.filename[:-4]], dtypes[info.filename[:-4]] = list(shape), str(dt)
    return shapes, dtypes


def coverage_summary(m, e):
    """Coverage of the columns touched by the optional merges (midlat, POWER top-up, precise traverse)."""
    pw = [c for c in e.columns if c.startswith("power_")]
    tv = [c for c in e.columns if c.startswith("trav_")]
    xs = [c for c in m.columns if c.startswith("x_")]
    s = dict(
        moon_rows_by_region={str(k): int(v) for k, v in m["region"].value_counts().items()},
        moon_targets_by_region={str(k): int(v) for k, v in m.groupby("region")["is_target"].sum().items()},
        moon_target_ids=sorted(m.loc[m["is_target"] == 1, "patch_id"].astype(str)),
        moon_x_coverage_by_region={str(r): coverage(g[xs + [c for c in TERRAIN if c in g]]) for r, g in m.groupby("region")},
        earth_power_coverage={c: round(float(e[c].notna().mean()), 3) for c in pw},
        earth_power_t2m_rows=int(e["power_t2m_c"].notna().sum()) if "power_t2m_c" in e else 0,
        earth_trav_coverage={c: round(float(e[c].notna().mean()), 3) for c in tv},
        earth_trav_method_counts={str(k): int(v) for k, v in e["trav_method"].value_counts(dropna=False).items()} if "trav_method" in e else {},
        earth_trav_precise_rows=int((e.get("trav_method") == "precise_111m").sum()) if "trav_method" in e else 0,
        earth_surface_ice=dict(coverage=round(float(e["x_surface_ice_frac"].notna().mean()), 3),
                               src_counts={str(k): int(v) for k, v in e["x_surface_ice_src"].value_counts().items()}
                               if "x_surface_ice_src" in e else {}),
        earth_frozen_ground=dict(coverage=round(float(e["x_frozen_ground_frac"].notna().mean()), 3),
                                 src_counts={str(k): int(v) for k, v in e["permafrost_src"].fillna("").value_counts().items()}
                                 if "permafrost_src" in e else {}),
    )
    if "precip_best_mm_yr" in e:
        s["earth_precip_best_coverage"] = round(float(e["precip_best_mm_yr"].notna().mean()), 3)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="", help="path of a zip to write for Colab upload")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m, e = moon_table(), earth_table()
    m.to_csv(OUT / "moon_tiles.csv", index=False)
    e.to_csv(OUT / "earth_tiles.csv", index=False)
    shared = sorted(set(m.columns) & set(e.columns) - {"patch_id", "body", "center_lat", "center_lon", "region", "stratum"})
    arrays, arrays_meta = {}, {}
    for f in ARRAY_FILES:
        if (P / f).exists():
            st = (P / f).stat()
            meta = dict(path=f"data/patches/{f}", size_mb=round(st.st_size / 1e6, 1),
                        mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)))
            try:
                arrays[f], meta["dtypes"] = npz_shapes(P / f)
            except Exception as ex:  # noqa: BLE001  e.g. still being written
                meta["error"] = f"unreadable (still being written?): {str(ex)[:120]}"
            arrays_meta[f] = meta
    manifest = dict(
        moon_rows=len(m), earth_rows=len(e), moon_targets=int(m["is_target"].sum()),
        moon_rows_by_region={str(k): int(v) for k, v in m["region"].value_counts().items()},
        shared_columns=shared, categories=category_map(), arrays=arrays, arrays_meta=arrays_meta,
        merge_inputs=INPUTS, coverage_summary=coverage_summary(m, e),
        coverage=dict(moon=coverage(m), earth=coverage(e)),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        notes=["x_* columns are the same physical quantity on both bodies",
               "moon_tiles.region: 'south_pole' (76-90S polar-stereographic grid) or 'midlat' (60S-60N extra tiles); "
               "columns absent for a region are NaN (mask, do not drop)",
               "Earth trav_method: 'precise_111m' overrides the coarse 'fast_local' values for the shortlisted tiles",
               "moon_tiles x_sunlit_frac_terrain is NaN for region='midlat' (no terrain-aware illumination product there)",
               "Earth x_surface_ice_frac = WorldCover snow/ice, else trav_ice_frac where WorldCover is NaN (x_surface_ice_src)",
               "Earth power_* topped up from earth_patches_env_powerfill.csv where NaN; permafrost_proxy / "
               "x_frozen_ground_frac recomputed (CCI permafrost where present, else POWER T2M < -2 C)",
               "Earth climate grids are coarser than the 10 km tiles (POWER 0.5 deg, LST 0.05-0.2 deg)",
               "WorldCover / SoilGrids / GSMaP have no data south of 60S -> NaN (mask, do not drop)",
               "Artemis targets are crater-centre anchors for 5 of 9 regions: see data/patches/artemis_region_zones.csv"],
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    cs = manifest["coverage_summary"]
    print(f"moon_tiles.csv: {m.shape}   earth_tiles.csv: {e.shape}   shared columns: {len(shared)}")
    print(f"moon by region: {cs['moon_rows_by_region']}   targets: {manifest['moon_targets']}")
    print(f"earth power_t2m_c rows: {cs['earth_power_t2m_rows']}/{len(e)}   trav methods: {cs['earth_trav_method_counts']}")
    print(f"x_frozen_ground_frac coverage: {cs['earth_frozen_ground']['coverage']}   arrays: "
          + ", ".join(f"{k}{'' if k in arrays else ' (unreadable)'}" for k in arrays_meta))
    for k, v in INPUTS.items():
        print(f"  input {k}: " + json.dumps({a: b for a, b in v.items() if not isinstance(b, list)}))
    if args.bundle:
        files = [OUT / "moon_tiles.csv", OUT / "earth_tiles.csv", OUT / "manifest.json", OUT / "moon_zones.csv",
                 ROOT / "docs/ML_PLAN.md", ROOT / "docs/ARTEMIS_ZONES.md", ROOT / "docs/DATA_PROFILE.md",
                 P / "moon_patch_arrays.npz", P / "earth_patch_arrays.npz", P / "artemis_region_zones.csv",
                 ROOT / "data/catalogs/moon_geologic_units.csv", ROOT / "config/anchors.csv",
                 ROOT / "config/categories.json", ROOT / "config/experiments.json", ROOT / "config/targets.csv",
                 ROOT / "config/earth_sites.csv"]
        nb = ROOT / "notebooks" / "ssl_colab.ipynb"
        if nb.exists():
            files.append(nb)
        with zipfile.ZipFile(args.bundle, "w", compression=zipfile.ZIP_STORED) as zf:   # npz already compressed
            for f in files:
                if f.exists():
                    zf.write(f, arcname=str(f.relative_to(ROOT)).replace("\\", "/"))
        print(f"bundle -> {args.bundle} ({Path(args.bundle).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
