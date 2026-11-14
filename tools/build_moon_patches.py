#!/usr/bin/env python
"""
Build the MOON side of the patch dataset: 10 km x 10 km patches tiling the lunar south polar
region (LOLA LDEM_80S extent, lat <= ~-80) + one patch per polar Moon target in config/targets.csv.

Outputs
  data/patches/moon_patches.csv         one row per patch (terrain, Diviner, LOLA illumination,
                                        geometric illumination, USGS unified geology)
  data/patches/moon_patch_arrays.npz    per patch: 64x64 elev (minus mean), slope, sun visibility;
                                        16x16 Diviner tmax / tavg (NaN where no cells)
  data/catalogs/moon_geologic_units.csv USGS Unified Geologic Map of the Moon unit table (DMU text)

Inputs (in RAW = ~/.cache/earth_analog_raw, see README_moon.md)
  LDEM_80S_80MPP_ADJ.TIF, AVGVISIB_75S_120M_201608.TIF, dlre_prp_south.tab (-> .npz),
  ugm_raster.zip (USGS unified geology raster 64 ppd) + ugm/ DOMU csv from the GIS v2 zip

Usage
  python tools/build_moon_patches.py [--refresh] [--workers 8]
"""
import argparse
import io
import math
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract as base  # noqa: E402  terrain_metrics, illumination, _slope_deg, BODY

RAW = Path(os.environ.get("ANALOG_RAW_DIR", os.path.expanduser("~/.cache/earth_analog_raw")))
OUT_DIR = ROOT / "data" / "patches"
CAT_DIR = ROOT / "data" / "catalogs"
DEM_PATH = RAW / "LDEM_80S_80MPP_ADJ.TIF"
VIS_PATH = RAW / "AVGVISIB_75S_120M_201608.TIF"
PRP_TAB = RAW / "dlre_prp_south.tab"
PRP_NPZ = RAW / "dlre_prp_south.npz"
UGM_ZIP = RAW / "ugm_raster.zip"
UGM_DIR = RAW / "ugm"
UGM_TIF_IN_ZIP = ("Unified_Geologic_Map_of_the_Moon_RASTER/numerical_versions/"
                  "Unified_Geologic_Map_of_the_Moon_64ppd.tif")
UGM_DN_IN_ZIP = "Unified_Geologic_Map_of_the_Moon_RASTER/Dn_GeologicUnit_RGBcolor.csv"
UGM_GIS_URL = "https://asc-astropedia.s3.us-west-2.amazonaws.com/Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2.zip"
UGM_DOMU_IN_ZIP = ("Unified_Geologic_Map_of_the_Moon_GIS/Lunar_GIS/Symbol_LayerDefinitions/"
                   "Unified_Geologic_Map_of_the_Moon_DOMU_descriptions.csv")

R_MOON = base.BODY["moon"]["R"]
PATCH_M = 10_000.0
HALF_EXTENT = 300_000.0          # grid covers [-300, 300] km in x and y (DEM is +-304 km)
NGRID = int(2 * HALF_EXTENT / PATCH_M)   # 60 x 60 candidate patches
DEM_RES = 80.0
DEM_N = int(PATCH_M / DEM_RES)   # 125 px per patch
NA = 64                          # elev / slope / sunvis grid (156.25 m/px)
ND = 16                          # Diviner grid (625 m/px)
NG = 32                          # geology sampling grid (312.5 m)
MIN_VALID = 0.8
GEO_CRS = f"+proj=longlat +R={R_MOON} +no_defs"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================= workers
_W = {}


def _init_worker(dem_path, vis_path):
    os.environ.setdefault("GDAL_CACHEMAX", "256")
    import rasterio
    _W["dem"] = rasterio.open(dem_path)
    _W["vis"] = rasterio.open(vis_path)


def _read_clip(src, r0, c0, h, w, fill):
    """Window read with out-of-raster parts filled (cheaper than rasterio boundless)."""
    from rasterio.windows import Window
    out = np.full((h, w), fill, dtype=np.float64)
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r0 + h, src.height), min(c0 + w, src.width)
    if rr1 > rr0 and cc1 > cc0:
        a = src.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)).astype(np.float64)
        out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a
    return out


def _bin_mean(vals, brow, bcol, n):
    ok = np.isfinite(vals)
    idx = (brow * n + bcol)[ok]
    s = np.bincount(idx, vals[ok], minlength=n * n)
    c = np.bincount(idx, minlength=n * n)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    m[c == 0] = np.nan
    return m.reshape(n, n)


def _process(task):
    """task = (patch_id, x_left, y_top, center_lat, is_target). Returns (rec|None, elev, slope, vis)."""
    pid, xl, yt, clat, is_target = task
    dem, vis = _W["dem"], _W["vis"]
    # --- DEM window (pixel aligned; targets rounded to the nearest pixel edge, <= 40 m shift)
    c0 = int(round((xl - dem.bounds.left) / DEM_RES))
    r0 = int(round((dem.bounds.top - yt) / DEM_RES))
    z = _read_clip(dem, r0, c0, DEM_N, DEM_N, np.nan)
    if dem.nodata is not None and np.isfinite(dem.nodata):
        z[z == dem.nodata] = np.nan
    vf = float(np.isfinite(z).mean())
    if vf < MIN_VALID and not is_target:
        return None
    # ground spacing: polar stereographic true at pole, k = 1/cos^2(colat/2)
    colat = math.radians(90.0 + clat)
    gscale = math.cos(colat / 2) ** 2
    dx = DEM_RES * gscale
    rec = dict(patch_id=pid)
    try:
        rec.update(base.terrain_metrics(z, dx, dx))
    except Exception as e:  # noqa: BLE001
        rec["terrain_error"] = str(e)[:80]
    # 64x64 elevation by bin-averaging 80 m pixel centres into 156.25 m bins
    k = (np.arange(DEM_N) + 0.5) * DEM_RES
    b = np.minimum((k / (PATCH_M / NA)).astype(int), NA - 1)
    BR, BC = np.meshgrid(b, b, indexing="ij")
    e64 = _bin_mean(z.ravel(), BR.ravel(), BC.ravel(), NA)
    slope64 = base._slope_deg(e64, PATCH_M / NA * gscale, PATCH_M / NA * gscale)
    with np.errstate(all="ignore"):
        e64 = e64 - np.nanmean(e64)
    # --- LOLA average sun visibility: all 120 m pixels whose centre lies inside the patch
    vres = vis.transform.a
    vc0 = int(math.floor((xl - vis.bounds.left) / vres))
    vr0 = int(math.floor((vis.bounds.top - yt) / vres))
    nv = int(math.ceil(PATCH_M / vres)) + 2
    v = _read_clip(vis, vr0, vc0, nv, nv, vis.nodata if vis.nodata is not None else -32768)
    if vis.nodata is not None:
        v[v == vis.nodata] = np.nan
    v = v * ((vis.scales or (1.0,))[0] or 1.0) + ((vis.offsets or (0.0,))[0] or 0.0)
    px = vis.bounds.left + (vc0 + np.arange(nv) + 0.5) * vres - xl          # metres from left edge
    py = yt - (vis.bounds.top - (vr0 + np.arange(nv) + 0.5) * vres)          # metres down from top
    PY, PX = np.meshgrid(py, px, indexing="ij")
    inside = (PX >= 0) & (PX < PATCH_M) & (PY >= 0) & (PY < PATCH_M)
    vin = v[inside]
    vin_f = vin[np.isfinite(vin)]
    if vin_f.size:
        rec.update(lola_avg_sun_vis_mean=float(vin_f.mean()),
                   lola_avg_sun_vis_p90=float(np.percentile(vin_f, 90)),
                   lola_avg_sun_vis_max=float(vin_f.max()),
                   lola_frac_psr=float((vin_f <= 0).mean()),
                   lola_frac_sunlit_gt50pct=float((vin_f > 0.5).mean()))
    bw = PATCH_M / NA
    vis64 = _bin_mean(vin, (PY[inside] / bw).astype(int), (PX[inside] / bw).astype(int), NA)
    rec.update(base.illumination(clat, "moon"))
    return rec, e64.astype(np.float32), slope64.astype(np.float32), vis64.astype(np.float32)


# ============================================================================= Diviner PRP
def ensure_prp_npz(refresh=False):
    if PRP_NPZ.exists() and not refresh:
        return
    if not PRP_TAB.exists():
        raise SystemExit(f"missing {PRP_TAB} (download {PRP_TAB.name} first)")
    log("converting Diviner PRP table -> npz ...")
    cols = ["tri_clon", "tri_clat", "temp_avg", "temp_max", "ice_depth"]
    parts = []
    for ch in pd.read_csv(PRP_TAB, skipinitialspace=True, usecols=cols, chunksize=500_000,
                          dtype="float32", engine="c"):
        parts.append(ch[cols].to_numpy(dtype=np.float32))
    a = np.vstack(parts)
    np.savez_compressed(PRP_NPZ, lon=a[:, 0], lat=a[:, 1], tavg=a[:, 2], tmax=a[:, 3], ice=a[:, 4])
    log(f"  {a.shape[0]:,} triangles -> {PRP_NPZ}")


def project(lon, lat, dem_crs):
    from rasterio.warp import transform
    x, y = transform(GEO_CRS, dem_crs, np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def unproject(x, y, dem_crs):
    from rasterio.warp import transform
    lon, lat = transform(dem_crs, GEO_CRS, np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def diviner_features(prp, px, py, patch_xl, patch_yt, n_rows):
    """Bin PRP triangle centres into patches (vectorised). patch_xl/yt: left/top edge per row index.
    Grid patches are found via integer grid arithmetic; overlapping target patches via masks."""
    tavg = prp["tavg"].astype(np.float64)
    tmax = prp["tmax"].astype(np.float64)
    ice = prp["ice"].astype(np.float64)
    tavg[tavg <= 0] = np.nan
    tmax[tmax <= 0] = np.nan
    # ice_depth codes: -999 = not stable within 2.87 m; -1 = coldest cells (tmax <= ~101 K), ice
    # stable AT the surface (verified empirically: all -1 cells have tmax 27-101 K); 0 = surface.
    ice[ice == -1] = 0.0
    feats = []
    g_tmax = np.full((n_rows, ND, ND), np.nan, np.float32)
    g_tavg = np.full((n_rows, ND, ND), np.nan, np.float32)

    def assign(rowidx, sub_r, sub_c, sel):
        r = rowidx[sel]
        df = pd.DataFrame(dict(row=r, tavg=tavg[sel], tmax=tmax[sel], ice=ice[sel]))
        df["stable"] = df["ice"] >= 0
        df["w1m"] = df["stable"] & (df["ice"] <= 1.0)
        df["surf"] = df["stable"] & (df["ice"] <= 0.0)
        df["cold"] = np.where(np.isfinite(df["tmax"]), df["tmax"] < 110, np.nan)
        g = df.groupby("row")
        f = pd.DataFrame(dict(
            dvn_tavg_mean_k=g["tavg"].mean(), dvn_tavg_min_k=g["tavg"].min(),
            dvn_tmax_mean_k=g["tmax"].mean(), dvn_tmax_max_k=g["tmax"].max(),
            dvn_tmax_min_k=g["tmax"].min(), dvn_frac_coldtrap_110k=g["cold"].mean(),
            dvn_frac_ice_stable=g["stable"].mean(), dvn_frac_ice_within_1m=g["w1m"].mean(),
            dvn_frac_ice_surface=g["surf"].mean(),
            dvn_n_cells=g.size()))
        f["dvn_ice_depth_median_m"] = df[df["stable"]].groupby("row")["ice"].median()
        feats.append(f)
        sr, sc = sub_r[sel], sub_c[sel]
        for arr, src in ((g_tmax, tmax[sel]), (g_tavg, tavg[sel])):
            ok = np.isfinite(src)
            flat = (r[ok] * ND + sr[ok]) * ND + sc[ok]
            s = np.bincount(flat, src[ok], minlength=n_rows * ND * ND)
            c = np.bincount(flat, minlength=n_rows * ND * ND)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = (s / c).reshape(n_rows, ND, ND)
            has = c.reshape(n_rows, ND, ND) > 0
            arr[has] = m[has]
    return assign, feats, g_tmax, g_tavg


# ============================================================================= geology
GEO_CLASSES = {  # keyword on DN-table Unit_Name / unit code -> class column
    "crater": lambda code, name: "Crater" in name and "Basin" not in name and "Formation" not in name,
    "basin": lambda code, name: "Basin" in name or "Formation" in name,
    "plains": lambda code, name: "Plains" in name,
    "terra": lambda code, name: "Terra" in name and "Plains" not in name,
    "mare": lambda code, name: "Mare" in name or "Dark Mantling" in name or "Plateau" in name,
}
GEO_AGES = {"copernican": "Copernican", "eratosthenian": "Eratosthenian", "imbrian": "Imbrian",
            "nectarian": "Nectarian", "prenectarian": "Pre-Nectarian"}


def _main_age(a):
    a = str(a)
    return "Pre-Nectarian" if a.startswith("Pre") else a.split("-")[0]


def ensure_geology(refresh=False):
    """Extract the 64 ppd unit raster + DN table, fetch DOMU descriptions, write unit catalog."""
    UGM_DIR.mkdir(parents=True, exist_ok=True)
    tif = UGM_DIR / "Unified_Geologic_Map_of_the_Moon_64ppd.tif"
    dncsv = UGM_DIR / "Dn_GeologicUnit_RGBcolor.csv"
    if not tif.exists() or not dncsv.exists():
        with zipfile.ZipFile(UGM_ZIP) as z:
            tif.write_bytes(z.read(UGM_TIF_IN_ZIP))
            dncsv.write_bytes(z.read(UGM_DN_IN_ZIP))
    domu = UGM_DIR / "Unified_Geologic_Map_of_the_Moon_DOMU_descriptions.csv"
    if not domu.exists():
        try:  # range-read only the 10 KB DOMU csv out of the 224 MB GIS zip
            domu.write_bytes(_http_zip_member(UGM_GIS_URL, UGM_DOMU_IN_ZIP))
        except Exception as e:  # noqa: BLE001
            log("  WARN: could not fetch DOMU descriptions:", e)
    dn = pd.read_csv(dncsv)
    dn["Unit"] = dn["Unit"].str.strip()
    cat_path = CAT_DIR / "moon_geologic_units.csv"
    if not cat_path.exists() or refresh:
        cat = dn.rename(columns={"DN": "dn", "Unit": "unit", "Unit_Name": "unit_class_name",
                                 "Unit_Age": "age", "GeologyUnit_color_hex": "color_hex",
                                 "Polygon_Count": "polygon_count", "Geodesic_AREA_m2": "area_m2"})
        cat = cat[["dn", "unit", "unit_class_name", "age", "color_hex", "polygon_count", "area_m2"]]
        if domu.exists():
            d = pd.read_csv(domu, encoding="utf-8-sig")
            d.columns = [c.strip().lower() for c in d.columns]
            d = d.rename(columns={"unit": "unit", "name": "name", "description": "description",
                                  "interpretation": "interpretation"})
            d["unit"] = d["unit"].astype(str).str.strip().replace({"Ios": "Iohs"})  # DMU vs raster code
            cat = cat.merge(d, on="unit", how="outer")
        else:
            cat["name"] = cat["unit_class_name"]
            cat["description"] = ""
            cat["interpretation"] = ""
        cat["name"] = cat["name"].fillna(cat["unit_class_name"])
        for c in ("description", "interpretation"):
            cat[c] = cat[c].astype(str).str.strip().replace({"nan": "", "n/a": ""})
        cat = cat.sort_values(["dn", "unit"], na_position="last")
        CAT_DIR.mkdir(parents=True, exist_ok=True)
        cat.to_csv(cat_path, index=False)
        log(f"  geology catalog: {len(cat)} units -> {cat_path}")
    cat = pd.read_csv(cat_path)
    return tif, dn, cat


def _http_zip_member(url, member):
    import requests

    class HF(io.RawIOBase):
        def __init__(s):
            s.ses = requests.Session()
            s.size = int(s.ses.head(url, timeout=60).headers["Content-Length"])
            s.p = 0

        def seekable(s):
            return True

        def readable(s):
            return True

        def seek(s, o, w=0):
            s.p = o if w == 0 else (s.p + o if w == 1 else s.size + o)
            return s.p

        def tell(s):
            return s.p

        def readinto(s, buf):
            n = min(len(buf), s.size - s.p)
            if n <= 0:
                return 0
            d = s.ses.get(url, headers={"Range": f"bytes={s.p}-{s.p + n - 1}"}, timeout=120).content
            buf[:len(d)] = d
            s.p += len(d)
            return len(d)

    with zipfile.ZipFile(io.BufferedReader(HF(), 1 << 20)) as z:
        return z.read(member)


def geology_features(tif, dn, cat, xl, yt, dem_crs):
    import rasterio
    from rasterio.windows import Window
    n = len(xl)
    with rasterio.open(tif) as s:
        ppd = 1.0 / s.transform.a
        row0 = int((90 - (-74.0)) * ppd)
        g = s.read(1, window=Window(0, row0, s.width, s.height - row0)).astype(np.int16)
        nod = s.nodata
        left, top = s.bounds.left, s.bounds.top
    off = (np.arange(NG) + 0.5) * PATCH_M / NG
    OY, OX = np.meshgrid(off, off, indexing="ij")
    X = (xl[:, None] + OX.ravel()[None, :]).ravel()
    Y = (yt[:, None] - OY.ravel()[None, :]).ravel()
    lon, lat = unproject(X, Y, dem_crs)
    lon = ((lon + 180) % 360) - 180
    cc = np.clip(((lon - left) * ppd).astype(int), 0, g.shape[1] - 1)
    rr = np.clip(((top - lat) * ppd).astype(int) - row0, 0, g.shape[0] - 1)
    code = g[rr, cc].reshape(n, NG * NG)
    if nod is not None:
        code[code == nod] = 0
    code[code < 0] = 0
    ncode = int(max(dn["DN"].max(), code.max())) + 1
    counts = np.zeros((n, ncode), np.int64)
    pidx = np.repeat(np.arange(n), NG * NG)
    np.add.at(counts, (pidx, code.ravel()), 1)
    valid = counts[:, 1:].sum(1)
    maj = counts[:, 1:].argmax(1) + 1
    dn_unit = dict(zip(dn["DN"], dn["Unit"]))
    dn_name = dict(zip(dn["DN"], dn["Unit_Name"]))
    dn_age = dict(zip(dn["DN"], dn["Unit_Age"]))
    unit_name = dict(zip(cat["unit"], cat["name"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(dict(
            moon_geo_unit=[dn_unit.get(m, "") if v else "" for m, v in zip(maj, valid)],
            moon_geo_unit_frac=np.where(valid > 0, counts[np.arange(n), maj] / valid, np.nan),
        ))
        out["moon_geo_unit_name"] = out["moon_geo_unit"].map(unit_name).fillna("")
        out["moon_geo_unit_age"] = [dn_age.get(m, "") if v else "" for m, v in zip(maj, valid)]
        out["moon_geo_n_units"] = (counts[:, 1:] > 0).sum(1)
        for cls, fn in GEO_CLASSES.items():
            cols = [d for d in dn["DN"] if fn(dn_unit[d], dn_name[d])]
            out[f"geo_frac_{cls}"] = np.where(valid > 0, counts[:, cols].sum(1) / valid, np.nan)
        for key, age in GEO_AGES.items():
            cols = [d for d in dn["DN"] if _main_age(dn_age[d]) == age]
            out[f"geo_frac_age_{key}"] = np.where(valid > 0, counts[:, cols].sum(1) / valid, np.nan)
        out["geo_valid_frac"] = valid / (NG * NG)
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="rebuild everything")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-lat", type=float, default=-80.0,
                    help="keep grid patches whose centre lat <= this (DEM corners reach ~-76.3)")
    args = ap.parse_args()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = OUT_DIR / "moon_patches.csv", OUT_DIR / "moon_patch_arrays.npz"

    for p in (DEM_PATH, VIS_PATH):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    ensure_prp_npz(args.refresh)
    tif, dn, cat = ensure_geology(args.refresh)
    if csv_path.exists() and npz_path.exists() and not args.refresh:
        log(f"outputs exist ({csv_path.name}, {npz_path.name}); use --refresh to rebuild")
        summarize(pd.read_csv(csv_path))
        return 0

    import rasterio
    with rasterio.open(DEM_PATH) as s:
        dem_crs = s.crs

    # ---------------------------------------------------------------- tasks
    ix, iy = np.meshgrid(np.arange(NGRID), np.arange(NGRID))
    ix, iy = ix.ravel(), iy.ravel()
    gx = -HALF_EXTENT + (ix + 0.5) * PATCH_M
    gy = HALF_EXTENT - (iy + 0.5) * PATCH_M
    glon, glat = unproject(gx, gy, dem_crs)
    tasks = []
    for k in range(len(ix)):
        if glat[k] > args.max_lat:
            continue
        pid = f"mp_{int(gx[k] / 1000):+04d}_{int(gy[k] / 1000):+04d}"
        tasks.append((pid, gx[k] - PATCH_M / 2, gy[k] + PATCH_M / 2, float(glat[k]), 0))
    tg = pd.read_csv(ROOT / "config" / "targets.csv")
    tg = tg[(tg["body"] == "moon") & (tg["lat"] <= -80)].reset_index(drop=True)
    tx, ty = project(tg["lon"].to_numpy(float), tg["lat"].to_numpy(float), dem_crs)
    tx, ty = np.atleast_1d(tx), np.atleast_1d(ty)
    for k, r in tg.iterrows():
        tasks.append((r["id"], tx[k] - PATCH_M / 2, ty[k] + PATCH_M / 2, float(r["lat"]), 1))
    meta = {t[0]: t for t in tasks}

    log(f"terrain + illumination for {len(tasks)} candidate patches on {args.workers} workers ...")
    recs, E, S, V = [], [], [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(str(DEM_PATH), str(VIS_PATH))) as pool:
        for i, res in enumerate(pool.imap(_process, tasks, chunksize=20)):
            if res is None:
                continue
            rec, e, s_, v = res
            recs.append(rec)
            E.append(e)
            S.append(s_)
            V.append(v)
    log(f"  kept {len(recs)} patches ({sum(1 for r in recs if r['patch_id'] in set(tg['id']))} targets) "
        f"in {time.time() - t0:.0f}s")
    df = pd.DataFrame(recs)
    is_t = df["patch_id"].isin(set(tg["id"])).to_numpy()
    df.insert(1, "is_target", is_t.astype(int))
    xl = np.array([meta[p][1] for p in df["patch_id"]])
    yt = np.array([meta[p][2] for p in df["patch_id"]])
    df.insert(1, "x_m", xl + PATCH_M / 2)
    df.insert(2, "y_m", yt - PATCH_M / 2)
    clon, clat = unproject(df["x_m"].to_numpy(), df["y_m"].to_numpy(), dem_crs)
    tmap = tg.set_index("id")
    clat = np.where(is_t, df["patch_id"].map(tmap["lat"]).fillna(0).to_numpy(), clat)
    clon = np.where(is_t, df["patch_id"].map(tmap["lon"]).fillna(0).to_numpy(), ((clon + 180) % 360) - 180)
    df.insert(1, "center_lat", np.round(clat, 5))
    df.insert(2, "center_lon", np.round(clon, 5))
    n = len(df)

    # ---------------------------------------------------------------- Diviner
    log("Diviner PRP binning ...")
    z = np.load(PRP_NPZ)
    prp = {k: z[k] for k in z.files}
    px, py = project(prp["lon"].astype(np.float64), prp["lat"].astype(np.float64), dem_crs)
    assign, feats, g_tmax, g_tavg = diviner_features(prp, px, py, xl, yt, n)
    # grid patches: integer arithmetic on the 10 km grid
    lut = np.full(NGRID * NGRID, -1, np.int64)
    for r_i, pid in enumerate(df["patch_id"]):
        if not is_t[r_i]:
            gx_i = int(round((xl[r_i] + HALF_EXTENT) / PATCH_M))
            gy_i = int(round((HALF_EXTENT - yt[r_i]) / PATCH_M))
            lut[gy_i * NGRID + gx_i] = r_i
    cix = np.floor((px + HALF_EXTENT) / PATCH_M).astype(np.int64)
    ciy = np.floor((HALF_EXTENT - py) / PATCH_M).astype(np.int64)
    ing = (cix >= 0) & (cix < NGRID) & (ciy >= 0) & (ciy < NGRID)
    rowidx = np.full(px.shape, -1, np.int64)
    rowidx[ing] = lut[ciy[ing] * NGRID + cix[ing]]
    sub_c = np.clip(np.floor(((px + HALF_EXTENT) % PATCH_M) / (PATCH_M / ND)).astype(np.int64), 0, ND - 1)
    sub_r = np.clip(np.floor(((HALF_EXTENT - py) % PATCH_M) / (PATCH_M / ND)).astype(np.int64), 0, ND - 1)
    assign(rowidx, sub_r, sub_c, rowidx >= 0)
    for r_i in np.where(is_t)[0]:
        dxp, dyp = px - xl[r_i], yt[r_i] - py
        sel = (dxp >= 0) & (dxp < PATCH_M) & (dyp >= 0) & (dyp < PATCH_M)
        ridx = np.full(px.shape, r_i, np.int64)
        sc = np.clip((dxp / (PATCH_M / ND)).astype(np.int64), 0, ND - 1)
        sr = np.clip((dyp / (PATCH_M / ND)).astype(np.int64), 0, ND - 1)
        assign(ridx, sr, sc, sel)
    dv = pd.concat(feats)
    dv = dv[~dv.index.duplicated(keep="last")]
    df = df.join(dv, how="left")
    df["dvn_n_cells"] = df["dvn_n_cells"].fillna(0).astype(int)
    log(f"  Diviner: {int((df['dvn_n_cells'] > 0).sum())}/{n} patches with cells, "
        f"median {int(df['dvn_n_cells'].median())} cells/patch")

    # ---------------------------------------------------------------- geology
    log("geology sampling ...")
    geo = geology_features(tif, dn, cat, xl, yt, dem_crs)
    df = pd.concat([df.reset_index(drop=True), geo.reset_index(drop=True)], axis=1)

    # ---------------------------------------------------------------- write
    front = ["patch_id", "center_lat", "center_lon", "x_m", "y_m", "is_target"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(csv_path, index=False, float_format="%.6g")
    arrays = dict(patch_id=df["patch_id"].to_numpy(str), center_lat=df["center_lat"].to_numpy(np.float32),
                  center_lon=df["center_lon"].to_numpy(np.float32), elev=np.stack(E), slope=np.stack(S),
                  sunvis=np.stack(V), dvn_tmax=g_tmax, dvn_tavg=g_tavg)
    size = sum(a.nbytes for a in arrays.values())
    if size > 500e6:
        for k in ("elev", "slope", "sunvis", "dvn_tmax", "dvn_tavg"):
            arrays[k] = arrays[k].astype(np.float16)
    np.savez_compressed(npz_path, **arrays)
    log(f"wrote {csv_path} ({n} rows, {df.shape[1]} cols) and {npz_path} "
        f"({npz_path.stat().st_size / 1e6:.0f} MB, raw {size / 1e6:.0f} MB)")
    summarize(df)
    log(f"total runtime {time.time() - t0:.0f}s")
    return 0


def summarize(df):
    print(f"\npatches: {len(df)}  (grid {int((df['is_target'] == 0).sum())}, targets {int(df['is_target'].sum())})")
    key = ["relief_m", "slope500_mean_deg", "elev_mean_m", "dvn_tavg_mean_k", "dvn_tmax_max_k",
           "dvn_frac_coldtrap_110k", "dvn_frac_ice_stable", "dvn_n_cells", "lola_avg_sun_vis_mean",
           "lola_frac_psr", "lola_avg_sun_vis_max", "moon_geo_unit_frac", "center_lat"]
    key = [k for k in key if k in df.columns]
    print(df[key].describe(percentiles=[0.5]).T[["count", "min", "50%", "max"]].round(3).to_string())
    print("\nmajority geology units:", df["moon_geo_unit"].value_counts().head(8).to_dict())
    cols = ["patch_id", "relief_m", "slope500_mean_deg", "lola_frac_psr", "lola_avg_sun_vis_max",
            "dvn_tavg_mean_k", "dvn_tmax_max_k", "dvn_frac_ice_stable", "dvn_n_cells", "moon_geo_unit"]
    print("\ntargets:")
    print(df.loc[df["is_target"] == 1, [c for c in cols if c in df.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
