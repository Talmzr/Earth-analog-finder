#!/usr/bin/env python
"""
Earth 10 km x 10 km terrain + land-cover patches for every candidate in data/patches/earth_candidates.csv.

Per patch (half-width 5 km, centred on the candidate):
  terrain       Copernicus GLO-30 DEM, bilinearly resampled onto a 128 x 128 lat/lon grid (78.1 m/px,
                comparable to the 80 m LOLA Moon DEM). Fetched as server-side crops from the Microsoft
                Planetary Computer data API (collection cop-dem-glo-30, no login) because the local link
                only sustains ~1.5 MB/s and whole 1024^2 COG blocks from AWS cost ~2 MB each; falls back
                to the AWS COGs (2x overview, ~60 m) if the API fails.
                extract.terrain_metrics -> relief, slope at 500 m baseline, TRI ...
  land cover    ESA WorldCover 2021 class fractions on the same 128 x 128 grid (nearest), via the Planetary
                Computer data API (fallback: extract_extra.g_worldcover); NaN south of 60 S
  illumination  extract.illumination(lat, "earth") (analytic solar geometry)
  arrays        64 x 64 elevation (2x2 block mean -> 156.25 m/px, minus mean) and slope (deg, from the
                64 x 64 grid), same layout as data/patches/moon_patch_arrays.npz

Outputs (rewritten every --flush-every completed patches, so partial results are usable; the
outputs double as the resume checkpoint):
  data/patches/earth_patches_terrain.csv
  data/patches/earth_patch_arrays.npz      patch_id, center_lat, center_lon, elev, slope, valid
  data/patches/earth_patches_failed.csv    patch_id, error, permanent (ocean / <50 % valid DEM)

Usage:  python tools/build_earth_patches.py [--workers 32] [--limit N] [--retry-failed] [--refresh]
"""
import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract as base  # noqa: E402

# GDAL config as environment variables so they apply in every worker thread
for _k, _v in base.GDAL_ENV.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", str(512 * 1024 * 1024))
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import extract_extra as xx  # noqa: E402

RAW = Path.home() / ".cache" / "earth_analog_raw"
OUT = ROOT / "data" / "patches"
CAND = OUT / "earth_candidates.csv"
CSV_PATH = OUT / "earth_patches_terrain.csv"
NPZ_PATH = OUT / "earth_patch_arrays.npz"
FAIL_PATH = OUT / "earth_patches_failed.csv"

HW_KM = 5.0
NG = 128                  # read grid: 10 km / 128 = 78.125 m
NA = 64                   # array grid: 156.25 m
R = base.BODY["earth"]["R"]
SOURCE_ORDER = {"named_site": 0, "impact_structure": 1, "volcano": 2, "landcover_sample": 3}
TILELIST_URL = "https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt"
TILELIST90_URL = "https://copernicus-dem-90m.s3.amazonaws.com/tileList.txt"
_TILES = None
_TILES90 = None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tile_set():
    global _TILES
    if _TILES is None:
        f = RAW / "copernicus_glo30_tileList.txt"
        if not f.exists():
            r = requests.get(TILELIST_URL, timeout=120)
            r.raise_for_status()
            f.write_text(r.text)
        _TILES = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
    return _TILES


def tile_set90():
    """GLO-90 tiles (adds Armenia / Azerbaijan, which are missing from the public GLO-30 release)."""
    global _TILES90
    if _TILES90 is None:
        f = RAW / "copernicus_glo90_tileList.txt"
        if not f.exists():
            r = requests.get(TILELIST90_URL, timeout=120)
            r.raise_for_status()
            f.write_text(r.text)
        _TILES90 = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
    return _TILES90


def _open_tile(name, tries=4):
    import rasterio
    url = "/vsicurl/" + base.EARTH_DEM_TEMPLATE.format(name=name)
    last = None
    for i in range(tries):
        try:
            try:
                return rasterio.open(url, OVERVIEW_LEVEL=0)    # 2x overview (~60 m)
            except Exception:  # noqa: BLE001  tile without overviews -> full resolution
                return rasterio.open(url)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * 2 ** i)
    raise RuntimeError(f"open failed {name}: {str(last)[:120]}")


PC_BBOX = "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/{w:.7f},{s:.7f},{e:.7f},{n:.7f}/{ng}x{ng}.tif"
_TL = threading.local()


def _session():
    if not hasattr(_TL, "s"):
        _TL.s = requests.Session()
        _TL.s.headers["User-Agent"] = "space-apps-analog-finder/1.0"
    return _TL.s


def _pc_crop(bounds, ng, params, tries=6):
    """Server-side crop from the Planetary Computer data API -> 2-D array, or None on HTTP 404."""
    from rasterio.io import MemoryFile
    w, s, e, n = bounds
    url = PC_BBOX.format(w=w, s=s, e=e, n=n, ng=ng)
    last = None
    for i in range(tries):
        try:
            r = _session().get(url, params=params, timeout=90)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            with MemoryFile(r.content) as mf, mf.open() as ds:
                return ds.read(1)
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(min(30, 1.5 * 2 ** i))
    raise RuntimeError(f"PC crop failed: {str(last)[:120]}")


def _bounds(lat, lon, hw_km=HW_KM):
    dlat = math.degrees(hw_km * 1000 / R)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _dem_tiles(w, s, e, n, glo90=False):
    names = []
    for la in range(math.floor(s), math.floor(n) + 1):
        for lo in range(math.floor(w), math.floor(e) + 1):
            if -90 <= la < 90:
                names.append(base._copernicus_tile_name(la, ((lo + 180) % 360) - 180))
    if glo90:
        tiles = tile_set90()
        names = [nm.replace("_COG_10_", "_COG_30_") for nm in names]
    else:
        tiles = tile_set()
    return [nm for nm in names if nm in tiles]


def read_dem(lat, lon, hw_km=HW_KM, ng=NG):
    """128 x 128 DEM grid (78 m) centred on lat/lon via PC server-side crops; AWS COG fallback."""
    w, s, e, n = _bounds(lat, lon, hw_km)
    names = _dem_tiles(w, s, e, n)
    coll, src = "cop-dem-glo-30", "pc"
    if not names:  # GLO-30 gap (Armenia / Azerbaijan) -> GLO-90
        names = _dem_tiles(w, s, e, n, glo90=True)
        coll, src = "cop-dem-glo-90", "pc_glo90"
    if not names:
        raise LookupError("no Copernicus tile (ocean)")
    try:
        z = np.full((ng, ng), np.nan)
        for nm in names:
            a = _pc_crop((w, s, e, n), ng, dict(collection=coll, item=nm, assets="data",
                                                 resampling="bilinear", nodata=-9999, return_mask="false"))
            if a is None:
                continue
            a = a.astype(np.float64)
            a[a <= -9998] = np.nan
            z = np.where(np.isnan(z), a, z)
    except RuntimeError:
        if coll != "cop-dem-glo-30":
            raise
        z, _, _, _ = read_dem_aws(lat, lon, hw_km, ng)
        src = "aws"
    xres, yres = (e - w) / ng, (n - s) / ng
    dx = float(base.haversine(lon, lat, lon + xres, lat, R))
    dy = float(base.haversine(lon, lat, lon, lat + yres, R))
    return z, dx, dy, len(names), src


def read_dem_aws(lat, lon, hw_km=HW_KM, ng=NG):
    """128 x 128 DEM grid (bilinear from the ~60 m overview) centred on lat/lon; NaN where no tile."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    w, s, e, n = _bounds(lat, lon, hw_km)
    xres, yres = (e - w) / ng, (n - s) / ng
    names = _dem_tiles(w, s, e, n)
    if not names:
        raise LookupError("no Copernicus tile (ocean)")
    last = None
    for attempt in range(3):
        srcs = []
        try:
            with rasterio.Env():
                srcs = [_open_tile(nm) for nm in names]
                mos, _ = merge(srcs, bounds=(w, s, e, n), res=(xres, yres), nodata=np.nan,
                               dtype="float64", resampling=Resampling.bilinear)
            z = np.full((ng, ng), np.nan)
            a = mos[0][:ng, :ng]
            z[:a.shape[0], :a.shape[1]] = a
            break
        except Exception as ex:  # noqa: BLE001
            last = ex
            time.sleep(2 * 2 ** attempt)
        finally:
            for sd in srcs:
                sd.close()
    else:
        raise RuntimeError(f"DEM read failed: {str(last)[:150]}")
    dx = float(base.haversine(lon, lat, lon + xres, lat, R))
    dy = float(base.haversine(lon, lat, lon, lat + yres, R))
    return z, dx, dy, len(names)


def worldcover(lat, lon):
    if lat < -60:
        return {}, "no WorldCover south of 60S"
    try:
        return worldcover_pc(lat, lon)
    except Exception as e:  # noqa: BLE001
        log(f"  WorldCover PC failed at {lat:.3f},{lon:.3f} ({str(e)[:60]}) -> AWS fallback")
    out, note = {}, ""
    for attempt in range(3):
        try:
            out, note = xx.g_worldcover(lat, lon, HW_KM)
            if out:
                return out, note
        except Exception as e:  # noqa: BLE001
            note = f"worldcover error: {str(e)[:100]}"
        time.sleep(1 + attempt)
    return out, note


def worldcover_pc(lat, lon, ng=NG):
    w, s, e, n = _bounds(lat, lon)
    a = np.zeros((ng, ng), np.uint8)
    for la in range(int(math.floor(s / 3)) * 3, int(math.floor(n / 3)) * 3 + 1, 3):
        for lo in range(int(math.floor(w / 3)) * 3, int(math.floor(e / 3)) * 3 + 1, 3):
            lo_w = ((lo + 180) % 360) - 180
            tile = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo_w >= 0 else 'W'}{abs(lo_w):03d}"
            b = _pc_crop((w, s, e, n), ng, dict(collection="esa-worldcover",
                                                item=f"ESA_WorldCover_10m_2021_v200_{tile}",
                                                assets="map", resampling="nearest", return_mask="false"))
            if b is not None:
                a = np.where(a == 0, b.astype(np.uint8), a)
    valid = a > 0
    if valid.mean() < 0.05:
        return {}, "no WorldCover data in window (ocean / outside coverage)"
    out = {f"wc_frac_{nm}": float((a[valid] == c).mean()) for c, nm in xx.WC_CLASSES.items()}
    out["wc_valid_frac"] = float(valid.mean())
    return out, ""


def process(row):
    pid, lat, lon = row["patch_id"], float(row["lat"]), float(row["lon"])
    rec = dict(patch_id=pid, source=row["source"], stratum=row["stratum"], name=row["name"],
               center_lat=lat, center_lon=lon, hw_km=HW_KM)
    try:
        z, dx, dy, nt, dsrc = read_dem(lat, lon)
        rec.update(base.terrain_metrics(z, dx, dy))
    except LookupError as e:
        return pid, None, None, None, (str(e), True)
    except RuntimeError as e:
        msg = str(e)
        return pid, None, None, None, (msg, "valid" in msg or "too small" in msg)
    except Exception as e:  # noqa: BLE001
        return pid, None, None, None, (f"{type(e).__name__}: {str(e)[:150]}", False)
    zv = z[np.isfinite(z)]
    rec.update(dem_fetch=dsrc, n_dem_tiles=nt, elev_min_m=float(zv.min()), elev_max_m=float(zv.max()), elev_range_m=float(zv.max() - zv.min()),
               frac_elev_zero=float(np.mean(np.abs(zv) < 0.01)))
    # 64x64 arrays: 2x2 block mean (156.25 m), slope from that grid, elevation minus mean
    e64 = base._block_mean(z, NG // NA, NG // NA)
    s64 = base._slope_deg(e64, dx * NG / NA, dy * NG / NA)
    valid = np.isfinite(e64)
    with np.errstate(all="ignore"):
        e64 = e64 - np.nanmean(e64)
    wc, note = worldcover(lat, lon)
    rec.update(wc)
    rec["wc_note"] = note
    rec.update(base.illumination(lat, "earth"))
    return pid, rec, e64.astype(np.float32), s64.astype(np.float32), valid


def _replace(tmp, dst):
    for i in range(10):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    log(f"WARNING could not replace {dst} (file locked?)")
    return False


def flush(cands, recs, arrs, fails, final=False):
    order = {p: i for i, p in enumerate(cands["patch_id"])}
    ids = sorted(recs, key=order.get)
    df = pd.DataFrame([recs[p] for p in ids])
    front = ["patch_id", "source", "stratum", "name", "center_lat", "center_lon", "hw_km"]
    wc_cols = [c for c in df.columns if c.startswith("wc_")]
    rest = [c for c in df.columns if c not in front and c not in wc_cols]
    df = df[front + rest + wc_cols]
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, float_format="%.6g")
    _replace(tmp, CSV_PATH)
    if ids:
        arrays = dict(patch_id=np.array(ids, dtype="<U32"),
                      center_lat=df["center_lat"].to_numpy(np.float32),
                      center_lon=df["center_lon"].to_numpy(np.float32),
                      elev=np.stack([arrs[p][0] for p in ids]), slope=np.stack([arrs[p][1] for p in ids]),
                      valid=np.stack([arrs[p][2] for p in ids]).astype(np.uint8))
        tmpn = NPZ_PATH.with_name("earth_patch_arrays.tmp.npz")
        (np.savez_compressed if final else np.savez)(tmpn, **arrays)
        _replace(tmpn, NPZ_PATH)
    if fails:
        fd = pd.DataFrame([dict(patch_id=p, error=e, permanent=perm) for p, (e, perm) in fails.items()])
        fd = fd.merge(cands[["patch_id", "source", "stratum", "name", "lat", "lon"]], on="patch_id", how="left")
        fd["_o"] = fd["patch_id"].map(order)
        tmpf = FAIL_PATH.with_suffix(".csv.tmp")
        fd.sort_values("_o").drop(columns="_o").to_csv(tmpf, index=False)
        _replace(tmpf, FAIL_PATH)
    return df


def load_existing(cands):
    recs, arrs, fails = {}, {}, {}
    if CSV_PATH.exists() and NPZ_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        z = np.load(NPZ_PATH)
        ids = list(z["patch_id"])
        if ids == list(df["patch_id"]):
            valid = z["valid"] if "valid" in z.files else np.isfinite(z["elev"])
            for i, p in enumerate(ids):
                recs[p] = {k: v for k, v in df.iloc[i].to_dict().items()}
                arrs[p] = (z["elev"][i], z["slope"][i], valid[i].astype(bool))
        else:
            log("existing CSV / NPZ out of sync -> ignoring them")
    if FAIL_PATH.exists():
        fd = pd.read_csv(FAIL_PATH)
        for _, r in fd.iterrows():
            fails[r["patch_id"]] = (r["error"], bool(r["permanent"]))
    return recs, arrs, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--flush-every", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated patch ids (test mode, no output written)")
    ap.add_argument("--retry-failed", action="store_true", help="also retry permanent failures")
    ap.add_argument("--refresh", action="store_true", help="ignore existing outputs")
    a = ap.parse_args()

    cands = pd.read_csv(CAND)
    cands["_o"] = cands["source"].map(SOURCE_ORDER).fillna(9)
    cands = cands.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    tile_set()

    if a.only:
        ids = [s.strip() for s in a.only.split(",")]
        for _, row in cands[cands["patch_id"].isin(ids)].iterrows():
            t = time.time()
            pid, rec, e, s, v = process(row)
            if rec is None:
                print(pid, "FAILED", v)
            else:
                print(pid, f"{time.time() - t:.1f}s", {k: (round(x, 3) if isinstance(x, float) else x)
                                                        for k, x in rec.items() if not k.startswith("noon_elev_m")})
        return

    recs, arrs, fails = ({}, {}, {}) if a.refresh else load_existing(cands)
    todo = cands[~cands["patch_id"].isin(recs)]
    if not a.retry_failed:
        perm = {p for p, (_, pf) in fails.items() if pf}
        todo = todo[~todo["patch_id"].isin(perm)]
    if a.limit:
        todo = todo.head(a.limit)
    log(f"{len(cands)} candidates, {len(recs)} done, {len(fails)} failed before; {len(todo)} to process "
        f"with {a.workers} threads")
    t0 = time.time()
    n_done = n_fail = 0
    since_flush = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, row) for _, row in todo.iterrows()]  # FIFO: named -> craters -> volcanoes -> samples
        for f in as_completed(futs):
            try:
                pid, rec, e, s, v = f.result()
            except Exception as exc:  # noqa: BLE001
                log("worker crashed:", exc)
                continue
            if rec is None:
                fails[pid] = v
                n_fail += 1
            else:
                recs[pid] = rec
                arrs[pid] = (e, s, v)
                fails.pop(pid, None)
                n_done += 1
            k = n_done + n_fail
            since_flush += 1
            if k % 200 == 0:
                el = time.time() - t0
                log(f"{k}/{len(todo)} processed ({n_done} ok, {n_fail} failed)  {k / el:.2f}/s  "
                    f"ETA {(len(todo) - k) / max(k / el, 1e-9) / 60:.1f} min")
            if since_flush >= a.flush_every:
                flush(cands, recs, arrs, fails)
                since_flush = 0
                log(f"  flushed {len(recs)} patches -> {CSV_PATH.name}, {NPZ_PATH.name}")
    df = flush(cands, recs, arrs, fails, final=True)
    log(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(df)} patches written, {len(fails)} failures")
    print(df.groupby(["source"]).size().to_string())
    if fails:
        fd = pd.DataFrame([dict(patch_id=p, error=e[:40], permanent=pf) for p, (e, pf) in fails.items()])
        print(fd.groupby(["permanent", "error"]).size().sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
