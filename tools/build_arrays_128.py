#!/usr/bin/env python
"""
Sharper terrain images for the self-supervised terrain model: 128 x 128 per 10 km tile (78 m/px,
the native LOLA resolution) instead of 64 x 64 (156 m/px). Same tiles, same order as the tables.

  data/patches/moon_patch_arrays_128.npz   patch_id, center_lat, center_lon, elev, slope, sunvis (float16)
  data/patches/earth_patch_arrays_128.npz  patch_id, center_lat, center_lon, elev, slope (float16), valid (uint8)

elev = elevation minus the tile mean (m); slope in degrees computed on the 78 m grid; sunvis = LOLA average
sun visibility (Moon only). Runs locally or on Colab (downloads the two Moon rasters if missing).

  python tools/build_arrays_128.py --body both [--limit 50] [--workers 32]
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import extract as base  # noqa: E402

RAW = Path(os.environ.get("ANALOG_RAW_DIR", Path.home() / ".cache" / "earth_analog_raw"))
P = ROOT / "data" / "patches"
N = 128
PATCH_M = 10_000.0
MOON_SRC = {
    "LDEM_80S_80MPP_ADJ.TIF": "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF",
    "AVGVISIB_75S_120M_201608.TIF": "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_75S_120M_201608.TIF",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_moon_raw():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in MOON_SRC.items():
        if not (RAW / name).exists():
            log("downloading", name)
            from pdownload import main as pdl
            pdl(url, str(RAW / name), 12)


def build_moon(limit=0):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    ensure_moon_raw()
    t = pd.read_csv(P / "moon_patches.csv", usecols=["patch_id", "center_lat", "center_lon", "x_m", "y_m"])
    if limit:
        t = t.head(limit)
    n = len(t)
    E = np.zeros((n, N, N), np.float16)
    S = np.zeros((n, N, N), np.float16)
    V = np.zeros((n, N, N), np.float16)
    with rasterio.open(RAW / "LDEM_80S_80MPP_ADJ.TIF") as dem, rasterio.open(RAW / "AVGVISIB_75S_120M_201608.TIF") as vis:
        dsc, dof = (dem.scales or (1,))[0] or 1, (dem.offsets or (0,))[0] or 0
        vsc, vof = (vis.scales or (1,))[0] or 1, (vis.offsets or (0,))[0] or 0
        for i, r in enumerate(t.itertuples(index=False)):
            b = (r.x_m - PATCH_M / 2, r.y_m - PATCH_M / 2, r.x_m + PATCH_M / 2, r.y_m + PATCH_M / 2)
            z = dem.read(1, window=from_bounds(*b, transform=dem.transform), out_shape=(N, N),
                         resampling=Resampling.bilinear, boundless=True, fill_value=np.nan, out_dtype="float64")
            if dem.nodata is not None and np.isfinite(dem.nodata):
                z[z == dem.nodata] = np.nan
            z = z * dsc + dof
            if np.nanmax(np.abs(z)) < 20:          # km -> m guard
                z *= 1000
            v = vis.read(1, window=from_bounds(*b, transform=vis.transform), out_shape=(N, N),
                         resampling=Resampling.bilinear, boundless=True, fill_value=0, out_dtype="float64")
            v = np.clip(v * vsc + vof, 0, 1)
            gscale = math.cos(math.radians(90.0 + r.center_lat) / 2) ** 2     # polar-stereographic scale
            px = PATCH_M / N * gscale
            s = base._slope_deg(z, px, px)
            with np.errstate(all="ignore"):
                z = z - np.nanmean(z)
            E[i], S[i], V[i] = np.nan_to_num(z), np.nan_to_num(s), v
            if (i + 1) % 500 == 0:
                log(f"moon {i + 1}/{n}")
    out = P / "moon_patch_arrays_128.npz"
    np.savez_compressed(out, patch_id=np.asarray(t.patch_id.astype(str).tolist(), dtype="U32"), center_lat=t.center_lat.values.astype(np.float32),
                        center_lon=t.center_lon.values.astype(np.float32), elev=E, slope=S, sunvis=V)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), {n} tiles")
    return E, S


def build_earth(limit=0, workers=32):
    import build_earth_patches as bep
    t = pd.read_csv(P / "earth_patches_terrain.csv", usecols=["patch_id", "center_lat", "center_lon"])
    if limit:
        t = t.head(limit)
    n = len(t)
    E = np.zeros((n, N, N), np.float16)
    S = np.zeros((n, N, N), np.float16)
    M = np.zeros((n, N, N), np.uint8)
    t0, done, fails = time.time(), [0], []

    def one(i):
        r = t.iloc[i]
        for k in range(3):
            try:
                res = bep.read_dem(float(r.center_lat), float(r.center_lon))
                z, dx, dy = res[0], res[1], res[2]
                assert z.shape == (N, N), z.shape
                s = base._slope_deg(z, dx, dy)
                valid = np.isfinite(z)
                with np.errstate(all="ignore"):
                    z = z - np.nanmean(z)
                E[i], S[i], M[i] = np.nan_to_num(z), np.nan_to_num(s), valid
                break
            except Exception as ex:  # noqa: BLE001
                err = str(ex)[:100]
                time.sleep(2 ** k)
        else:
            fails.append((r.patch_id, err))
        done[0] += 1
        if done[0] % 300 == 0:
            el = time.time() - t0
            log(f"earth {done[0]}/{n}  {done[0] / el:.1f}/s  ETA {(n - done[0]) / (done[0] / el) / 60:.1f} min  fails={len(fails)}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(n)))
    out = P / "earth_patch_arrays_128.npz"
    np.savez_compressed(out, patch_id=np.asarray(t.patch_id.astype(str).tolist(), dtype="U32"), center_lat=t.center_lat.values.astype(np.float32),
                        center_lon=t.center_lon.values.astype(np.float32), elev=E, slope=S, valid=M)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), {n} tiles, {len(fails)} failed {fails[:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="both", choices=["moon", "earth", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.body in ("moon", "both"):
        build_moon(a.limit)
    if a.body in ("earth", "both"):
        build_earth(a.limit, a.workers)
