"""Earth land-then-drive-to-ice metrics per 10x10 km tile (20x20 km window, ~55 m grid).
Usage: python tools/earth_traverse.py [--workers 24] [--limit N] [--resume]"""
import argparse, math, os, sys, time, threading, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import extract as base
import extract_extra as xx
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform as rtransform
from scipy import ndimage
warnings.filterwarnings("ignore")
IN = ROOT / "data/patches/earth_patches_terrain.csv"
OUT = ROOT / "data/patches/earth_traverse.csv"
RAW = Path.home() / ".cache/earth_analog_raw/ant_rock"
R = 6371008.8
STEP = 0.001                       # deg lat per pixel ~111 m (coarse: reads COG overviews)
HALF = 10.0                        # km half-window
N = int(round(2 * HALF * 1000 / (STEP * math.pi / 180 * R)))   # ~360
PX_M = STEP * math.pi / 180 * R
C0, C1 = N // 4, 3 * N // 4        # central 10 km
_ant = None
_lock = threading.Lock()

def ant_mask():
    global _ant
    with _lock:
        if _ant is None:
            M = np.zeros((4000, 4000), bool)
            for ix in (0, 1):
                for iy in (0, 1):
                    with rasterio.open(RAW / f"rock_{ix}{iy}.png") as s:
                        a = s.read(4) if s.count >= 4 else s.read(1)
                    M[(1 - iy) * 2000:(2 - iy) * 2000, ix * 2000:(ix + 1) * 2000] = a > 0
            _ant = M
    return _ant

def window(lat, lon):
    dlat = HALF * 1000 / R * 180 / math.pi
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat, STEP / max(math.cos(math.radians(lat)), 1e-3)

def merged(urls, b, res, **kw):
    srcs = []
    for u in urls:
        try:
            srcs.append(rasterio.open(u))
        except Exception:
            pass
    if not srcs:
        return None
    try:
        a, _ = merge(srcs, bounds=b[:4], res=(res, STEP), **kw)
    finally:
        for s in srcs: s.close()
    return a[0]

def fit(a):
    o = np.full((N, N), np.nan if a.dtype.kind == "f" else 0, a.dtype)
    h, w = min(N, a.shape[0]), min(N, a.shape[1])
    o[:h, :w] = a[:h, :w]
    return o

def dem(lat, lon, b):
    urls = []
    for la in range(math.floor(b[1]), math.floor(b[3]) + 1):
        for lo in range(math.floor(b[0]), math.floor(b[2]) + 1):
            name = base._copernicus_tile_name(la, ((lo + 180) % 360) - 180)
            urls.append("/vsicurl/" + base.EARTH_DEM_TEMPLATE.format(name=name))
    a = merged(urls, b, b[4], nodata=np.nan, dtype="float32", resampling=rasterio.enums.Resampling.nearest)
    return None if a is None else fit(a)

def wcover(lat, lon, b):
    urls = []
    for la in range(int(math.floor(b[1] / 3)) * 3, int(math.floor(b[3] / 3)) * 3 + 1, 3):
        for lo in range(int(math.floor(b[0] / 3)) * 3, int(math.floor(b[2] / 3)) * 3 + 1, 3):
            t = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            urls.append("/vsicurl/" + xx.SRC["worldcover"].format(tile=t))
    a = merged(urls, b, b[4], nodata=0, dtype="uint8", resampling=rasterio.enums.Resampling.nearest)
    return None if a is None else fit(a)

def ant_rock(lat, lon, b):
    xs = b[0] + (np.arange(N) + .5) * b[4]
    ys = b[3] - (np.arange(N) + .5) * STEP
    X, Y = np.meshgrid(xs, ys)
    x, y = rtransform("EPSG:4326", "EPSG:3031", X.ravel().tolist(), Y.ravel().tolist())
    half, res = 2800000, 1400
    c = np.floor((np.array(x) + half) / res).astype(int); r = np.floor((half - np.array(y)) / res).astype(int)
    ok = (c >= 0) & (c < 4000) & (r >= 0) & (r < 4000)
    out = np.zeros(N * N, bool)
    out[ok] = ant_mask()[r[ok], c[ok]]
    return out.reshape(N, N)

def process(row):
    pid, lat, lon = row
    res = dict(patch_id=pid, trav_flat_bare_frac=np.nan, trav_ice_frac=np.nan, trav_dist_flat_to_ice_km=np.nan,
               trav_dist_min_flat_to_ice_km=np.nan, trav_dist_gt10_flag=0, trav_valid_frac=0.0, trav_source="none")
    b = window(lat, lon)
    z = dem(lat, lon, b)
    if z is None:
        return res
    dv = ~np.isnan(z)
    gy, gx = np.gradient(z, PX_M, PX_M)
    flat = np.degrees(np.arctan(np.hypot(gx, gy))) < 8
    if lat < -60:
        rock = ant_rock(lat, lon, b)
        bare, ice, valid, src = rock, ~rock & dv, dv, "antarctic_rock"
    else:
        w = wcover(lat, lon, b)
        if w is None or (w > 0).mean() < 0.05:
            return res
        valid = (w > 0) & dv
        bare, ice, src = (w == 60) | (w == 100), (w == 70), "worldcover"
    res["trav_valid_frac"] = float(valid.mean()); res["trav_source"] = src
    if valid.mean() < 0.05:
        res["trav_source"] = "none"; return res
    fb = flat & bare & valid
    cen = np.zeros_like(fb); cen[C0:C1, C0:C1] = True
    res["trav_flat_bare_frac"] = float(fb[C0:C1, C0:C1].sum() / max(valid[C0:C1, C0:C1].sum(), 1))
    res["trav_ice_frac"] = float(ice.sum() / max(valid.sum(), 1))
    sel = fb & cen
    if sel.any():
        if ice.any():
            d = ndimage.distance_transform_edt(~ice) * PX_M / 1000
            dd = d[sel]
            res["trav_dist_flat_to_ice_km"] = float(min(np.median(dd), 10)); res["trav_dist_min_flat_to_ice_km"] = float(dd.min())
        else:
            res["trav_dist_flat_to_ice_km"] = 10.0; res["trav_dist_min_flat_to_ice_km"] = 10.0; res["trav_dist_gt10_flag"] = 1
    return res

def safe(row):
    for k in range(3):
        try:
            return process(row)
        except Exception as e:
            err = str(e)[:100]; time.sleep(2)
    r = dict(patch_id=row[0], trav_source="none", trav_valid_frac=0.0, trav_dist_gt10_flag=0); r["error"] = err
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int); ap.add_argument("--out")
    a = ap.parse_args()
    df = pd.read_csv(IN, usecols=["patch_id", "center_lat", "center_lon"])
    if a.ids: df = df[df.patch_id.isin(a.ids)]
    if a.limit: df = df.head(a.limit)
    df = df.iloc[a.start:a.end]
    if a.out: OUT = ROOT / a.out
    done = []
    if a.resume and OUT.exists():
        old = pd.read_csv(OUT); done = old.to_dict("records")
        df = df[~df.patch_id.isin(old.patch_id)]
    rows = list(df.itertuples(index=False, name=None))
    out = list(done); t0 = time.time(); nf = 0
    def save(): pd.DataFrame(out).to_csv(OUT, index=False)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(safe, r) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); out.append(r); nf += "error" in r
            if i % 300 == 0 or i == len(rows):
                save(); el = time.time() - t0
                print(f"{i}/{len(rows)} {el:.0f}s ETA {el / i * (len(rows) - i):.0f}s fails={nf}", flush=True)
    save(); print("done fails", nf)
