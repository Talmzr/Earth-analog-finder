"""Build the EARTH environment / climate / soil / geology feature table for 10 km x 10 km patches.

    python tools/build_earth_env.py                 # all candidates -> data/patches/earth_patches_env.csv
    python tools/build_earth_env.py --sites         # the 30 named sites in config/earth_sites.csv (test)
    python tools/build_earth_env.py --fast-only     # only the quick pass (0.2 deg JAXA, 5 km SoilGrids)

Strategy: global rasters are downloaded ONCE into ~/.cache/earth_analog_raw and sampled locally; the point
APIs (NASA POWER, Macrostrat) are deduplicated, run concurrently in background threads (named sites and impact
structures first), cached as JSON on disk and capped in wall time. The output CSV is (atomically) rewritten
after every phase, so downstream users always get the full final schema with whatever is done so far;
data/patches/earth_patches_env.status.json says which sources are complete.

  phase A (fast):  JAXA level-0 (0.2 deg) normals, SoilGrids 5 km aggregate, CCI permafrost files present
  phase B (full):  JAXA level-1 (0.05 deg LST/NDVI, 0.1 deg AMSR2), SoilGrids 1 km, CCI ground temperature NH
  background:      NASA POWER climatology (0.5 x 0.625 deg cells), Macrostrat units

Sources and units: see data/patches/README_earth_env.md.
"""
import argparse
import json
import math
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.simplefilter("ignore", RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("ANALOG_RAW_DIR", Path.home() / ".cache" / "earth_analog_raw"))
HW_KM = 5.0  # patch half-width
KM_PER_DEG = 111.32

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "space-apps-2026-analog-finder"
_ad = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
SESSION.mount("https://", _ad)

# ----------------------------------------------------------------------------- logging
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class Progress:
    def __init__(self, name, total, every=None):
        self.name, self.total, self.n, self.t0 = name, total, 0, time.time()
        self.every = every or max(1, total // 20)
        self.lock = threading.Lock()

    def tick(self, k=1):
        with self.lock:
            self.n += k
            if self.n % self.every == 0 or self.n == self.total:
                el = time.time() - self.t0
                rate = self.n / max(el, 1e-6)
                eta = (self.total - self.n) / max(rate, 1e-6)
                log(f"{self.name}: {self.n}/{self.total} ({rate:.1f}/s, ETA {eta / 60:.1f} min)")


# ----------------------------------------------------------------------------- HTTP helpers
def get_json(url, params=None, tries=6, timeout=90, backoff=2.0):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(backoff * (2 ** i) + np.random.rand())
    raise RuntimeError(f"GET failed {url} {params}: {last}")


def download(url, path, tries=6):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".dl")
    last = None
    for i in range(tries):
        try:
            with SESSION.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                n = int(r.headers.get("Content-Length", 0))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if n and tmp.stat().st_size != n:
                raise IOError(f"short read {tmp.stat().st_size}/{n}")
            tmp.replace(path)
            return path
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"download failed {url}: {last}")


def wait_for(paths, timeout_s, what):
    """Wait for files being fetched by another process (e.g. tools/pdownload.py) to appear."""
    t = time.time()
    paths = [Path(p) for p in paths]
    while not all(p.exists() for p in paths):
        if time.time() - t > timeout_s:
            log(f"{what}: not available after {timeout_s / 60:.0f} min -> capped")
            return False
        time.sleep(10)
    return True


def win_deg(lat, hw_km=HW_KM):
    dlat = hw_km / KM_PER_DEG
    dlon = dlat / max(math.cos(math.radians(lat)), 0.02)
    return dlat, min(dlon, 180.0)


# ============================================================================= JAXA Earth monthly normals
JAXA = "https://s3.ap-northeast-1.wasabisys.com/je-pds/cog/v1"
# level 1 = 8 tiles (90 x 90 deg) at native res (0.05 deg MODIS LST/NDVI/AOT, 0.1 deg GSMaP/AMSR2);
# level 0 = 2 tiles (180 x 180 deg) at 0.2 deg.
TILES = {
    1: [(lo, la) for lo in [("W180.00", "W090.00"), ("W090.00", "E000.00"), ("E000.00", "E090.00"),
                            ("E090.00", "E180.00")] for la in [("S90.00", "N00.00"), ("N00.00", "N90.00")]],
    0: [(lo, ("S90.00", "N90.00")) for lo in [("W180.00", "E000.00"), ("E000.00", "E180.00")]],
}
# key: (collection, asset, dn slope, valid_min_dn, valid_max_dn, widen-to km if patch window empty)
JAXA_PRODUCTS = {
    "lst_day": ("NASA.EOSDIS_Terra.MODIS_MOD11C1-LST.daytime.v061_global_monthly-normal", "LST_2012_2021",
                0.02, 7500, 65534, None),
    "lst_night": ("NASA.EOSDIS_Terra.MODIS_MOD11C1-LST.nighttime.v061_global_monthly-normal", "LST_2012_2021",
                  0.02, 7500, 65534, None),
    "ndvi": ("JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly-normal", "ndvi_2012_2021",
             0.0001, 0, 10000, None),
    "aot": ("JAXA.JASMES_Terra.MODIS-Aqua.MODIS_taua.v811_global_monthly-normal", "taua_2012_2021",
            0.001, 0, 65534, 25.0),
    "sm": ("JAXA.G-Portal_GCOM-W.AMSR2_standard.L3-SMC.daytime.v3_global_monthly-normal", "SMC_2012_2021",
           0.1, 0, 1000, 25.0),
    "precip": ("JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly-normal", "PRECIP_2012_2021",
               1.0, 0.0, 1e4, 25.0),
}
LEVELS_FAST = {"lst_day": 0, "lst_night": 0, "ndvi": 0, "aot": 0, "sm": 0, "precip": 0}
# full: native 0.05 deg for LST/NDVI and 0.1 deg AMSR2; AOT (smooth field) and GSMaP stay at 0.2 deg to save ~400 MB
LEVELS_FULL = {"lst_day": 1, "lst_night": 1, "ndvi": 1, "aot": 0, "sm": 1, "precip": 0}
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def jaxa_tiles(prod, lvl):
    coll, asset = JAXA_PRODUCTS[prod][:2]
    out = []
    for m in range(1, 13):
        for (lo0, lo1), (la0, la1) in TILES[lvl]:
            rel = f"{m:02d}/{lvl}/{lo0}-{lo1}/{lo0}-{la0}-{lo1}-{la1}-{asset}.tif"
            out.append((m, f"{JAXA}/{coll}/{rel}", RAW / "jaxa" / prod / rel.replace("/", "_")))
    return out


def jaxa_complete(prod, lvl):
    return all(p.exists() for _, _, p in jaxa_tiles(prod, lvl))


def jaxa_download(levels, workers=16, cap_s=None):
    order = ["sm", "lst_night", "lst_day", "ndvi", "aot", "precip"]
    jobs = [t for p in order if p in levels for t in jaxa_tiles(p, levels[p]) if not t[2].exists()]
    if not jobs:
        return
    deadline = time.time() + cap_s if cap_s else None
    log(f"JAXA: downloading {len(jobs)} COG tiles with {workers} threads")
    pr = Progress("JAXA download", len(jobs))
    errs = []

    def one(j):
        if deadline and time.time() > deadline:
            return
        try:
            download(j[1], j[2])
        except Exception as e:  # noqa: BLE001
            errs.append(f"{j[1]}: {e}")
        pr.tick()

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))
    for e in errs[:10]:
        log("  JAXA download error:", e)


def jaxa_global_month(prod, lvl, month):
    """Mosaic the tiles of one month into a global float32 array (north-up, origin -180/90)."""
    import rasterio
    coll, asset, slope, vmin, vmax = JAXA_PRODUCTS[prod][:5]
    G = res = None
    for _, _, path in [t for t in jaxa_tiles(prod, lvl) if t[0] == month]:
        if not path.exists():
            continue
        with rasterio.open(path) as s:
            a = s.read(1).astype(np.float32)
            t = s.transform
        if res is None:
            res = t.a
            G = np.full((int(round(180 / res)), int(round(360 / res))), np.nan, dtype=np.float32)
        bad = (a < vmin) | (a > vmax) | ~np.isfinite(a)
        a *= slope
        a[bad] = np.nan
        r0, c0 = int(round((90 - t.f) / res)), int(round((t.c + 180) / res))
        G[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
    return G, res


def window_mean(G, res, lat, lon, hw_km):
    """Mean of the pixels intersecting the patch box (always includes the containing pixel)."""
    H, W = G.shape
    dlat, dlon = win_deg(lat, hw_km)
    r0 = max(int(math.floor((90 - (lat + dlat)) / res)), 0)
    r1 = min(int(math.floor((90 - (lat - dlat)) / res)), H - 1)
    c0 = int(math.floor((lon - dlon + 180) / res))
    c1 = int(math.floor((lon + dlon + 180) / res))
    w = G[r0:r1 + 1][:, np.arange(c0, c1 + 1) % W]
    if not np.isfinite(w).any():
        return np.nan
    return float(np.nanmean(w))


def step_jaxa(pts, args, fast):
    want = LEVELS_FAST if fast else LEVELS_FULL
    jaxa_download(want, workers=args.threads, cap_s=None if fast else args.cap_min * 60)
    # per product: use the wanted level if complete, else fall back to the fast level
    levels = {p: (lv if jaxa_complete(p, lv) else LEVELS_FAST[p]) for p, lv in want.items()}
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    N = len(pts)
    M, filled, resd = {}, {}, {}
    for prod, lvl in levels.items():
        fb = JAXA_PRODUCTS[prod][5]
        arr = np.full((N, 12), np.nan)
        fill = np.zeros(N, dtype=bool)
        for m in range(1, 13):
            G, res = jaxa_global_month(prod, lvl, m)
            if G is None:
                continue
            resd[prod] = res
            for i in range(N):
                v = window_mean(G, res, lat[i], lon[i], HW_KM)
                if not np.isfinite(v) and fb:
                    v = window_mean(G, res, lat[i], lon[i], fb)
                    fill[i] |= bool(np.isfinite(v))
                arr[i, m - 1] = v
            del G
        M[prod], filled[prod] = arr, fill
        log(f"JAXA {prod} (level {lvl}, {resd.get(prod)} deg): any-month coverage "
            f"{np.isfinite(arr).any(1).mean() * 100:.1f}%")

    d, n = M["lst_day"], M["lst_night"]
    both = np.where(np.isfinite(d) & np.isfinite(n), (d + n) / 2, np.nan)
    monthly = np.where(np.isfinite(both), both, np.where(np.isfinite(d), d, n))
    out = pd.DataFrame(index=pts.index)
    out["lst_day_max_k"] = np.nanmax(d, 1)
    out["lst_day_mean_k"] = np.nanmean(d, 1)
    out["lst_night_min_k"] = np.nanmin(n, 1)
    out["lst_night_mean_k"] = np.nanmean(n, 1)
    out["lst_mean_k"] = np.nanmean(monthly, 1)
    out["lst_diurnal_range_k"] = np.nanmean(d - n, 1)
    out["lst_annual_range_k"] = np.nanmax(monthly, 1) - np.nanmin(monthly, 1)
    out["lst_day_months_valid"] = np.isfinite(d).sum(1)
    out["lst_night_months_valid"] = np.isfinite(n).sum(1)
    out["lst_res_deg"] = resd.get("lst_day", np.nan)

    sm = M["sm"]
    out["soil_moist_mean"] = np.nanmean(sm, 1)
    out["soil_moist_max"] = np.nanmax(sm, 1)
    out["soil_moist_months_valid"] = np.isfinite(sm).sum(1)
    out["soil_moist_widened_25km"] = filled["sm"]

    mm = M["precip"] * 24 * DAYS_IN_MONTH  # mm/h monthly-normal rate -> mm/month
    nval = np.isfinite(mm).sum(1)
    out["precip_mm_yr"] = np.where(nval >= 9, np.nanmean(mm, 1) * 12, np.nan)
    out["precip_wettest_month_mm"] = np.where(nval >= 9, np.nanmax(mm, 1), np.nan)
    out["precip_widened_25km"] = filled["precip"]

    out["ndvi_mean"] = np.nanmean(M["ndvi"], 1)
    out["ndvi_max"] = np.nanmax(M["ndvi"], 1)
    out["ndvi_res_deg"] = resd.get("ndvi", np.nan)
    out["aot_mean"] = np.nanmean(M["aot"], 1)
    out["aot_max"] = np.nanmax(M["aot"], 1)
    out["aot_widened_25km"] = filled["aot"]
    return out, {p: dict(level=levels[p], res_deg=resd.get(p)) for p in levels}


# ============================================================================= SoilGrids 2.0 (aggregated)
SG = "https://files.isric.org/soilgrids/latest/data_aggregated/{r}m/{p}/{p}_0-5cm_mean_{r}.tif"
# property -> (output column, divisor to output unit)
SG_PROPS = {"sand": ("sg_sand_pct", 10.0), "silt": ("sg_silt_pct", 10.0), "clay": ("sg_clay_pct", 10.0),
            "bdod": ("sg_bdod_g_cm3", 100.0), "cfvo": ("sg_cfvo_pct", 10.0)}


def sg_path(p, r):
    return RAW / "soilgrids" / f"{p}_0-5cm_mean_{r}.tif"


def step_soil(pts, args, r):
    import rasterio
    from rasterio.warp import transform as wtransform
    from rasterio.windows import from_bounds
    if r == 5000:
        with ThreadPoolExecutor(5) as ex:
            list(ex.map(lambda p: download(SG.format(p=p, r=r), sg_path(p, r)), SG_PROPS))
    out = pd.DataFrame(index=pts.index)
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()

    def one(p):
        col, div = SG_PROPS[p]
        vals = np.full(len(pts), np.nan)
        with rasterio.open(sg_path(p, r)) as s:
            xs, ys = wtransform("EPSG:4326", s.crs, list(lon), list(lat))
            nod = s.nodata if s.nodata is not None else -32768
            hw = HW_KM * 1000
            for i, (x, y) in enumerate(zip(xs, ys)):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                win = from_bounds(x - hw, y - hw, x + hw, y + hw, transform=s.transform)
                win = win.round_offsets().round_lengths()
                if win.width < 1 or win.height < 1:
                    from rasterio.windows import Window
                    rr, cc = s.index(x, y)
                    win = Window(cc, rr, 1, 1)
                a = s.read(1, window=win, boundless=True, fill_value=nod).astype(float)
                bad = (a == nod) | (a < 0)
                if p != "cfvo":
                    bad |= a == 0  # 0 = no soil (water / ice / rock mask) for texture and density
                a[bad] = np.nan
                if np.isfinite(a).any():
                    vals[i] = np.nanmean(a) / div
        return col, vals

    with ThreadPoolExecutor(len(SG_PROPS)) as ex:
        for col, vals in ex.map(one, list(SG_PROPS)):
            out[col] = vals
    out.loc[out["sg_sand_pct"].isna(), "sg_cfvo_pct"] = np.nan
    out["sg_res_m"] = r
    log(f"SoilGrids {r} m: coverage {out['sg_sand_pct'].notna().mean() * 100:.1f}%")
    return out


# ============================================================================= ESA CCI Permafrost v5 (2023)
CEDA = "https://dap.ceda.ac.uk/neodc/esacci/permafrost/data"
PF_FILES = {
    ("PFR", "nh"): f"{CEDA}/permafrost_extent/L4/area4/pp/v05.0/northern_hemisphere/2023/"
                   "ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc",
    ("PFR", "ant"): f"{CEDA}/permafrost_extent/L4/area4/pp/v05.0/Antarctica/2023/"
                    "ESACCI-PERMAFROST-L4-PFR-MODISLST_CRYOGRID-AREA27_PP-2023-fv05.0.nc",
    ("GTD", "nh"): f"{CEDA}/ground_temperature/L4/area4/pp/v05.0/northern_hemisphere/2023/"
                   "ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc",
    ("GTD", "ant"): f"{CEDA}/ground_temperature/L4/area4/pp/v05.0/Antarctica/2023/"
                    "ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA27_PP-2023-fv05.0.nc",
}
# (file kind, netCDF variable, output column, scale, offset, nodata)
PF_VARS = [("PFR", "PFR", "pf_frac_pct", 1.0, 0.0, 255),
           ("GTD", "GST", "pf_gst_c", 0.01, -273.15, 65535),
           ("GTD", "T2m", "pf_magt_2m_c", 0.01, -273.15, 65535),
           ("GTD", "T10m", "pf_magt_10m_c", 0.01, -273.15, 65535)]


def pf_path(kind, region):
    return RAW / "permafrost" / PF_FILES[(kind, region)].rsplit("/", 1)[-1]


def _pf_grid(path):
    """netCDF lat/lon -> (top, bottom, left, res). GDAL presents netCDF north-up: top = max(lat) + res/2."""
    import rasterio
    with rasterio.open(f'netcdf:"{path}":lat') as t:
        la = t.read(1).ravel().astype(float)
    with rasterio.open(f'netcdf:"{path}":lon') as t:
        lo = t.read(1).ravel().astype(float)
    res = abs(la[1] - la[0])
    return la.max() + res / 2, la.min() - res / 2, lo.min() - res / 2, res


def step_permafrost(pts, args):
    import rasterio
    from rasterio.windows import Window
    for (kind, region), url in PF_FILES.items():  # small files: fetch here; the 361 MB GTD-NH comes via pdownload
        p = pf_path(kind, region)
        if not p.exists() and not Path(str(p) + ".part").exists():
            try:
                download(url, p)
            except Exception as e:  # noqa: BLE001
                log("Permafrost download failed", p.name, e)
    lat, lon = pts["lat"].to_numpy(), pts["lon"].to_numpy()
    out = pd.DataFrame(index=pts.index)
    have = {}
    for kind, var, col, sc, off, nod in PF_VARS:  # sequential: libnetcdf/HDF5 are not thread-safe
        vals = np.full(len(pts), np.nan)
        dom = np.zeros(len(pts), dtype=bool)
        for region in ("nh", "ant"):
            path = pf_path(kind, region)
            have[f"{kind}_{region}"] = path.exists()
            if not path.exists():
                continue
            top, bottom, left, res = _pf_grid(path)
            idx = np.where((lat <= top) & (lat >= bottom))[0]
            dom[idx] = True
            with rasterio.open(f'netcdf:"{path}":{var}') as s:
                H, W = s.height, s.width
                for i in idx[np.argsort(-lat[idx])]:  # sweep north->south for block-cache locality
                    dlat, dlon = win_deg(lat[i])
                    r0 = max(int(math.floor((top - (lat[i] + dlat)) / res)), 0)
                    r1 = min(int(math.floor((top - (lat[i] - dlat)) / res)), H - 1)
                    c0 = max(int(math.floor((lon[i] - dlon - left) / res)), 0)
                    c1 = min(int(math.floor((lon[i] + dlon - left) / res)), W - 1)
                    if r1 < r0 or c1 < c0:
                        continue
                    a = s.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)).astype(float)
                    a[a == nod] = np.nan
                    if np.isfinite(a).any():
                        vals[i] = np.nanmean(a) * sc + off
        out[col] = vals
        if col == "pf_frac_pct":
            out["pf_in_cci_domain"] = dom
        log(f"Permafrost {col}: coverage {np.isfinite(vals).mean() * 100:.1f}% (in CCI domain {dom.mean() * 100:.1f}%)")
    return out, have


# ============================================================================= NASA POWER climatology
POWER = "https://power.larc.nasa.gov/api/temporal/climatology/point"
POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "FROST_DAYS", "RH2M", "PRECTOTCORR", "WS2M", "PS",
                "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_UV_INDEX", "ALLSKY_SRF_ALB"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def power_cell(lat, lon):
    """Snap to the MERRA-2 0.5 x 0.625 deg cell centre POWER uses for meteorology."""
    la = min(max(round(lat / 0.5) * 0.5, -90), 90)
    lo = ((round(lon / 0.625) * 0.625 + 180) % 360) - 180
    return round(la, 3), round(lo, 4)


POWER_COOLDOWN = [0.0]  # shared across workers: POWER answers bursts with HTTP 429 for a few minutes
POWER_LOCK = threading.Lock()


def power_fetch(cell, tries=12):
    la, lo = cell
    path = RAW / "power" / f"{la:+08.3f}_{lo:+09.4f}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    params = dict(parameters=",".join(POWER_PARAMS), community="RE", longitude=lo, latitude=la, format="JSON")
    js, last = None, None
    for i in range(tries):
        wait = POWER_COOLDOWN[0] - time.time()
        if wait > 0:
            time.sleep(wait + np.random.rand() * 5)
        try:
            r = SESSION.get(POWER, params=params, timeout=120, headers={"Accept": "application/json"})
            if r.status_code == 429:
                with POWER_LOCK:
                    POWER_COOLDOWN[0] = max(POWER_COOLDOWN[0], time.time() + 45)
                raise requests.HTTPError("HTTP 429 (rate limited)")
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** i, 30))
    if js is None:
        raise RuntimeError(f"POWER {cell}: {last}")
    p = js["properties"]["parameter"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p))
    return p


def power_features(p):
    def ann(k):
        v = p.get(k, {}).get("ANN")
        return np.nan if v is None or v <= -998 else float(v)

    def mon(k):
        return [p[k][m] for m in MONTHS if k in p and p[k].get(m, -999) > -998]

    tmin, tmax, t = mon("T2M_MIN"), mon("T2M_MAX"), mon("T2M")
    return dict(
        power_t2m_c=ann("T2M"), power_t2m_min_c=ann("T2M_MIN"), power_t2m_max_c=ann("T2M_MAX"),
        power_t2m_coldest_month_min_c=min(tmin) if tmin else np.nan,
        power_t2m_warmest_month_max_c=max(tmax) if tmax else np.nan,
        power_t2m_annual_range_c=(max(t) - min(t)) if t else np.nan,
        power_t2m_diurnal_range_c=ann("T2M_RANGE"),
        power_frost_days=ann("FROST_DAYS"), power_rh2m_pct=ann("RH2M"), power_ws2m_ms=ann("WS2M"),
        power_ps_kpa=ann("PS"), power_precip_mm_yr=ann("PRECTOTCORR") * 365.25,
        power_sw_allsky_kwh_m2_d=ann("ALLSKY_SFC_SW_DWN"), power_sw_clrsky_kwh_m2_d=ann("CLRSKY_SFC_SW_DWN"),
        power_uv_index=ann("ALLSKY_SFC_UV_INDEX"), power_albedo=ann("ALLSKY_SRF_ALB"),
    )


# ============================================================================= Macrostrat
MSTRAT = "https://macrostrat.org/api/v2/geologic_units/map"
SCALE_RANK = {"large": 0, "medium": 1, "small": 2, "tiny": 3}


def mstrat_fetch(key):
    lat, lon = key
    path = RAW / "macrostrat" / f"{lat:+09.4f}_{lon:+010.4f}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    js = get_json(MSTRAT, params=dict(lat=round(lat, 5), lng=round(((lon + 180) % 360) - 180, 5)), tries=5)
    d = js.get("success", {}).get("data", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    return d


def mstrat_features(d):
    if not d:
        return dict(mstrat_unit="", mstrat_lith="", mstrat_age="", mstrat_descrip="", mstrat_scale="",
                    mstrat_t_age_ma=np.nan, mstrat_b_age_ma=np.nan, mstrat_text_all="")
    d = sorted(d, key=lambda u: SCALE_RANK.get(u.get("scale"), 9))  # finest map scale first
    u = d[0]

    def s(x, k):
        return (x.get(k) or "").strip()

    def first(k):
        return next((s(x, k) for x in d if s(x, k)), "")

    desc = ""
    for x in d:
        desc = " | ".join(v for v in (s(x, "descrip"), s(x, "comments")) if v)
        if desc:
            break
    b, t = s(u, "b_int_name"), s(u, "t_int_name")
    age = b if b == t else f"{b}-{t}".strip("-")
    seen, parts = set(), []
    for x in d:
        for k in ("name", "strat_name", "lith", "descrip", "comments", "best_int_name"):
            v = s(x, k)
            if v and v.lower() not in seen:
                seen.add(v.lower())
                parts.append(v)
    return dict(mstrat_unit=s(u, "name") or first("name"), mstrat_lith=s(u, "lith") or first("lith"),
                mstrat_age=age or s(u, "best_int_name"), mstrat_descrip=desc, mstrat_scale=u.get("scale", ""),
                mstrat_t_age_ma=float(u["t_age"]) if u.get("t_age") is not None else np.nan,
                mstrat_b_age_ma=float(u["b_age"]) if u.get("b_age") is not None else np.nan,
                mstrat_text_all="; ".join(parts)[:1500])


# ============================================================================= background API runner
class ApiStep(threading.Thread):
    """Fetch unique keys (priority-ordered) with N workers; results land in self.res as they arrive."""

    def __init__(self, name, keys, fetch, feat, workers, cap_s, on_progress):
        super().__init__(daemon=True)
        self.name_, self.keys, self.fetch, self.feat = name, keys, fetch, feat
        self.workers, self.cap_s, self.on_progress = workers, cap_s, on_progress
        self.res, self.failed, self.skipped = {}, 0, 0
        self.lock = threading.Lock()
        self.finished = False

    def run(self):
        deadline = time.time() + self.cap_s
        pr = Progress(self.name_, len(self.keys))
        last_flush = [time.time()]
        log(f"{self.name_}: {len(self.keys)} unique requests, {self.workers} workers, cap {self.cap_s / 60:.0f} min")

        def one(k):
            if time.time() > deadline:
                with self.lock:
                    self.skipped += 1
                return
            try:
                f = self.feat(self.fetch(k))
                with self.lock:
                    self.res[k] = f
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.failed += 1
                    if self.failed <= 5:
                        log(f"  {self.name_} error: {str(e)[:150]}")
            pr.tick()
            if time.time() - last_flush[0] > 60:
                last_flush[0] = time.time()
                self.on_progress()

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(one, self.keys))
        self.finished = True
        log(f"{self.name_}: done {len(self.res)}/{len(self.keys)} (failed {self.failed}, capped {self.skipped})")

    def status(self):
        return dict(done=len(self.res), total=len(self.keys), failed=self.failed, capped=self.skipped,
                    finished=self.finished)


# ============================================================================= output assembly
FINAL_COLUMNS = [
    "patch_id", "lat", "lon",
    # temperature & environmental extremes
    "lst_day_max_k", "lst_day_mean_k", "lst_night_min_k", "lst_night_mean_k", "lst_mean_k",
    "lst_diurnal_range_k", "lst_annual_range_k", "lst_day_months_valid", "lst_night_months_valid", "lst_res_deg",
    "power_t2m_c", "power_t2m_min_c", "power_t2m_max_c", "power_t2m_coldest_month_min_c",
    "power_t2m_warmest_month_max_c", "power_t2m_annual_range_c", "power_t2m_diurnal_range_c", "power_frost_days",
    "power_rh2m_pct", "power_ws2m_ms", "power_ps_kpa", "power_precip_mm_yr", "power_sw_allsky_kwh_m2_d",
    "power_sw_clrsky_kwh_m2_d", "power_uv_index", "power_albedo", "power_cell_lat", "power_cell_lon",
    # water / ice / permafrost
    "pf_frac_pct", "pf_gst_c", "pf_magt_2m_c", "pf_magt_10m_c", "pf_in_cci_domain",
    "permafrost_proxy", "permafrost_flag", "permafrost_src",
    "soil_moist_mean", "soil_moist_max", "soil_moist_months_valid", "soil_moist_widened_25km",
    "precip_mm_yr", "precip_wettest_month_mm", "precip_widened_25km", "precip_best_mm_yr", "precip_best_src",
    # biology / habitability
    "ndvi_mean", "ndvi_max", "ndvi_res_deg",
    # dust / aerosol
    "aot_mean", "aot_max", "aot_widened_25km",
    # geology / surface materials
    "sg_sand_pct", "sg_silt_pct", "sg_clay_pct", "sg_bdod_g_cm3", "sg_cfvo_pct", "sg_res_m",
    "mstrat_unit", "mstrat_lith", "mstrat_age", "mstrat_descrip", "mstrat_scale", "mstrat_t_age_ma",
    "mstrat_b_age_ma", "mstrat_text_all",
]
TEXT_COLS = {"permafrost_src", "precip_best_src", "mstrat_unit", "mstrat_lith", "mstrat_age", "mstrat_descrip",
             "mstrat_scale", "mstrat_text_all", "patch_id"}


class Output:
    def __init__(self, pts, path, n_input):
        self.pts, self.path, self.n_input = pts, Path(path), n_input
        self.parts, self.status = {}, {}
        self.api = {}
        self.lock = threading.Lock()
        self.ready = False  # periodic API flushes only after the raster columns exist (no regressions on re-runs)

    def set(self, name, df, status=None):
        with self.lock:
            self.parts[name] = df
            if status is not None:
                self.status[name] = status

    def assemble(self):
        pts = self.pts
        out = pts[["patch_id", "lat", "lon"]].copy()
        for name, df in list(self.parts.items()):
            for c in df.columns:
                out[c] = df[c].values
        pw = self.api.get("power")
        if pw is not None:
            cells = [power_cell(a, b) for a, b in zip(pts["lat"], pts["lon"])]
            with pw.lock:
                res = dict(pw.res)
            keys = np.array(list(res.keys()), dtype=float).reshape(-1, 2)
            used = []
            for c in cells:  # own cell if fetched, else nearest fetched cell within 1 deg (see power_cell_lat/lon)
                if c in res:
                    used.append(c)
                elif len(keys):
                    dd = np.hypot(keys[:, 0] - c[0], (keys[:, 1] - c[1] + 180) % 360 - 180)
                    j = int(dd.argmin())
                    used.append((float(keys[j, 0]), float(keys[j, 1])) if dd[j] <= 1.0 else None)
                else:
                    used.append(None)
            pdf = pd.DataFrame([res[u] if u is not None else {} for u in used], index=out.index)
            for c in pdf.columns:
                out[c] = pdf[c]
            out["power_cell_lat"] = [u[0] if u is not None else np.nan for u in used]
            out["power_cell_lon"] = [u[1] if u is not None else np.nan for u in used]
        ms = self.api.get("macrostrat")
        if ms is not None:
            keys = list(zip(pts["lat"].round(5), pts["lon"].round(5)))
            with ms.lock:
                rows = [ms.res.get(k, {}) for k in keys]
            mdf = pd.DataFrame(rows, index=out.index)
            for c in mdf.columns:
                out[c] = mdf[c]
        out = out.reindex(columns=FINAL_COLUMNS)
        # derived: permafrost flag (CCI PFR where available, else POWER mean-annual T2M < -2 C proxy)
        t2m = out["power_t2m_c"].astype(float)
        out["permafrost_proxy"] = np.where(t2m.notna(), (t2m < -2.0).astype(float), np.nan)
        cci = out["pf_frac_pct"].astype(float)
        out["permafrost_flag"] = np.where(cci.notna(), (cci >= 50).astype(float), out["permafrost_proxy"])
        out["permafrost_src"] = np.where(cci.notna(), "esa_cci_pfr_2023",
                                         np.where(t2m.notna(), "proxy_power_t2m_lt_-2C", ""))
        g, pp = out["precip_mm_yr"].astype(float), out["power_precip_mm_yr"].astype(float)
        out["precip_best_mm_yr"] = g.where(g.notna(), pp)
        out["precip_best_src"] = np.where(g.notna(), "gsmap", np.where(pp.notna(), "power_merra2", ""))
        for c in out.columns:
            if c in TEXT_COLS or c in ("lat", "lon"):
                continue
            if out[c].dtype == bool:
                out[c] = out[c].astype(int)
            elif out[c].dtype != object:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
        return out

    def write(self, phase, force=True):
        if not (force or self.ready):
            return None
        with self.lock:
            out = self.assemble()
            tmp = self.path.with_suffix(".csv.tmp")
            out.to_csv(tmp, index=False)
            for i in range(10):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    time.sleep(1)
            cov = {}
            for c in out.columns:
                if out[c].dtype == object:
                    cov[c] = round(float((out[c].fillna("").astype(str).str.len() > 0).mean() * 100), 1)
                else:
                    cov[c] = round(float(out[c].notna().mean() * 100), 1)
            st = dict(written_at=time.strftime("%Y-%m-%d %H:%M:%S"), phase=phase, rows=len(out),
                      elapsed_s=round(time.time() - T0), sources=dict(self.status),
                      api={k: v.status() for k, v in self.api.items()}, coverage_pct=cov)
            self.path.with_suffix(".status.json").write_text(json.dumps(st, indent=1, default=str))
            log(f"WROTE {self.path.name} [{phase}] rows={len(out)}; "
                + ", ".join(f"{k}={v.status()['done']}/{v.status()['total']}" for k, v in self.api.items()))
            return out


# ============================================================================= main
PRIORITY_SOURCES = ("named_site", "impact_structure")


def load_points(args):
    if args.sites:
        df = pd.read_csv(ROOT / "config" / "earth_sites.csv")
        df = pd.DataFrame(dict(patch_id=df["id"], lat=df["lat"].astype(float), lon=df["lon"].astype(float),
                               source="named_site"))
    else:
        df = pd.read_csv(args.input)
        if "source" not in df:
            df["source"] = ""
    df["patch_id"] = df["patch_id"].astype(str)
    return df[["patch_id", "lat", "lon", "source"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", action="store_true", help="run on config/earth_sites.csv (test)")
    ap.add_argument("--input", default=str(ROOT / "data" / "patches" / "earth_candidates.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default="jaxa,soil,permafrost,power,macrostrat")
    ap.add_argument("--fast-only", action="store_true", help="skip phase B (full-resolution rasters)")
    ap.add_argument("--threads", type=int, default=16, help="JAXA download threads")
    ap.add_argument("--power-workers", type=int, default=8)
    ap.add_argument("--mstrat-workers", type=int, default=8)
    ap.add_argument("--cap-min", type=float, default=15.0, help="wall-time cap per slow source (minutes)")
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else (
        RAW / "steps" / "earth_sites_env.csv" if args.sites else ROOT / "data" / "patches" / "earth_patches_env.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = load_points(args)
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    log(f"{len(pts)} points -> {out_path}; steps {sorted(steps)}")
    O = Output(pts, out_path, len(pts))

    # priority order for the point APIs: named sites + impact structures first
    prio = np.argsort(~pts["source"].isin(PRIORITY_SOURCES).to_numpy(), kind="stable")
    if "power" in steps:
        cells = list(dict.fromkeys(power_cell(pts.lat[i], pts.lon[i]) for i in prio))
        O.api["power"] = ApiStep("POWER", cells, power_fetch, power_features, args.power_workers,
                                 args.cap_min * 60, lambda: O.write("partial (APIs running)", force=False))
    if "macrostrat" in steps:
        keys = list(dict.fromkeys((round(pts.lat[i], 5), round(pts.lon[i], 5)) for i in prio))
        O.api["macrostrat"] = ApiStep("Macrostrat", keys, mstrat_fetch, mstrat_features, args.mstrat_workers,
                                      args.cap_min * 60, lambda: O.write("partial (APIs running)", force=False))
    for t in O.api.values():
        t.start()

    def run(name, fn, *a):
        t = time.time()
        try:
            r = fn(pts, args, *a)
            log(f"== {name} done in {time.time() - t:.0f}s")
            return r
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"== {name} FAILED: {e}")
            return None

    # ---- phase A: fast rasters (skipped when all full-resolution inputs are already cached)
    full_cached = all(jaxa_complete(p, lv) for p, lv in LEVELS_FULL.items()) and pf_path("GTD", "nh").exists()
    if full_cached:
        log("all full-resolution rasters cached -> skipping the fast pass")
        steps_a = set()
    else:
        steps_a = steps
    if "jaxa" in steps_a:
        r = run("JAXA fast", step_jaxa, True)
        if r:
            O.set("jaxa", r[0], dict(jaxa=r[1]))
    if "soil" in steps_a:
        r = run("SoilGrids 5 km", step_soil, 5000)
        if r is not None:
            O.set("soil", r, dict(soilgrids_res_m=5000))
    if "permafrost" in steps:
        r = run("Permafrost", step_permafrost)
        if r:
            O.set("pf", r[0], dict(permafrost_files=r[1]))
    if steps_a:
        O.write("phase A: fast rasters (0.2 deg JAXA, 5 km SoilGrids)")
        O.ready = True

    # ---- phase B: full-resolution rasters
    if not args.fast_only:
        if "jaxa" in steps:
            r = run("JAXA full", step_jaxa, False)
            if r:
                O.set("jaxa", r[0], dict(jaxa=r[1]))
                O.write("phase B: JAXA full resolution", force=O.ready)
        if "permafrost" in steps and not pf_path("GTD", "nh").exists():
            if wait_for([pf_path("GTD", "nh")], args.cap_min * 60, "CCI GTD northern hemisphere"):
                r = run("Permafrost (with GTD NH)", step_permafrost)
                if r:
                    O.set("pf", r[0], dict(permafrost_files=r[1]))
                    O.write("phase B: permafrost ground temperature", force=O.ready)
        if "soil" in steps and "soil" not in O.parts:  # fast pass skipped: best soil available right now
            have_1km = all(sg_path(p, 1000).exists() for p in SG_PROPS)
            r = run("SoilGrids", step_soil, 1000 if have_1km else 5000)
            if r is not None:
                O.set("soil", r, dict(soilgrids_res_m=1000 if have_1km else 5000))
        if "soil" in steps and O.status.get("soil", {}).get("soilgrids_res_m") != 1000:
            if wait_for([sg_path(p, 1000) for p in SG_PROPS], args.cap_min * 60, "SoilGrids 1 km"):
                r = run("SoilGrids 1 km", step_soil, 1000)
                if r is not None:
                    O.set("soil", r, dict(soilgrids_res_m=1000))
                    O.write("phase B: SoilGrids 1 km", force=O.ready)
    O.write("rasters done (APIs may still be running)")
    O.ready = True

    for t in O.api.values():
        t.join()
    out = O.write("final")
    n_now = len(pd.read_csv(args.input, usecols=["patch_id"])) if not args.sites else len(pts)
    if n_now != len(pts):
        log(f"WARNING: input now has {n_now} rows (ran on {len(pts)}) -> re-run to refresh (caches make it cheap)")
    cov = json.loads(out_path.with_suffix(".status.json").read_text())["coverage_pct"]
    print(pd.Series(cov).to_string())


if __name__ == "__main__":
    main()
