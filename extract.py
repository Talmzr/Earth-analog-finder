#!/usr/bin/env python
"""
Extract comparable features for lunar/Martian target sites and Earth analog sites.

Categories
  terrain       DEM windows -> relief, slope at a common 500 m baseline, roughness (TRI)
                  Moon : NASA PGDA LOLA south-pole DEM, 80 m/px COG
                  Mars : USGS MOLA 128 ppd COG (~463 m/px)
                  Earth: Copernicus GLO-30 DEM COGs on AWS (30 m)
  environment   Earth only: MODIS NDVI (MOD13Q1), MODIS LST day/night (MOD11A2) via ORNL DAAC API,
                  ERA5 temperature + precipitation via Open-Meteo archive API
  illumination  analytic solar geometry from latitude + body obliquity (no download)
  geology       curated tags in config/*.csv (no download)

Usage
  python extract.py                 # everything
  python extract.py --skip-env      # terrain + illumination only
  python extract.py --only haughton,lonar,moon_haworth
Results -> data/target_features.csv, data/earth_features.csv  (per-site JSON cache in data/cache/)
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "config"
DATA = ROOT / "data"
CACHE = DATA / "cache"

# ----------------------------------------------------------------------------- data sources
MOON_DEM_URL = os.environ.get(
    "MOON_DEM_URL", "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF")
MARS_DEM_URL = os.environ.get(
    "MARS_DEM_URL",
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/wms_basemaps3/Mars/MOLA/"
    "mola128_mola64_merge_90Nto90S_SimpleC_clon0_cog.tif")
EARTH_DEM_TEMPLATE = os.environ.get(
    "EARTH_DEM_TEMPLATE",
    "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
MODIS_API = os.environ.get("MODIS_API", "https://modis.ornl.gov/rst/api/v1")
OPENMETEO_API = os.environ.get("OPENMETEO_API", "https://archive-api.open-meteo.com/v1/archive")

BODY = {
    #        sphere radius (m)  obliquity (deg) used for solar declination
    "earth": dict(R=6371008.8, obliquity=23.44),
    "moon":  dict(R=1737400.0, obliquity=1.54),
    "mars":  dict(R=3396190.0, obliquity=25.19),
}
SLOPE_BASELINE_M = 500.0   # common scale so 30 m, 80 m and 463 m DEMs are comparable
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)

warnings.filterwarnings("ignore", message="Mean of empty slice")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "space-apps-2026-analog-finder"


def http_get_json(url, params=None, tries=4, timeout=60):
    """GET with retries/backoff. Separate function so tests can monkeypatch it."""
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout,
                            headers={"Accept": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET failed {url}: {last}")


# ----------------------------------------------------------------------------- geometry helpers
def haversine(lon1, lat1, lon2, lat2, R):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def norm_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


# ----------------------------------------------------------------------------- terrain
def _geo_crs(body):
    from rasterio.crs import CRS
    return CRS.from_proj4(f"+proj=longlat +R={BODY[body]['R']} +no_defs")


def _pixel_ground_spacing(src, row, col, body):
    """Ground distance (m) of one pixel step in x and y at (row, col), any projection."""
    from rasterio.warp import transform
    xs, ys = [], []
    for dr, dc in ((0, 0), (0, 1), (1, 0)):
        x, y = src.xy(row + dr, col + dc)
        xs.append(x)
        ys.append(y)
    if src.crs.is_geographic:
        lons, lats = xs, ys
    else:
        lons, lats = transform(src.crs, _geo_crs(body), xs, ys)
    R = BODY[body]["R"]
    dx = float(haversine(lons[0], lats[0], lons[1], lats[1], R))
    dy = float(haversine(lons[0], lats[0], lons[2], lats[2], R))
    return dx, dy


def _read_window_single(url, body, lat, lon, hw_km):
    """Read a square window (half-width hw_km) around lat/lon from one remote raster."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds

    with rasterio.Env(**GDAL_ENV):
        path = url if url.startswith("/") or os.path.exists(url) else "/vsicurl/" + url
        with rasterio.open(path) as src:
            lon_n = norm_lon(lon)
            if src.crs is None or src.crs.is_geographic:
                cx, cy = lon_n, lat
                R = BODY[body]["R"]
                dlat = math.degrees(hw_km * 1000 / R)
                dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
                bounds = (cx - dlon, cy - dlat, cx + dlon, cy + dlat)
            else:
                xs, ys = transform(_geo_crs(body), src.crs, [lon_n], [lat])
                cx, cy = xs[0], ys[0]
                # projected units assumed metres (true for both planetary products used here)
                h = hw_km * 1000.0
                bounds = (cx - h, cy - h, cx + h, cy + h)
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=np.nan,
                           out_dtype="float64", masked=False)
            nod = src.nodata
            if nod is not None and not (isinstance(nod, float) and math.isnan(nod)):
                arr[arr == nod] = np.nan
            scale = (src.scales or (1.0,))[0] or 1.0
            offset = (src.offsets or (0.0,))[0] or 0.0
            arr = arr * scale + offset
            r0 = int(win.row_off + win.height // 2)
            c0 = int(win.col_off + win.width // 2)
            dx, dy = _pixel_ground_spacing(src, r0, c0, body)
    return arr, dx, dy


def _copernicus_tile_name(lat_floor, lon_floor):
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def _read_earth_window(lat, lon, hw_km):
    """Mosaic Copernicus GLO-30 tiles covering the window (ocean tiles are simply absent)."""
    import rasterio
    from rasterio.merge import merge

    R = BODY["earth"]["R"]
    dlat = math.degrees(hw_km * 1000 / R)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    w, s, e, n = lon - dlon, lat - dlat, lon + dlon, lat + dlat
    srcs, open_errors = [], []
    with rasterio.Env(**GDAL_ENV):
        for la in range(math.floor(s), math.floor(n) + 1):
            for lo in range(math.floor(w), math.floor(e) + 1):
                name = _copernicus_tile_name(la, ((lo + 180) % 360) - 180)
                url = EARTH_DEM_TEMPLATE.format(name=name)
                path = url if os.path.exists(url) else "/vsicurl/" + url
                try:
                    srcs.append(rasterio.open(path))
                except Exception as ex:  # noqa: BLE001  (missing tile = ocean, or network problem)
                    open_errors.append(f"{name}: {str(ex)[:120]}")
        if not srcs:
            raise RuntimeError("no Copernicus tile could be opened - ocean, or network blocked? "
                               + "; ".join(open_errors[:2]))
        try:
            mosaic, tr = merge(srcs, bounds=(w, s, e, n), nodata=np.nan, dtype="float64")
            arr = mosaic[0]
            # pixel spacing at window centre
            resx, resy = tr.a, -tr.e
            dx = float(haversine(lon, lat, lon + resx, lat, R))
            dy = float(haversine(lon, lat, lon, lat + resy, R))
        finally:
            for sds in srcs:
                sds.close()
    return arr, dx, dy


def _block_mean(a, fy, fx):
    if fx <= 1 and fy <= 1:
        return a
    H, W = (a.shape[0] // fy) * fy, (a.shape[1] // fx) * fx
    a = a[:H, :W].reshape(H // fy, fy, W // fx, fx)
    with np.errstate(all="ignore"):
        cnt = np.sum(~np.isnan(a), axis=(1, 3))
        m = np.nanmean(a, axis=(1, 3)) if np.any(cnt) else np.full((H // fy, W // fx), np.nan)
    m[cnt < 0.5 * fy * fx] = np.nan
    return m


def _slope_deg(z, dx, dy):
    gy, gx = np.gradient(z, dy, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _tri(z):
    """Terrain Ruggedness Index: mean |z - neighbour| over 8 neighbours."""
    p = np.pad(z, 1, constant_values=np.nan)
    diffs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = p[1 + dr:1 + dr + z.shape[0], 1 + dc:1 + dc + z.shape[1]]
            diffs.append(np.abs(z - nb))
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(diffs), axis=0)


def terrain_metrics(z, dx, dy):
    valid = np.isfinite(z)
    if valid.mean() < 0.5:
        raise RuntimeError(f"window only {valid.mean():.0%} valid")
    zv = z[valid]
    out = dict(
        native_res_m=round((dx + dy) / 2, 1),
        valid_frac=round(float(valid.mean()), 3),
        elev_mean_m=float(np.mean(zv)),
        elev_std_m=float(np.std(zv)),
        relief_m=float(np.percentile(zv, 95) - np.percentile(zv, 5)),
    )
    s_nat = _slope_deg(z, dx, dy)
    out["slope_native_mean_deg"] = float(np.nanmean(s_nat))
    fx = max(1, int(round(SLOPE_BASELINE_M / dx)))
    fy = max(1, int(round(SLOPE_BASELINE_M / dy)))
    z5 = _block_mean(z, fy, fx)
    if min(z5.shape) < 3:
        raise RuntimeError("window too small for 500 m baseline")
    s5 = _slope_deg(z5, dx * fx, dy * fy)
    s5v = s5[np.isfinite(s5)]
    out.update(
        slope500_mean_deg=float(np.mean(s5v)),
        slope500_p90_deg=float(np.percentile(s5v, 90)),
        frac_slope500_gt10=float(np.mean(s5v > 10)),
        frac_slope500_gt20=float(np.mean(s5v > 20)),
        tri500_mean_m=float(np.nanmean(_tri(z5))),
    )
    return out


def extract_terrain(body, lat, lon, hw_km):
    if body == "moon":
        if lat > -80:
            raise RuntimeError("outside LOLA 80S polar DEM coverage")
        z, dx, dy = _read_window_single(MOON_DEM_URL, "moon", lat, lon, hw_km)
    elif body == "mars":
        z, dx, dy = _read_window_single(MARS_DEM_URL, "mars", lat, lon, hw_km)
    else:
        z, dx, dy = _read_earth_window(lat, lon, hw_km)
    return terrain_metrics(z, dx, dy)


# ----------------------------------------------------------------------------- illumination
def illumination(lat, body, n_days=360, n_hours=96):
    """Solar geometry from latitude + obliquity (no topography, no refraction).
    Each 'year' is the body's own year; for the Moon the declination cycle is ~1 Earth year."""
    eps = math.radians(BODY[body]["obliquity"])
    phi = math.radians(lat)
    t = np.arange(n_days) / n_days
    dec = np.arcsin(np.sin(eps) * np.sin(2 * np.pi * t))[:, None]
    h = np.linspace(-np.pi, np.pi, n_hours, endpoint=False)[None, :]
    elev = np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)))
    noon = elev.max(axis=1)
    low = elev.min(axis=1)
    day = elev[elev > 0]
    out = dict(
        noon_elev_max_deg=float(noon.max()),
        noon_elev_min_deg=float(noon.min()),
        noon_elev_mean_deg=float(noon.mean()),
        frac_polar_day=float(np.mean(low > 0)),
        frac_polar_night=float(np.mean(noon < 0)),
        mean_day_elev_deg=float(day.mean()) if day.size else 0.0,
        frac_daylight=float(np.mean(elev > 0)),
    )
    if body == "earth":  # calendar months for 'light twin' month matching
        doy = np.arange(365)
        d = np.degrees(np.arcsin(np.sin(eps) * np.sin(2 * np.pi * (doy - 80) / 365.0)))
        noon_e = 90 - np.abs(lat - d)
        month = np.minimum((doy / 365.0 * 12).astype(int), 11)
        for m in range(12):
            out[f"noon_elev_m{m + 1:02d}"] = float(noon_e[month == m].mean())
    return out


# ----------------------------------------------------------------------------- environment (Earth)
def _modis_annual(product, bands, lat, lon, year, km):
    dates = http_get_json(f"{MODIS_API}/{product}/dates",
                          params=dict(latitude=lat, longitude=lon))["dates"]
    ds = [d["modis_date"] for d in dates if d["modis_date"].startswith(f"A{year}")]
    if not ds:
        raise RuntimeError(f"{product}: no dates for {year}")
    res = {b: [] for b in bands}
    for b in bands:
        for i in range(0, len(ds), 10):  # API allows max 10 dates per request
            chunk = ds[i:i + 10]
            js = http_get_json(f"{MODIS_API}/{product}/subset", params=dict(
                latitude=lat, longitude=lon, band=b, startDate=chunk[0], endDate=chunk[-1],
                kmAboveBelow=km, kmLeftRight=km))
            scale = float(js.get("scale") or 1)
            for rec in js.get("subset", []):
                v = np.asarray(rec["data"], dtype=float)
                res[b].append((v, scale))
    return res


def extract_environment(lat, lon, year):
    out, errs = {}, []
    # NDVI (MOD13Q1, 250 m, 16-day). fill = -3000, valid -2000..10000
    try:
        r = _modis_annual("MOD13Q1", ["250m_16_days_NDVI"], lat, lon, year, km=2)
        meds = []
        for v, sc in r["250m_16_days_NDVI"]:
            v = v[(v >= -2000) & (v <= 10000)]
            if v.size:
                meds.append(np.median(v) * sc)
        out["ndvi_mean"] = float(np.mean(meds)) if meds else np.nan
        out["ndvi_max"] = float(np.max(meds)) if meds else np.nan
    except Exception as e:  # noqa: BLE001
        errs.append(f"ndvi: {e}")
    # LST (MOD11A2, 1 km, 8-day). fill = 0, scale 0.02 K
    try:
        r = _modis_annual("MOD11A2", ["LST_Day_1km", "LST_Night_1km"], lat, lon, year, km=1)
        agg = {}
        for b in ("LST_Day_1km", "LST_Night_1km"):
            meds = [np.median(v[v > 0]) * sc for v, sc in r[b] if np.any(v > 0)]
            agg[b] = float(np.mean(meds)) if meds else np.nan
        out["lst_day_k"] = agg["LST_Day_1km"]
        out["lst_night_k"] = agg["LST_Night_1km"]
        out["lst_day_night_range_k"] = agg["LST_Day_1km"] - agg["LST_Night_1km"]
    except Exception as e:  # noqa: BLE001
        errs.append(f"lst: {e}")
    # ERA5 via Open-Meteo, 10 years
    try:
        js = http_get_json(OPENMETEO_API, params=dict(
            latitude=lat, longitude=lon, start_date=f"{year - 9}-01-01", end_date=f"{year}-12-31",
            daily="temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,"
                  "wind_speed_10m_max,shortwave_radiation_sum",
            timezone="UTC"))
        d = pd.DataFrame(js["daily"])
        d["time"] = pd.to_datetime(d["time"])
        yr = d.groupby(d["time"].dt.year)
        out.update(
            t_mean_c=float(d["temperature_2m_mean"].mean()),
            t_min_annual_c=float(yr["temperature_2m_min"].min().mean()),
            t_max_annual_c=float(yr["temperature_2m_max"].max().mean()),
            t_diurnal_range_c=float((d["temperature_2m_max"] - d["temperature_2m_min"]).mean()),
            precip_mm_yr=float(yr["precipitation_sum"].sum().mean()),
            wind_max_mean_kmh=float(d["wind_speed_10m_max"].mean()),
            sw_rad_mj_m2_day=float(d["shortwave_radiation_sum"].mean()),
        )
    except Exception as e:  # noqa: BLE001
        errs.append(f"era5: {e}")
    return out, errs


# ----------------------------------------------------------------------------- connectivity check
def _open_meta(url):
    import rasterio
    with rasterio.Env(**GDAL_ENV):
        path = url if os.path.exists(url) else "/vsicurl/" + url
        with rasterio.open(path) as s:
            return f"{s.width}x{s.height}, {s.crs.to_string()[:40] if s.crs else 'no CRS'}"


def check_sources():
    """Quick reachability test of every data source; prints a table and returns True if all OK."""
    ok_all = True
    tests = [
        ("Moon LOLA DEM (PGDA)", lambda: _open_meta(MOON_DEM_URL)),
        ("Mars MOLA DEM (USGS S3)", lambda: _open_meta(MARS_DEM_URL)),
        ("Earth Copernicus GLO-30 (AWS)", lambda: _open_meta(EARTH_DEM_TEMPLATE.format(
            name=_copernicus_tile_name(19, 76)))),
        ("MODIS ORNL API", lambda: f"{len(http_get_json(f'{MODIS_API}/MOD13Q1/dates', params=dict(latitude=19.98, longitude=76.51), tries=1)['dates'])} dates"),
        ("Open-Meteo ERA5 API", lambda: f"{len(http_get_json(OPENMETEO_API, params=dict(latitude=19.98, longitude=76.51, start_date='2024-01-01', end_date='2024-01-03', daily='temperature_2m_mean'), tries=1)['daily']['time'])} days"),
    ]

    for name, fn in tests:
        try:
            msg, flag = fn(), "OK  "
        except Exception as e:  # noqa: BLE001
            msg, flag, ok_all = str(e)[:150], "FAIL", False
        print(f"  [{flag}] {name:<32} {msg}")
    return ok_all



# ----------------------------------------------------------------------------- driver
def process_site(row, kind, args):
    sid = row["id"]
    cache_file = CACHE / f"{kind}_{sid}.json"
    cached = json.loads(cache_file.read_text()) if cache_file.exists() and not args.refresh else {}
    rec = dict(cached)
    rec.update(id=sid, name=row["name"], lat=row["lat"], lon=row["lon"])
    body = row["body"] if kind == "target" else "earth"
    rec["body"] = body
    errors = []

    if not args.skip_terrain and "relief_m" not in rec:
        try:
            rec.update(extract_terrain(body, float(row["lat"]), float(row["lon"]),
                                       float(row["half_width_km"])))
        except Exception as e:  # noqa: BLE001
            errors.append(f"terrain: {e}")
            if args.verbose:
                traceback.print_exc()

    rec.update(illumination(float(row["lat"]), body))

    if kind == "earth" and not args.skip_env and "t_mean_c" not in rec:
        env, errs = extract_environment(float(row["lat"]), float(row["lon"]), args.year)
        rec.update(env)
        errors += errs

    rec["errors"] = " | ".join(errors)
    cache_file.write_text(json.dumps({k: v for k, v in rec.items() if k != "errors"}, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-terrain", action="store_true")
    ap.add_argument("--skip-env", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated site ids")
    ap.add_argument("--year", type=int, default=2024, help="MODIS year; ERA5 uses the 10 years ending here")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true", help="ignore cache")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--check", action="store_true", help="only test that every data source is reachable")
    args = ap.parse_args()
    if args.check:
        print("Checking data sources...")
        return 0 if check_sources() else 1

    CACHE.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(CFG / "targets.csv")
    earth = pd.read_csv(CFG / "earth_sites.csv")
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    for kind, df, outname in (("target", targets, "target_features.csv"),
                              ("earth", earth, "earth_features.csv")):
        rows = [r for _, r in df.iterrows() if not only or r["id"] in only]
        if not rows:
            continue
        print(f"\n== {kind}: {len(rows)} sites ==")
        recs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_site, r, kind, args): r["id"] for r in rows}
            for f in as_completed(futs):
                rec = f.result()
                flag = "OK " if not rec["errors"] else "ERR"
                print(f"  [{flag}] {rec['id']:<28} {rec['errors'][:110]}")
                recs.append(rec)
        out = pd.DataFrame(recs)
        # merge with any previously extracted sites not in this run
        path = DATA / outname
        if path.exists() and only:
            old = pd.read_csv(path)
            old = old[~old["id"].isin(out["id"])]
            out = pd.concat([old, out], ignore_index=True)
        order = {sid: i for i, sid in enumerate(df["id"])}
        out = out.sort_values("id", key=lambda s: s.map(order))
        out.to_csv(path, index=False)
        print(f"  -> {path}")
    print("\nDone. Next: python compare.py")


if __name__ == "__main__":
    sys.exit(main())
