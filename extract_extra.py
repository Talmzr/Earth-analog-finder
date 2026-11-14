#!/usr/bin/env python
"""
Second-pass extraction: every extra NASA (and partner-agency) layer that adds information
beyond extract.py. Writes data/target_extra.csv and data/earth_extra.csv.
Column meanings, sources and skipped datasets: DATA_SOURCES.md.

MOON  (NASA LRO / PDS / PGDA)
  diviner_polar   Diviner Polar Resource Product (south): avg + max surface temperature,
                  ice-stability depth on ~240 m triangles                    lat <= -76
  diviner_gcp     Diviner Global Cumulative Product: bolometric temperature for every local
                  time on a 0.5 deg grid -> daily max / min / mean           all Moon
  diviner_rock    Diviner rock abundance + regolith temperature anomaly      |lat| <= 60
  lola_illum      LOLA average solar visibility (Mazarico et al.), incl. topographic shadow,
                  permanently-shadowed fraction                              lat <= -75
  lola_albedo     LOLA 1064 nm normal albedo, 10 px/deg                      all Moon
  lola_terrain    LOLA global 118 m DEM -> same terrain metrics as extract.py, for sites
                  outside the 80S polar DEM (Tranquillitatis pit)            lat > -80
MARS  (NASA MGS / Odyssey / PDS / JPL-PSI SWIM / USGS)
  tes_ti          TES day + night thermal inertia (Putzig & Mellon 2007)     all Mars
  tes_albedo      TES bolometric albedo                                      all Mars
  swim            SWIM 2.0 subsurface-ice consistency (<1 m, 1-5 m, >5 m)    |lat| <= 60
  human_ez        NASA 2015 Human Landing Site Exploration Zones: inside? nearest + distance
  mcd             Mars Climate Database v6 (ESA/CNES/LMD, complement): surface temperature
                  over the year and day, surface pressure
EARTH
  power           NASA POWER climatology 2001-2025: air temp, frost days, humidity, wind,
                  pressure, solar + UV, surface albedo
  modis_vcf       MODIS MOD44B vegetation continuous fields (% bare, % tree)
  modis_lc        MODIS MCD12Q1 land-cover class (IGBP)
  worldcover      ESA WorldCover 10 m land-cover fractions (bare, snow/ice, water ...)
  jaxa            JAXA GSMaP precipitation + AMSR2 soil moisture, monthly normals
  macrostrat      Macrostrat bedrock unit / lithology / age at the site (complement)

Usage
  python extract_extra.py --check              # reachability of every source (~1 min)
  python extract_extra.py                      # everything (first run downloads ~0.7 GB once)
  python extract_extra.py --only moon_haworth,haughton --groups diviner_polar,power
Large downloads are kept OUTSIDE the project (default ~/.cache/earth_analog_raw, override
with ANALOG_RAW_DIR) so they do not sync to OneDrive / git.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import extract as base  # reuse window readers, terrain metrics, HTTP helper

ROOT = Path(__file__).resolve().parent
CFG, DATA = ROOT / "config", ROOT / "data"
CACHE = DATA / "cache"
RAW = Path(os.environ.get("ANALOG_RAW_DIR", Path.home() / ".cache" / "earth_analog_raw"))
SESSION = base.SESSION
R_MOON, R_MARS, R_EARTH = base.BODY["moon"]["R"], base.BODY["mars"]["R"], base.BODY["earth"]["R"]

# ----------------------------------------------------------------------------- source registry
PDS_DIV = "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1"
USGS = "https://asc-pds-services.s3.us-west-2.amazonaws.com"
JAXA = "https://s3.ap-northeast-1.wasabisys.com/je-pds/cog/v1"
SRC = {
    "diviner_prp_south": f"{PDS_DIV}/data_derived_prp/dlre_prp_south.tab",
    "diviner_gcp": PDS_DIV + "/data_derived_gcp/global_cumul_avg_cyl_{band}_002.tab",
    "diviner_ra": f"{PDS_DIV}/data_derived_gdr_l3/cylindrical/img/dgdr_ra_avg_cyl_032_img.img",
    "diviner_st": f"{PDS_DIV}/data_derived_gdr_l3/cylindrical/img/dgdr_st_avg_cyl_032_img.img",
    "lola_avgvis_75s": "https://pgda.gsfc.nasa.gov/data/MoonIllumination/AVGVISIB_75S_120M_201608.TIF",
    "lola_albedo": f"{USGS}/mosaic/Lunar_LRO_LOLA_Albedo_Global_10ppd.tif",
    "lola_ldem_118m": f"{USGS}/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
    "tes_ti_day": "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/mgst_9001/data/global_ti_day_2007.img",
    "tes_ti_night": "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/mgst_9001/data/global_ti_night_2007.img",
    "tes_albedo": f"{USGS}/mosaic/Mars_MGS_TES_Albedo_mosaic_global_7410m.tif",
    "swim_c0_1": "https://swim.psi.edu/output/SWIM2/Global/Composite/SWIM2_c0_1.tif",
    "swim_c1_5": "https://swim.psi.edu/output/SWIM2/Global/Composite/SWIM2_c1_5.tif",
    "swim_c_5": "https://swim.psi.edu/output/SWIM2/Global/Composite/SWIM2_c_5.tif",
    "human_ez": f"{USGS}/wms_basemaps3/Mars/Human_EZ/ExplorationZones_Proposed_Oct2015_clon0",
    "mcd": "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py",
    "power": "https://power.larc.nasa.gov/api/temporal/climatology/point",
    "modis_api": base.MODIS_API,
    "worldcover": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
    "jaxa_gsmap": f"{JAXA}/JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly-normal",
    "jaxa_amsr2_sm": f"{JAXA}/JAXA.G-Portal_GCOM-W.AMSR2_standard.L3-SMC.daytime.v3_global_monthly-normal",
    "macrostrat": "https://macrostrat.org/api/v2/geologic_units/map",
}

GROUPS = {
    # lola_albedo exists but the USGS 10 ppd map is mostly zero-filled -> not used
    "moon": ["diviner_polar", "diviner_gcp", "diviner_rock", "lola_illum", "lola_terrain"],
    "mars": ["tes_ti", "tes_albedo", "swim", "human_ez", "mcd"],
    "earth": ["power", "modis_vcf", "modis_lc", "worldcover", "jaxa", "macrostrat"],
}


# ----------------------------------------------------------------------------- generic helpers
def download(url, name=None, min_bytes=1000):
    """Download once into RAW (streamed), return local path."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / (name or url.rsplit("/", 1)[-1])
    if path.exists() and path.stat().st_size >= min_bytes:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    with SESSION.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(path)
    return path


def http_range(url, start, length, tries=4):
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=90)
            if r.status_code == 206 and len(r.content) == length:
                return r.content
            raise RuntimeError(f"HTTP {r.status_code}, {len(r.content)} bytes (wanted {length})")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"range GET failed {url}: {last}")


def stats(prefix, v, pcts=(), extra=None):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    out = {}
    if v.size == 0:
        return out
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_min"] = float(v.min())
    out[f"{prefix}_max"] = float(v.max())
    for p in pcts:
        out[f"{prefix}_p{p}"] = float(np.percentile(v, p))
    return out


def cyl_rows_window(lat, lon, hw_km, R, ppd, lat_top, lon_left, n_lines, n_samples):
    """Line/sample ranges of a simple-cylindrical grid around lat/lon (lon wraps)."""
    dlat = math.degrees(hw_km * 1000 / R)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    l0 = max(0, int(math.floor((lat_top - (lat + dlat)) * ppd)))
    l1 = min(n_lines - 1, int(math.floor((lat_top - (lat - dlat)) * ppd)))
    s_c = ((lon - lon_left) % 360.0) * ppd
    cols = np.arange(int(math.floor(s_c - dlon * ppd)), int(math.floor(s_c + dlon * ppd)) + 1) % n_samples
    return l0, l1, cols


def read_cyl_img(url, lat, lon, hw_km, R, ppd, lat_top, lon_left, n_lines, n_samples, dtype):
    """Range-read the rows of a raw PDS simple-cylindrical image covering the window."""
    item = np.dtype(dtype).itemsize
    rec = n_samples * item
    l0, l1, cols = cyl_rows_window(lat, lon, hw_km, R, ppd, lat_top, lon_left, n_lines, n_samples)
    buf = http_range(url, l0 * rec, (l1 - l0 + 1) * rec)
    arr = np.frombuffer(buf, dtype=dtype).reshape(l1 - l0 + 1, n_samples)
    return arr[:, cols].astype(float)


def read_local_cyl_tif(path, lat, lon, hw_km, R, lon_wrap=True):
    """Window from a global simple-cylindrical GeoTIFF in metres (x = R*lon_rad)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.Env(**base.GDAL_ENV):
        p = str(path) if Path(str(path)).exists() else "/vsicurl/" + str(path)
        with rasterio.open(p) as s:
            dlat = math.degrees(hw_km * 1000 / R)
            dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
            k = math.pi / 180 * R
            lon_n = base.norm_lon(lon)
            # some products use 0..360 longitudes
            if s.bounds.left >= -1 and lon_n < 0:
                lon_n += 360
            b = ((lon_n - dlon) * k, (lat - dlat) * k, (lon_n + dlon) * k, (lat + dlat) * k)
            win = from_bounds(*b, transform=s.transform).round_offsets().round_lengths()
            if win.width < 1 or win.height < 1:
                win = from_bounds(*b, transform=s.transform)
                win = win.__class__(int(win.col_off), int(win.row_off), 1, 1)
            a = s.read(1, window=win, boundless=True, fill_value=np.nan, out_dtype="float64")
            if s.nodata is not None and np.isfinite(s.nodata):
                a[a == s.nodata] = np.nan
            a[a < -1e30] = np.nan
            sc, of = (s.scales or (1.0,))[0] or 1.0, (s.offsets or (0.0,))[0] or 0.0
            return a * sc + of


def gc_dist(lon1, lat1, lon2, lat2, R):
    return base.haversine(lon1, lat1, lon2, lat2, R)


# ============================================================================= MOON
_PRP = None


def _load_prp():
    """Diviner south PRP -> compact npz (lon, lat, tavg, tmax, ice_depth). 605 MB download once."""
    global _PRP
    if _PRP is not None:
        return _PRP
    npz = RAW / "dlre_prp_south.npz"
    if not npz.exists():
        tab = download(SRC["diviner_prp_south"], "dlre_prp_south.tab", min_bytes=600_000_000)
        cols = ["tri_clon", "tri_clat", "temp_avg", "temp_max", "ice_depth"]
        parts = []
        for ch in pd.read_csv(tab, skipinitialspace=True, usecols=cols, chunksize=500_000,
                              dtype="float32"):
            parts.append(ch[cols].to_numpy())
        a = np.vstack(parts)
        np.savez_compressed(npz, lon=a[:, 0], lat=a[:, 1], tavg=a[:, 2], tmax=a[:, 3], ice=a[:, 4])
    z = np.load(npz)
    _PRP = {k: z[k] for k in z.files}
    return _PRP


PRP_ROW, PRP_NCOL, PRP_PER_COL, PRP_X0, PRP_STEP = 210, 1200, 2400, 299.52036, 599.04072 / 1199


def _prp_window(lat, lon, hw):
    """Range-read only the PRP rows around the site. The table is a 1200 x 1200 grid of cell pairs
    (2 triangles each) in Moon-centred x/y km: column = x descending, row = y ascending."""
    X = R_MOON / 1000 * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    Y = R_MOON / 1000 * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    n = int(hw / PRP_STEP * 1.1) + 3
    ic0, iy0 = int(round((PRP_X0 - X) / PRP_STEP)), int(round((Y + PRP_X0) / PRP_STEP))
    cols = [c for c in range(ic0 - n, ic0 + n + 1) if 0 <= c < PRP_NCOL]
    y_lo, y_hi = max(0, iy0 - n), min(PRP_NCOL - 1, iy0 + n)

    def one(c):
        first = c * PRP_PER_COL + 2 * y_lo
        cnt = 2 * (y_hi - y_lo + 1)
        buf = http_range(SRC["diviner_prp_south"], (first + 1) * PRP_ROW, cnt * PRP_ROW)  # +1: header row
        return [ln.split(",")[9:] for ln in buf.decode("ascii", "replace").splitlines() if ln.count(",") == 14]

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [r for part in ex.map(one, cols) for r in part]
    return np.array(rows, dtype=float)   # clon, clat, calt, temp_avg, temp_max, ice_depth


def g_diviner_polar(lat, lon, hw):
    if lat > -76:
        return {}, "outside Diviner south PRP (<= -76)"
    a = _prp_window(lat, lon, hw)
    d = gc_dist(a[:, 0], a[:, 1], lon, lat, R_MOON)
    sel = d <= hw * 1000
    if sel.sum() < 10:
        return {}, f"only {sel.sum()} Diviner cells in window"
    tavg, tmax, ice = a[sel, 3], a[sel, 4], a[sel, 5]
    tavg[tavg <= 0] = np.nan
    tmax[tmax <= 0] = np.nan
    ice = np.where(ice == -1, 0.0, ice)    # -1 = undocumented code for the coldest cold traps: ice stable at the surface
    stable = ice >= 0                       # -999 = ice not stable in the modelled column
    out = dict(dvn_n_cells=int(sel.sum()))
    out.update(stats("dvn_tavg_k", tavg, (5, 50, 95)))
    out.update(stats("dvn_tmax_k", tmax, (5, 50, 95)))
    out.update(
        dvn_frac_coldtrap_110k=float(np.nanmean(tmax < 110)),   # surface H2O-ice stable
        dvn_frac_ice_stable=float(stable.mean()),
        dvn_frac_ice_surface=float((stable & (ice <= 0.01)).mean()),
        dvn_frac_ice_within_1m=float((stable & (ice <= 1.0)).mean()),
        dvn_ice_depth_median_m=float(np.median(ice[stable])) if stable.any() else np.nan,
    )
    return out, ""


def _gcp_band(lat):
    """File tag: 00n10n, 10n20n ... 80n90n, 10s00s, 20s10s ... 90s80s."""
    lo = min(max(int(math.floor(lat / 10.0)) * 10, -90), 80)
    hi = lo + 10
    if lo >= 0:
        return f"{lo:02d}n{hi:02d}n", lo
    return f"{abs(lo):02d}s{abs(hi):02d}s", lo


def g_diviner_gcp(lat, lon, hw):
    """96 local-time bins of bolometric T for the 0.5 deg cell(s) at the site."""
    band, lat_lo = _gcp_band(lat)
    url = SRC["diviner_gcp"].format(band=band)
    rb, nlat, nlt = 113, 20, 96
    ilat = min(int((lat - lat_lo) / 0.5), nlat - 1)
    ilon = min(int((base.norm_lon(lon) + 180) / 0.5), 719)
    want_lon, want_lat = -179.75 + 0.5 * ilon, lat_lo + 0.25 + 0.5 * ilat
    rows = []
    # verified for northern bands: lon-major, lat ascending; other layouts tried as fallback
    for k in (ilon * nlat + ilat, ilon * nlat + (nlat - 1 - ilat), ilat * 720 + ilon, (nlat - 1 - ilat) * 720 + ilon):
        buf = http_range(url, rb + k * nlt * rb, nlt * rb).decode("ascii", "replace")
        rows = [list(map(float, ln.split(","))) for ln in buf.strip().splitlines() if ln.count(",") == 10]
        if rows and abs(rows[0][0] - want_lon) < 0.01 and abs(rows[0][1] - want_lat) < 0.01:
            break
    else:
        return {}, f"GCP row layout not recognised (got {rows[0][:2] if rows else None})"
    a = np.array(rows)
    tb = a[:, 10]
    tb[tb < 0] = np.nan
    cov = float(np.isfinite(tb).mean())
    if cov == 0:
        return {}, "no Diviner GCP data in cell"
    out = dict(dvn_tbol_max_k=float(np.nanmax(tb)), dvn_tbol_min_k=float(np.nanmin(tb)),
               dvn_tbol_mean_k=float(np.nanmean(tb)), dvn_tbol_ltime_coverage=cov,
               dvn_tbol_noon_k=float(np.nanmean(tb[(a[:, 2] > 11) & (a[:, 2] < 13)])),
               dvn_tbol_midnight_k=float(np.nanmean(tb[(a[:, 2] < 1) | (a[:, 2] > 23)])))
    out["dvn_tbol_range_k"] = out["dvn_tbol_max_k"] - out["dvn_tbol_min_k"]
    return out, ""


def g_diviner_rock(lat, lon, hw):
    if abs(lat) > 60:
        return {}, "outside Diviner rock map (|lat| <= 60)"
    out = {}
    for key, name, scale in (("diviner_ra", "dvn_rock_abund", 0.001), ("diviner_st", "dvn_regolith_temp_anom_k", 0.02)):
        a = read_cyl_img(SRC[key], lat, lon, max(hw, 2.0), R_MOON, 32, 60.0, 0.0, 3840, 11520, "<i2")
        a[a == -32768] = np.nan
        out.update(stats(name, a * scale, (50, 95)))
    return out, ""


def g_lola_illum(lat, lon, hw):
    if lat > -75:
        return {}, "outside LOLA south illumination map (<= -75)"
    z, _, _ = base._read_window_single(SRC["lola_avgvis_75s"], "moon", lat, lon, hw)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {}, "no illumination pixels"
    return dict(lola_avg_sun_vis_mean=float(z.mean()), lola_avg_sun_vis_p90=float(np.percentile(z, 90)),
                lola_avg_sun_vis_max=float(z.max()), lola_frac_psr=float((z <= 0).mean()),
                lola_frac_sunlit_gt50pct=float((z > 0.5).mean())), ""


def g_lola_albedo(lat, lon, hw):
    p = download(SRC["lola_albedo"], min_bytes=10_000_000)
    a = read_local_cyl_tif(p, lat, lon, max(hw, 3.0), R_MOON)
    a[a <= 0] = np.nan
    if not np.isfinite(a).any():
        return {}, "no LOLA albedo in window"
    return dict(lola_albedo_raw_mean=float(np.nanmean(a)),
                lola_albedo=float(np.nanmean(a)) * LOLA_ALBEDO_SCALE), ""


LOLA_ALBEDO_SCALE = 0.001   # DN -> normal albedo (verified against published ~0.3 highland values)


def g_lola_terrain(lat, lon, hw):
    if lat <= -80:
        return {}, "polar DEM already used by extract.py"
    z, dx, dy = base._read_window_single(SRC["lola_ldem_118m"], "moon", lat, lon, max(hw, 8.0))
    m = base.terrain_metrics(z, dx, dy)
    m["terrain_source"] = "LOLA LDEM 118 m (extract_extra)"
    return m, ""


# ============================================================================= MARS
def g_tes_ti(lat, lon, hw):
    out = {}
    for key in ("tes_ti_day", "tes_ti_night"):
        a = read_cyl_img(SRC[key], lat, lon, max(hw, 6.0), R_MARS, 20, 90.0, -180.0, 3600, 7200, ">i2")
        a[a <= 0] = np.nan
        out.update(stats(key, a, (50,)))
    return out, ""


def g_tes_albedo(lat, lon, hw):
    p = download(SRC["tes_albedo"], min_bytes=10_000_000)
    a = read_local_cyl_tif(p, lat, lon, max(hw, 8.0), R_MARS)
    return stats("tes_albedo", a), ""


def g_swim(lat, lon, hw):
    if abs(lat) > 60:
        return {}, "outside SWIM map (|lat| <= 60)"
    out = {}
    for key, name in (("swim_c0_1", "swim_ci_0_1m"), ("swim_c1_5", "swim_ci_1_5m"), ("swim_c_5", "swim_ci_gt5m")):
        a = read_local_cyl_tif(SRC[key], lat, lon, max(hw, 8.0), R_MARS)
        a[a <= -99] = np.nan
        out.update(stats(name, a))
    return out, ""


_EZ = None


def _load_ez():
    global _EZ
    if _EZ is None:
        import shapefile
        base_url = SRC["human_ez"]
        paths = {ext: download(f"{base_url}.{ext}", f"human_ez.{ext}", min_bytes=10) for ext in ("shp", "shx", "dbf", "prj")}
        sf = shapefile.Reader(str(paths["shp"]).rsplit(".", 1)[0])
        prj = paths["prj"].read_text(errors="ignore")
        fields = [f[0] for f in sf.fields[1:]]
        recs = []
        for sr in sf.shapeRecords():
            attrs = dict(zip(fields, sr.record))
            recs.append((attrs, np.array(sr.shape.points, dtype=float), list(sr.shape.parts) + [len(sr.shape.points)]))
        _EZ = (recs, prj, fields)
    return _EZ


def _xy_to_lonlat(pts, prj):
    """EZ shapefile is either geographic or equidistant cylindrical in metres."""
    if "PROJCS" in prj.upper() and np.abs(pts).max() > 400:
        k = math.pi / 180 * R_MARS
        return pts[:, 0] / k, pts[:, 1] / k
    return pts[:, 0], pts[:, 1]


def _pip(x, y, poly):
    xs, ys = poly[:, 0], poly[:, 1]
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if ((ys[i] > y) != (ys[j] > y)) and (x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i] + 1e-30) + xs[i]):
            inside = not inside
        j = i
    return inside


def g_human_ez(lat, lon, hw):
    recs, prj, fields = _load_ez()
    name_field = next((f for f in fields if f.lower() in ("label", "name", "ez_name", "site", "sitename", "zone")), fields[0])
    best = (np.inf, "", False)
    lon_n = base.norm_lon(lon)
    for attrs, pts, parts in recs:
        lo, la = _xy_to_lonlat(pts, prj)
        lo = base.norm_lon(lo)
        d = float(np.min(gc_dist(lo, la, lon_n, lat, R_MARS))) / 1000
        inside = any(_pip(lon_n, lat, np.column_stack([lo[a:b], la[a:b]])) for a, b in zip(parts[:-1], parts[1:]))
        if inside:
            d = 0.0
        if d < best[0]:
            best = (d, str(attrs.get(name_field, "")).strip(), inside)
    return dict(ez_nearest=best[1], ez_dist_km=round(best[0], 1), ez_inside=bool(best[2]),
                ez_n_zones=len(recs)), ""


def _mcd_point(lat, lon, ls, lt):
    params = dict(datekeyhtml=1, ls=ls, localtime=lt, latitude=lat, longitude=base.norm_lon(lon),
                  altitude=10, zkey=3, isfixedlt="on", dust=1, hrkey=1, zonmean="off",
                  var1="tsurf", var2="ps", var3="none", var4="none", dpi=80, islog="off",
                  colorm="jet", proj="cyl", iswind="off")
    last = None
    for i in range(3):
        try:
            txt = SESSION.get(SRC["mcd"], params=params, timeout=90).text
            ts = re.search(r"Surface temperature \(K\)\s*\.+\s*([-\d.]+)", txt)
            ps = re.search(r"Surface pressure \(Pa\)\s*\.+\s*([-\d.]+)", txt)
            if ts and ps:
                return float(ts.group(1)), float(ps.group(1))
            raise RuntimeError("MCD output not parsed")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"MCD failed: {last}")


def g_mcd(lat, lon, hw):
    ts, ps = {}, []
    for ls in (0, 90, 180, 270):
        for lt in (4, 14):           # pre-dawn minimum and early-afternoon maximum
            t, p = _mcd_point(lat, lon, ls, lt)
            ts[(ls, lt)] = t
            ps.append(p)
    day = [ts[(ls, 14)] for ls in (0, 90, 180, 270)]
    night = [ts[(ls, 4)] for ls in (0, 90, 180, 270)]
    return dict(mcd_tsurf_max_k=max(day), mcd_tsurf_min_k=min(night),
                mcd_tsurf_mean_k=float(np.mean(day + night)),
                mcd_tsurf_day_mean_k=float(np.mean(day)), mcd_tsurf_night_mean_k=float(np.mean(night)),
                mcd_tsurf_diurnal_range_k=float(np.mean(np.array(day) - np.array(night))),
                mcd_psurf_mean_pa=float(np.mean(ps)), mcd_psurf_min_pa=float(np.min(ps))), ""


# ============================================================================= EARTH
POWER_PARAMS = ["T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "FROST_DAYS", "RH2M", "PRECTOTCORR", "WS2M",
                "PS", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_UV_INDEX", "ALLSKY_SRF_ALB"]


def g_power(lat, lon, hw):
    js = base.http_get_json(SRC["power"], params=dict(parameters=",".join(POWER_PARAMS), community="RE",
                                                      longitude=lon, latitude=lat, format="JSON"))
    p = js["properties"]["parameter"]
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    def ann(k):
        v = p.get(k, {}).get("ANN")
        return np.nan if v is None or v <= -998 else float(v)

    t_month = [p["T2M"][m] for m in months if p["T2M"].get(m, -999) > -998]
    return dict(
        power_t2m_c=ann("T2M"), power_t2m_max_c=ann("T2M_MAX"), power_t2m_min_c=ann("T2M_MIN"),
        power_t2m_diurnal_range_c=ann("T2M_RANGE"),
        power_t2m_coldest_month_c=min(t_month) if t_month else np.nan,
        power_t2m_warmest_month_c=max(t_month) if t_month else np.nan,
        power_frost_days=ann("FROST_DAYS"), power_rh_pct=ann("RH2M"),
        power_precip_mm_yr=ann("PRECTOTCORR") * 365.25, power_wind_ms=ann("WS2M"), power_psurf_kpa=ann("PS"),
        power_sw_allsky_kwh_m2_d=ann("ALLSKY_SFC_SW_DWN"), power_sw_clrsky_kwh_m2_d=ann("CLRSKY_SFC_SW_DWN"),
        power_uv_index=ann("ALLSKY_SFC_UV_INDEX"), power_albedo=ann("ALLSKY_SRF_ALB"),
    ), ""


def _modis_latest(product, band, lat, lon, km):
    dates = base.http_get_json(f"{base.MODIS_API}/{product}/dates", params=dict(latitude=lat, longitude=lon))["dates"]
    d = dates[-1]["modis_date"]
    js = base.http_get_json(f"{base.MODIS_API}/{product}/subset", params=dict(
        latitude=lat, longitude=lon, band=band, startDate=d, endDate=d, kmAboveBelow=km, kmLeftRight=km))
    sub = js["subset"][0]
    data = pd.to_numeric(pd.Series(sub["data"]), errors="coerce").to_numpy(dtype=float)
    return data, float(pd.to_numeric(js.get("scale"), errors="coerce") or 1), d


def g_modis_vcf(lat, lon, hw):
    out = {}
    for band, name in (("Percent_NonVegetated", "modis_pct_bare"), ("Percent_Tree_Cover", "modis_pct_tree"),
                       ("Percent_NonTree_Vegetation", "modis_pct_nontree_veg")):
        v, sc, d = _modis_latest("MOD44B", band, lat, lon, 2)
        v = v[(v >= 0) & (v <= 100)]           # 200 = water, 253 = fill
        out[name] = float(v.mean()) if v.size else np.nan
    out["modis_vcf_date"] = d
    return out, ""


IGBP = {1: "evergreen needleleaf forest", 2: "evergreen broadleaf forest", 3: "deciduous needleleaf forest",
        4: "deciduous broadleaf forest", 5: "mixed forest", 6: "closed shrubland", 7: "open shrubland",
        8: "woody savanna", 9: "savanna", 10: "grassland", 11: "permanent wetland", 12: "cropland",
        13: "urban", 14: "cropland/natural mosaic", 15: "permanent snow and ice", 16: "barren", 17: "water"}


def g_modis_lc(lat, lon, hw):
    v, _, d = _modis_latest("MCD12Q1", "LC_Type1", lat, lon, 2)
    v = v[(v >= 1) & (v <= 17)].astype(int)
    if not v.size:
        return {}, "no MCD12Q1 data"
    vals, cnt = np.unique(v, return_counts=True)
    maj = int(vals[np.argmax(cnt)])
    return dict(modis_lc_majority=IGBP[maj], modis_lc_frac_barren=float(np.mean(v == 16)),
                modis_lc_frac_snow_ice=float(np.mean(v == 15)), modis_lc_year=d[1:5]), ""


WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare_sparse",
              70: "snow_ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}


def g_worldcover(lat, lon, hw):
    import rasterio
    from rasterio.merge import merge
    dlat = math.degrees(hw * 1000 / R_EARTH)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    w, s, e, n = lon - dlon, lat - dlat, lon + dlon, lat + dlat
    srcs = []
    with rasterio.Env(**base.GDAL_ENV):
        for la in range(int(math.floor(s / 3)) * 3, int(math.floor(n / 3)) * 3 + 1, 3):
            for lo in range(int(math.floor(w / 3)) * 3, int(math.floor(e / 3)) * 3 + 1, 3):
                tile = f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
                try:
                    srcs.append(rasterio.open("/vsicurl/" + SRC["worldcover"].format(tile=tile)))
                except Exception:  # noqa: BLE001  (no tile: ocean, or outside 60S-84N)
                    pass
        if not srcs:
            return {}, "no WorldCover tile (coverage is 60S-84N land)"
        try:
            res = 0.0005  # ~50 m: reads overviews, keeps it fast
            mos, _ = merge(srcs, bounds=(w, s, e, n), res=res, nodata=0, resampling=rasterio.enums.Resampling.nearest)
        finally:
            for x in srcs:
                x.close()
    a = mos[0]
    valid = a > 0
    if valid.mean() < 0.05:
        return {}, "WorldCover window empty"
    out = {f"wc_frac_{nm}": float((a[valid] == c).mean()) for c, nm in WC_CLASSES.items()}
    out["wc_valid_frac"] = float(valid.mean())
    return out, ""


_JAXA_CACHE = {}


def _jaxa_item_url(coll, month, lat, lon):
    """Walk the JAXA Earth STAC (collection/MM/level/lonband/latband.json) to the COG covering lat/lon."""
    key = (coll, month, round(lat), round(lon))
    if key in _JAXA_CACHE:
        return _JAXA_CACHE[key]

    def rng(name):
        m = re.findall(r"([NSEW])(\d+\.\d+)", name)
        return [(-1 if h in "SW" else 1) * float(v) for h, v in m]

    def children(url):
        js = base.http_get_json(url)
        return [ln["href"] for ln in js["links"] if ln["rel"] in ("child", "item")], js

    root = f"{coll}/{month:02d}"
    kids, _ = children(f"{root}/catalog.json")
    levels = sorted(int(k.split("/")[1]) for k in kids if re.match(r"\./\d+/catalog.json", k))
    lvl = 1 if 1 in levels else levels[-1]
    lon_kids, _ = children(f"{root}/{lvl}/catalog.json")
    lb = next(k for k in lon_kids if "-" in k and rng(k)[0] <= lon < rng(k)[1] + 1e-9)
    lat_kids, _ = children(f"{root}/{lvl}/{lb[2:]}")
    li = next(k for k in lat_kids if k.endswith(".json") and "-" in k and rng(k)[0] <= lat < rng(k)[1] + 1e-9)
    item_url = f"{root}/{lvl}/{lb[2:].rsplit('/', 1)[0]}/{li[2:]}"
    js = base.http_get_json(item_url)
    href = next(iter(js["assets"].values()))["href"]
    url = item_url.rsplit("/", 1)[0] + "/" + href[2:] if href.startswith("./") else href
    _JAXA_CACHE[key] = url
    return url


def _jaxa_point(coll, lat, lon, km):
    import rasterio
    from rasterio.windows import from_bounds
    vals = []
    for m in range(1, 13):
        url = _jaxa_item_url(coll, m, lat, lon)
        with rasterio.Env(**base.GDAL_ENV):
            with rasterio.open("/vsicurl/" + url) as s:
                d = max(km / 111.0, abs(s.transform.a))
                win = from_bounds(lon - d, lat - d, lon + d, lat + d, transform=s.transform).round_offsets().round_lengths()
                a = s.read(1, window=win, boundless=True, fill_value=np.nan, out_dtype="float64")
                if s.nodata is not None and np.isfinite(s.nodata):
                    a[a == s.nodata] = np.nan
                a[(a < -9000) | (a > 1e30)] = np.nan
                vals.append(float(np.nanmean(a)) if np.isfinite(a).any() else np.nan)
    return np.array(vals)


DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def g_jaxa(lat, lon, hw):
    out, errs = {}, []
    if abs(lat) <= 60:
        try:
            r = _jaxa_point(SRC["jaxa_gsmap"], lat, lon, 10)      # mm/hr monthly-normal rate
            out["jaxa_gsmap_precip_mm_yr"] = float(np.nansum(r * 24 * DAYS_IN_MONTH)) if np.isfinite(r).any() else np.nan
            out["jaxa_gsmap_wettest_month_mm"] = float(np.nanmax(r * 24 * DAYS_IN_MONTH)) if np.isfinite(r).any() else np.nan
        except Exception as e:  # noqa: BLE001
            errs.append(f"gsmap: {str(e)[:80]}")
    else:
        errs.append("gsmap: outside 60S-60N")
    try:
        sm = _jaxa_point(SRC["jaxa_amsr2_sm"], lat, lon, 15)      # volumetric soil moisture (%)
        out["jaxa_amsr2_soil_moist_mean"] = float(np.nanmean(sm)) if np.isfinite(sm).any() else np.nan
        out["jaxa_amsr2_soil_moist_max"] = float(np.nanmax(sm)) if np.isfinite(sm).any() else np.nan
        out["jaxa_amsr2_months_valid"] = int(np.isfinite(sm).sum())
    except Exception as e:  # noqa: BLE001
        errs.append(f"amsr2: {str(e)[:80]}")
    return out, "; ".join(errs)


def g_macrostrat(lat, lon, hw):
    js = base.http_get_json(SRC["macrostrat"], params=dict(lat=lat, lng=base.norm_lon(lon)))
    d = js.get("success", {}).get("data", [])
    if not d:
        return {}, "no Macrostrat unit at point"
    u = d[0]
    return dict(mstrat_unit=u.get("name", ""), mstrat_lith=u.get("lith", ""),
                mstrat_age=f"{u.get('b_int_name', '')}-{u.get('t_int_name', '')}",
                mstrat_age_ma=float(u.get("t_int_age") or np.nan)), ""


FUNCS = {k: v for k, v in globals().items() if k.startswith("g_")}


# ============================================================================= cross-body columns
def add_cross_body(df, main):
    """x_* columns put the same physical quantity from different instruments side by side."""
    m = main.set_index("id") if main is not None and len(main) else pd.DataFrame()
    df = df.set_index("id")

    def col(c, src=df):
        return src[c] if c in src.columns else pd.Series(np.nan, index=src.index)

    body = df["body"]
    xm = pd.DataFrame(index=df.index)
    lst_d = col("lst_day_k", m).reindex(df.index)
    lst_n = col("lst_night_k", m).reindex(df.index)
    xm["x_surf_temp_max_k"] = np.select(
        [body == "moon", body == "mars", body == "earth"],
        [col("dvn_tmax_k_mean").fillna(col("dvn_tbol_max_k")), col("mcd_tsurf_max_k"), lst_d], np.nan)
    xm["x_surf_temp_min_k"] = np.select(
        [body == "moon", body == "mars", body == "earth"],
        [col("dvn_tbol_min_k"), col("mcd_tsurf_min_k"), lst_n], np.nan)
    xm["x_surf_temp_mean_k"] = np.select(
        [body == "moon", body == "mars", body == "earth"],
        [col("dvn_tavg_k_mean").fillna(col("dvn_tbol_mean_k")), col("mcd_tsurf_mean_k"), (lst_d + lst_n) / 2], np.nan)
    xm["x_albedo"] = np.select([body == "moon", body == "mars", body == "earth"],
                               [col("lola_albedo"), col("tes_albedo_mean"), col("power_albedo")], np.nan)
    sun_earth = col("frac_daylight", m).reindex(df.index)
    xm["x_sun_visibility"] = np.where(body == "moon", col("lola_avg_sun_vis_mean"),
                                      np.where(body == "earth", sun_earth, np.nan))
    return pd.concat([df, xm], axis=1).reset_index()


# ============================================================================= driver
def process(row, kind, args):
    sid = row["id"]
    body = row["body"] if kind == "target" else "earth"
    cache_file = CACHE / f"extra_{kind}_{sid}.json"
    rec = json.loads(cache_file.read_text()) if cache_file.exists() and not args.refresh else {}
    done = set(rec.get("_groups_done", []))
    errs = []
    lat, lon, hw = float(row["lat"]), float(row["lon"]), float(row["half_width_km"])
    for g in GROUPS[body]:
        if args.groups and g not in args.groups:
            continue
        if g in done:
            continue
        try:
            vals, note = FUNCS["g_" + g](lat, lon, hw)
            rec.update(vals)
            if note:
                errs.append(f"{g}: {note}")
            done.add(g)        # "not applicable" notes are final; exceptions are retried next run
        except Exception as e:  # noqa: BLE001
            errs.append(f"{g}: ERROR {str(e)[:140]}")
            if args.verbose:
                traceback.print_exc()
    rec.update(id=sid, name=row["name"], body=body, lat=lat, lon=lon, _groups_done=sorted(done))
    cache_file.write_text(json.dumps(rec, default=float))
    rec["extra_notes"] = " | ".join(errs)
    return rec


def check():
    tests = {
        "Diviner PRP south (PDS)": lambda: f"{len(http_range(SRC['diviner_prp_south'], 0, 210))} B header",
        "Diviner GCP (PDS)": lambda: g_diviner_gcp(8.3, 33.2, 5)[0].get("dvn_tbol_max_k"),
        "Diviner rock/regolith (PDS)": lambda: g_diviner_rock(8.3, 33.2, 5)[0].get("dvn_rock_abund_mean"),
        "LOLA illumination (PGDA)": lambda: g_lola_illum(-89.67, 129.78, 5)[0].get("lola_avg_sun_vis_mean"),
        "LOLA albedo (USGS)": lambda: g_lola_albedo(-85, 0, 10)[0].get("lola_albedo_raw_mean"),
        "TES thermal inertia (PDS)": lambda: g_tes_ti(18.38, 77.58, 10)[0].get("tes_ti_night_mean"),
        "TES albedo (USGS)": lambda: g_tes_albedo(18.38, 77.58, 10)[0].get("tes_albedo_mean"),
        "SWIM ice (PSI/JPL)": lambda: g_swim(47.2, -175.7, 15)[0].get("swim_ci_0_1m_mean"),
        "Human EZ (NASA/USGS)": lambda: g_human_ez(18.38, 77.58, 15)[0],
        "Mars Climate Database": lambda: _mcd_point(18.38, 77.58, 90, 14),
        "NASA POWER": lambda: g_power(75.38, -89.68, 10)[0].get("power_t2m_c"),
        "MODIS MOD44B (ORNL)": lambda: g_modis_vcf(35.03, -111.02, 5)[0].get("modis_pct_bare"),
        "ESA WorldCover": lambda: g_worldcover(35.03, -111.02, 5)[0].get("wc_frac_bare_sparse"),
        "JAXA GSMaP item": lambda: _jaxa_item_url(SRC["jaxa_gsmap"], 1, 35.03, -111.02)[-60:],
        "Macrostrat": lambda: g_macrostrat(35.03, -111.02, 5)[0].get("mstrat_lith"),
    }
    ok = True
    for name, fn in tests.items():
        try:
            msg, flag = fn(), "OK  "
        except Exception as e:  # noqa: BLE001
            msg, flag, ok = str(e)[:150], "FAIL", False
        print(f"  [{flag}] {name:<30} {str(msg)[:110]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="")
    ap.add_argument("--groups", default="", help="comma-separated group names (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if args.check:
        return 0 if check() else 1
    CACHE.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if any(g in args.groups or not args.groups for g in ("diviner_polar",)):
        print(f"raw downloads -> {RAW}")
    for kind, cfg, main_csv, out_csv in (("target", "targets.csv", "target_features.csv", "target_extra.csv"),
                                         ("earth", "earth_sites.csv", "earth_features.csv", "earth_extra.csv")):
        df = pd.read_csv(CFG / cfg)
        rows = [r for _, r in df.iterrows() if not only or r["id"] in only]
        if not rows:
            continue
        print(f"\n== {kind}: {len(rows)} sites ==")
        recs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process, r, kind, args): r["id"] for r in rows}
            for f in as_completed(futs):
                rec = f.result()
                flag = "ERR" if "ERROR" in rec["extra_notes"] else "OK "
                print(f"  [{flag}] {rec['id']:<28} {rec['extra_notes'][:150]}")
                recs.append(rec)
        out = pd.DataFrame(recs).drop(columns=["_groups_done"], errors="ignore")
        path = DATA / out_csv
        if path.exists() and only:
            old = pd.read_csv(path)
            out = pd.concat([old[~old["id"].isin(out["id"])], out], ignore_index=True)
        main_df = pd.read_csv(DATA / main_csv) if (DATA / main_csv).exists() else None
        out = add_cross_body(out, main_df)
        order = {sid: i for i, sid in enumerate(df["id"])}
        out = out.sort_values("id", key=lambda s: s.map(order))
        out.to_csv(path, index=False)
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
