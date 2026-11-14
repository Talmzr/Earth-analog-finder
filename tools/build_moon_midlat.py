#!/usr/bin/env python
"""
Mid-latitude Moon tiles (60S-60N): ~2,000 extra 10 x 10 km tiles so the Moon side is not only
south-polar highlands (adds mare basalt, fresh Copernican craters, low-latitude terra/basin/plains)
and so the Mare Tranquillitatis pit target (8.3N, 33.2E) gets a tile.

Same feature definitions as data/ml/moon_tiles.csv (extract.terrain_metrics on a 10 km ground
window, geometric illumination, Diviner GCP day/night, USGS unified geology fractions) plus
Diviner rock abundance + regolith temperature anomaly (|lat| <= 60 products).

Sources
  terrain   SLDEM2015 LOLA+Kaguya TC merge, 512 ppd (59.2 m/px) equirectangular, 60N-60S, USGS S3
            mosaic/LolaKaguya_Topo/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif. The GeoTIFF is
            NOT tiled (one strip per row, uncompressed int16, scale 0.5 m), so GDAL window reads are
            slow; instead the strip offsets are parsed from the TIFF header and each needed row span is
            fetched with a plain HTTP range request (threaded). Tiles are placed in latitude BANDS that
            share the same 169 rows, so neighbouring tiles in a band are fetched with one request per
            row (spans merged where that is cheaper than an extra request).
  Diviner   GCP global_cumul_avg_cyl_<band>_002.tab: one 0.5 deg cell x 96 local-time bins per tile
            (range read, same layout check as extract_extra.g_diviner_gcp); rock abundance and
            regolith temperature anomaly dgdr_ra/st_avg_cyl_032 (32 ppd, rows range-read once per band)
  geology   USGS Unified Geologic Map of the Moon 64 ppd raster (local, ~/.cache/earth_analog_raw/ugm)

Sampling: latitude bands every 3 deg anchored on the pit latitude (8.3N, ..., -57.7 ... 59.3),
candidate slots every 20 km along each band (bands 91 km apart -> >= 20 km spacing), each candidate
classified by its geology (stratum: mare / terra / crater / crater_copernican / basin / plains /
mixed), quota-stratified random selection (fixed seed). The pit tile is always included.

No PSR / ice at these latitudes: Diviner GCP maxima are >= ~250 K everywhere in 60S-60N (no
cold traps), so x_shadowed_frac = x_surface_ice_frac = x_frozen_ground_frac = 0 and the PRP-only
columns (dvn_tavg_*, dvn_tmax_*, ice depth) are NaN. LOLA illumination columns are NaN (polar
product); x_sunlit_frac_terrain is filled with the geometric daylight fraction (illum_source).

Outputs (the main tables are NOT touched; merge later)
  data/patches/moon_midlat_patches.csv     one row per tile, moon_tiles.csv column names + region
  data/patches/moon_midlat_arrays_128.npz  patch_id, center_lat, center_lon, elev, slope (128x128 f16)
  ~/.cache/earth_analog_raw/moon_midlat_ckpt/  per-band checkpoints (resume by re-running)

  python tools/build_moon_midlat.py [--tiles 2000] [--threads 48] [--time-budget-min 24]
"""
import argparse
import math
import os
import pickle
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import extract as base  # noqa: E402
import extract_extra as ex  # noqa: E402
import build_moon_patches as bmp  # noqa: E402  GEO_CLASSES, GEO_AGES, _main_age, ensure_geology
from build_ml_tables import grazing_sun_frac  # noqa: E402

RAW = Path(os.environ.get("ANALOG_RAW_DIR", os.path.expanduser("~/.cache/earth_analog_raw")))
OUT_DIR = ROOT / "data" / "patches"
CKPT = RAW / "moon_midlat_ckpt"
SLDEM_URL = ("https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/LolaKaguya_Topo/"
             "Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif")
PPD, NCOLS, NROWS, LAT_TOP = 512, 184320, 61440, 60.0
Z_SCALE, Z_NODATA = 0.5, -32768
R = base.BODY["moon"]["R"]
PX_M = R * math.radians(1.0 / PPD)          # 59.225 m
TILE_PX = 169                               # 169 x 59.225 m = 10.01 km
HALF_DEG = TILE_PX / 2 / PPD                # 0.165 deg
PATCH_M = TILE_PX * PX_M
N128 = 128
NG = 32                                     # geology sampling grid per tile (as polar)
MIN_VALID = 0.8
PIT = dict(id="moon_tranquillitatis_pit", lat=8.3, lon=33.2)
SEED = 20261114
QUOTA = dict(mare=0.24, terra=0.20, crater=0.16, crater_copernican=0.13, basin=0.13, plains=0.11, mixed=0.03)
COP_CRATER_DN = (1, 2, 3)                   # Cc, Ccc, Csc


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================= HTTP
_tl = threading.local()
STATS = dict(req=0, bytes=0)
_slock = threading.Lock()


def session():
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=2))
        _tl.s = s
    return s


def rget(url, start, length, tries=6):
    last = None
    for i in range(tries):
        try:
            r = session().get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=60)
            if r.status_code == 206 and len(r.content) == length:
                with _slock:
                    STATS["req"] += 1
                    STATS["bytes"] += length
                return r.content
            last = f"HTTP {r.status_code} {len(r.content)}B"
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(min(2 ** i, 20))
    raise RuntimeError(f"range GET failed {url} @{start}: {last}")


def strip_offsets():
    """Row byte offsets of the striped SLDEM GeoTIFF (BigTIFF tag 273), cached locally."""
    cache = CKPT / "sldem_strip_offsets.npy"
    if cache.exists():
        return np.load(cache)
    h = rget(SLDEM_URL, 0, 16)
    assert h[:4] == b"II+\x00", h[:4]
    ifd = struct.unpack("<Q", h[8:16])[0]
    n = struct.unpack("<Q", rget(SLDEM_URL, ifd, 8))[0]
    ent = rget(SLDEM_URL, ifd + 8, n * 20)
    tags = {}
    for i in range(n):
        tag, typ, c, v = struct.unpack("<HHQQ", ent[i * 20:(i + 1) * 20])
        tags[tag] = (typ, c, v)
    assert tags[256][2] == NCOLS and tags[257][2] == NROWS and tags[259][2] == 1 and tags[258][2] == 16
    typ, c, off = tags[273]
    sz = 8 if typ == 16 else 4
    offs = np.frombuffer(rget(SLDEM_URL, off, c * sz), dtype="<u8" if sz == 8 else "<u4").astype(np.int64)
    assert len(offs) == NROWS
    np.save(cache, offs)
    return offs


# ============================================================================= sampling
def band_rows():
    """Band top rows (169-row bands), every 3 deg, anchored on the pit latitude."""
    out = []
    lo = math.ceil((-LAT_TOP + HALF_DEG + 0.01 - PIT["lat"]) / 3.0)
    hi = math.floor((LAT_TOP - HALF_DEG - 0.01 - PIT["lat"]) / 3.0)
    for k in range(lo, hi + 1):
        lat = PIT["lat"] + 3.0 * k
        r0 = int(round((LAT_TOP - lat) * PPD - TILE_PX / 2))
        if 0 <= r0 and r0 + TILE_PX <= NROWS:
            out.append(r0)
    return out


def band_lat(r0):
    return LAT_TOP - (r0 + TILE_PX / 2) / PPD


def tile_ncol(lat):
    return int(round(TILE_PX / math.cos(math.radians(lat))))


def candidates(rng):
    rows = []
    for b, r0 in enumerate(band_rows()):
        lat = band_lat(r0)
        nc = tile_ncol(lat)
        step = 2 * nc                                            # 20 km centre spacing
        pit_band = abs(lat - PIT["lat"]) < 0.01
        if pit_band:
            pc0 = int(round((PIT["lon"] + 180) * PPD - nc / 2))
            off = pc0 % step
        else:
            off = int(rng.integers(0, step))
        for c0 in range(off, NCOLS - nc + 1, step):
            lon = (c0 + nc / 2) / PPD - 180.0
            tgt = int(pit_band and c0 == pc0)
            pid = PIT["id"] if tgt else f"mm_{lat:+06.2f}_{lon:+07.2f}"
            rows.append(dict(patch_id=pid, band=b, r0=r0, c0=c0, ncol=nc, center_lat=lat, center_lon=lon, is_target=tgt))
    return pd.DataFrame(rows)


def geology_codes(cand):
    """NG x NG UGM unit codes inside each candidate tile (lat/lon box)."""
    import rasterio
    from rasterio.windows import Window
    tif, dn, cat = bmp.ensure_geology(False)
    codes = np.zeros((len(cand), NG * NG), np.int16)
    off = (np.arange(NG) + 0.5) / NG
    with rasterio.open(tif) as s:
        ppd = 1.0 / s.transform.a
        left, top = s.bounds.left, s.bounds.top
        for b, g in cand.groupby("band"):
            lat = g["center_lat"].iloc[0]
            ra = int((top - (lat + HALF_DEG)) * ppd) - 1
            rb = int((top - (lat - HALF_DEG)) * ppd) + 2
            arr = s.read(1, window=Window(0, ra, s.width, rb - ra)).astype(np.int16)
            lat_pts = (lat + HALF_DEG) - off * 2 * HALF_DEG                       # NG
            c0 = g["c0"].to_numpy()[:, None]
            nc = g["ncol"].to_numpy()[:, None]
            lon_pts = (c0 + off[None, :] * nc) / PPD - 180.0                        # n x NG
            rr = np.clip(((top - lat_pts) * ppd).astype(int) - ra, 0, arr.shape[0] - 1)
            cc = np.clip(((lon_pts - left) * ppd).astype(int), 0, arr.shape[1] - 1)
            code = arr[rr[None, :, None], cc[:, None, :]].reshape(len(g), NG * NG)
            codes[g.index.to_numpy()] = code
        nod = s.nodata
    if nod is not None:
        codes[codes == nod] = 0
    codes[codes < 0] = 0
    return codes, dn, cat


def geology_table(codes, dn, cat):
    """Same columns / logic as build_moon_patches.geology_features, from unit codes."""
    n = len(codes)
    ncode = int(max(dn["DN"].max(), codes.max())) + 1
    counts = np.zeros((n, ncode), np.int64)
    np.add.at(counts, (np.repeat(np.arange(n), codes.shape[1]), codes.ravel()), 1)
    valid = counts[:, 1:].sum(1)
    maj = counts[:, 1:].argmax(1) + 1
    dn_unit = dict(zip(dn["DN"], dn["Unit"]))
    dn_name = dict(zip(dn["DN"], dn["Unit_Name"]))
    dn_age = dict(zip(dn["DN"], dn["Unit_Age"]))
    unit_name = dict(zip(cat["unit"], cat["name"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame(dict(
            moon_geo_unit=[dn_unit.get(m, "") if v else "" for m, v in zip(maj, valid)],
            moon_geo_unit_frac=np.where(valid > 0, counts[np.arange(n), maj] / valid, np.nan)))
        out["moon_geo_unit_name"] = out["moon_geo_unit"].map(unit_name).fillna("")
        out["moon_geo_unit_age"] = [dn_age.get(m, "") if v else "" for m, v in zip(maj, valid)]
        out["moon_geo_n_units"] = (counts[:, 1:] > 0).sum(1)
        for cls, fn in bmp.GEO_CLASSES.items():
            cols = [d for d in dn["DN"] if fn(dn_unit[d], dn_name[d])]
            out[f"geo_frac_{cls}"] = np.where(valid > 0, counts[:, cols].sum(1) / valid, np.nan)
        for key, age in bmp.GEO_AGES.items():
            cols = [d for d in dn["DN"] if bmp._main_age(dn_age[d]) == age]
            out[f"geo_frac_age_{key}"] = np.where(valid > 0, counts[:, cols].sum(1) / valid, np.nan)
        out["geo_valid_frac"] = valid / codes.shape[1]
        out["geo_frac_copernican_crater"] = np.where(valid > 0, counts[:, list(COP_CRATER_DN)].sum(1) / valid, np.nan)
    return out


def stratum(geo):
    cls = ["mare", "terra", "crater", "basin", "plains"]
    f = geo[[f"geo_frac_{c}" for c in cls]].fillna(0).to_numpy()
    s = np.array(cls, dtype=object)[f.argmax(1)]
    s = np.where(f.max(1) < 0.4, "mixed", s)
    s = np.where(geo["geo_frac_copernican_crater"].fillna(0) >= 0.3, "crater_copernican", s)
    return s


def select(cand, n_tiles, rng):
    avail = cand["stratum"].value_counts().to_dict()
    want = {k: int(round(q * n_tiles)) for k, q in QUOTA.items()}
    # classes with too few candidates: take all, give the rest to the others (proportionally)
    for _ in range(5):
        short = {k: want[k] - avail.get(k, 0) for k in want if want[k] > avail.get(k, 0)}
        spare = sum(short.values())
        if not spare:
            break
        for k in short:
            want[k] = avail.get(k, 0)
        open_ = [k for k in want if want[k] < avail.get(k, 0)]
        tot = sum(QUOTA[k] for k in open_)
        for k in open_:
            want[k] += int(round(spare * QUOTA[k] / tot))
    keep = [cand.index[cand["is_target"] == 1]]
    for k, m in want.items():
        idx = cand.index[(cand["stratum"] == k) & (cand["is_target"] == 0)].to_numpy()
        keep.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    sel = cand.loc[np.unique(np.concatenate(keep))].copy()
    return sel, want, avail


# ============================================================================= DEM band fetch
def plan_clusters(c0s, ncols, req_rate, bw):
    """Merge tile column spans of one band; pick the merge gap minimising est. time."""
    order = np.argsort(c0s)
    spans = [(int(c0s[i]), int(c0s[i] + ncols[i])) for i in order]
    best = None
    for gap in (0, 300, 1000, 2500, 5000, 10000, 20000, 10 ** 9):
        cl = [list(spans[0])]
        for a, b in spans[1:]:
            if a - cl[-1][1] <= gap:
                cl[-1][1] = max(cl[-1][1], b)
            else:
                cl.append([a, b])
        nreq = len(cl) * TILE_PX
        nbytes = sum(b - a for a, b in cl) * 2 * TILE_PX
        t = nreq / req_rate + nbytes / bw
        if best is None or t < best[0]:
            best = (t, gap, cl)
    return best


def fetch_band(pool, offs, r0, clusters):
    bufs = [np.empty((TILE_PX, b - a), np.int16) for a, b in clusters]

    def one(ci, ri):
        a, b = clusters[ci]
        raw = rget(SLDEM_URL, int(offs[r0 + ri]) + 2 * a, 2 * (b - a))
        bufs[ci][ri] = np.frombuffer(raw, "<i2")

    futs = [pool.submit(one, ci, ri) for ci in range(len(clusters)) for ri in range(TILE_PX)]
    return bufs, futs


def tile_arrays(z, lat):
    """terrain_metrics + 128x128 elev/slope from one 169 x ncol window (m)."""
    from scipy.ndimage import map_coordinates
    lon0 = 0.0
    dx = float(base.haversine(lon0, lat, lon0 + 1.0 / PPD, lat, R))     # as extract._pixel_ground_spacing
    dy = float(base.haversine(lon0, lat, lon0, lat + 1.0 / PPD, R))
    rec = base.terrain_metrics(z, dx, dy)
    h, w = z.shape
    zf = np.where(np.isfinite(z), z, np.nanmean(z))
    yy = (np.arange(N128) + 0.5) * h / N128 - 0.5
    xx = (np.arange(N128) + 0.5) * w / N128 - 0.5
    YY, XX = np.meshgrid(yy, xx, indexing="ij")
    z128 = map_coordinates(zf, [YY.ravel(), XX.ravel()], order=1, mode="nearest").reshape(N128, N128)
    s128 = base._slope_deg(z128, w * dx / N128, h * dy / N128)
    e128 = z128 - z128.mean()
    return rec, e128.astype(np.float16), np.nan_to_num(s128).astype(np.float16)


# ============================================================================= Diviner
def gcp_cell(lat, lon):
    band, lat_lo = ex._gcp_band(lat)
    ilat = min(int((lat - lat_lo) / 0.5), 19)
    ilon = min(int((base.norm_lon(lon) + 180) / 0.5), 719)
    return band, lat_lo, ilat, ilon


def fetch_gcp(key):
    """One 0.5 deg cell x 96 local-time bins (logic of extract_extra.g_diviner_gcp)."""
    band, lat_lo, ilat, ilon = key
    url = ex.SRC["diviner_gcp"].format(band=band)
    rb, nlat, nlt = 113, 20, 96
    want_lon, want_lat = -179.75 + 0.5 * ilon, lat_lo + 0.25 + 0.5 * ilat
    for k in (ilon * nlat + ilat, ilon * nlat + (nlat - 1 - ilat), ilat * 720 + ilon, (nlat - 1 - ilat) * 720 + ilon):
        buf = rget(url, rb + k * nlt * rb, nlt * rb).decode("ascii", "replace")
        rows = [list(map(float, ln.split(","))) for ln in buf.strip().splitlines() if ln.count(",") == 10]
        if rows and abs(rows[0][0] - want_lon) < 0.01 and abs(rows[0][1] - want_lat) < 0.01:
            break
    else:
        return dict(gcp_err="layout")
    a = np.array(rows)
    tb = a[:, 10].copy()
    tb[tb < 0] = np.nan
    cov = float(np.isfinite(tb).mean())
    out = dict(gcp_clon=want_lon, gcp_clat=want_lat, dvn_tbol_ltime_cov=cov)
    if cov > 0:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.update(dvn_tbol_min_k=float(np.nanmin(tb)), dvn_tbol_max_k=float(np.nanmax(tb)),
                       dvn_tbol_mean_k=float(np.nanmean(tb)),
                       dvn_tbol_noon_k=float(np.nanmean(tb[(a[:, 2] > 11) & (a[:, 2] < 13)])),
                       dvn_tbol_midnight_k=float(np.nanmean(tb[(a[:, 2] < 1) | (a[:, 2] > 23)])))
    return out


def fetch_rock_band(lat):
    """Rows of the Diviner rock-abundance (RA) and regolith-temperature (ST) 32 ppd maps for a band."""
    out = {}
    l0 = int(math.floor((60.0 - (lat + HALF_DEG)) * 32))
    l1 = int(math.floor((60.0 - (lat - HALF_DEG)) * 32))
    for key in ("diviner_ra", "diviner_st"):
        buf = rget(ex.SRC[key], l0 * 11520 * 2, (l1 - l0 + 1) * 11520 * 2)
        out[key] = np.frombuffer(buf, "<i2").reshape(l1 - l0 + 1, 11520).astype(np.float64)
    out["l0"] = l0
    return out


def rock_feats(rb, lat, lon, half_lon_deg):
    lines = rb["l0"] + np.arange(rb["diviner_ra"].shape[0])
    plat = 60.0 - (lines + 0.5) / 32
    plon = (np.arange(11520) + 0.5) / 32                          # 0..360
    rin = np.abs(plat - lat) <= HALF_DEG
    cin = np.abs(((plon - lon + 180) % 360) - 180) <= half_lon_deg
    out = {}
    for key, name, scale in (("diviner_ra", "dvn_rock_abund", 0.001), ("diviner_st", "dvn_regolith_temp_anom_k", 0.02)):
        a = rb[key][np.ix_(rin, cin)].copy()
        a[a == -32768] = np.nan
        out.update(ex.stats(name, a * scale, (50, 95)))
        out[f"{name}_n_px"] = int(np.isfinite(a).sum())
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=48, help="SLDEM range-request threads")
    ap.add_argument("--dvn-threads", type=int, default=12)
    ap.add_argument("--time-budget-min", type=float, default=24.0, help="stop starting new bands after this")
    ap.add_argument("--max-bands", type=int, default=0, help="debug: only this many bands")
    ap.add_argument("--req-rate", type=float, default=250.0, help="cost model: requests/s")
    ap.add_argument("--bw", type=float, default=7e6, help="cost model: bytes/s")
    args = ap.parse_args()
    t0 = time.time()
    CKPT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- candidates + geology strata
    cand = candidates(rng)
    codes, dn, cat = geology_codes(cand)
    geo = geology_table(codes, dn, cat)
    cand = pd.concat([cand, geo], axis=1)
    cand["stratum"] = stratum(geo)
    cand = cand[cand["geo_valid_frac"] > 0.5].copy()
    sel, want, avail = select(cand, args.tiles, rng)
    log(f"{len(band_rows())} bands, {len(cand)} candidates; available per stratum {avail}")
    log(f"selected {len(sel)} tiles: {sel['stratum'].value_counts().to_dict()}")
    assert (sel["patch_id"] == PIT["id"]).sum() == 1, "pit tile missing"

    # ---------------------------------------------------------------- Diviner (background pool)
    dpool = ThreadPoolExecutor(args.dvn_threads)
    gcp_path = CKPT / "gcp_cells.pkl"
    gcp = pickle.loads(gcp_path.read_bytes()) if gcp_path.exists() else {}
    keys = sorted({gcp_cell(r.center_lat, r.center_lon) for r in sel.itertuples()} - set(gcp))
    gfut = {k: dpool.submit(fetch_gcp, k) for k in keys}
    bands = sorted(sel["band"].unique())
    rfut = {}
    log(f"Diviner: {len(keys)} GCP cells to fetch ({len(gcp)} cached), rock rows for {len(bands)} bands")

    # ---------------------------------------------------------------- DEM bands
    offs = strip_offsets()
    pit_band = int(sel.loc[sel["patch_id"] == PIT["id"], "band"].iloc[0])
    rest = [b for b in bands if b != pit_band]
    order = [pit_band] + rest[0::2] + rest[1::2]                 # interleaved: partial runs stay spread
    if args.max_bands:
        order = order[:args.max_bands]
    pool = ThreadPoolExecutor(args.threads)
    todo = [b for b in order if not (CKPT / f"band_{b:03d}.pkl").exists()]
    log(f"DEM: {len(order) - len(todo)} bands cached, {len(todo)} to fetch")

    def start(b):
        g = sel[sel["band"] == b]
        est, gap, cl = plan_clusters(g["c0"].to_numpy(), g["ncol"].to_numpy(), args.req_rate, args.bw)
        r0 = int(g["r0"].iloc[0])
        rf = rfut.get(b) or dpool.submit(fetch_rock_band, band_lat(r0))
        rfut[b] = rf
        bufs, futs = fetch_band(pool, offs, r0, cl)
        return dict(b=b, g=g, cl=cl, bufs=bufs, futs=futs, est=est, gap=gap, t=time.time())

    n_done_tiles, t_dem, stopped = 0, time.time(), False
    pending = start(todo[0]) if todo else None
    for i, b in enumerate(todo):
        nxt = start(todo[i + 1]) if i + 1 < len(todo) and (time.time() - t0) / 60 < args.time_budget_min else None
        if i + 1 < len(todo) and nxt is None:
            stopped = True
        cur = pending
        s0 = dict(STATS)
        for f in cur["futs"]:
            f.result()
        rb = rfut[b].result()
        recs, E, S = [], [], []
        for r in cur["g"].itertuples():
            ci = next(k for k, (a, bb) in enumerate(cur["cl"]) if a <= r.c0 and r.c0 + r.ncol <= bb)
            a = cur["cl"][ci][0]
            z = cur["bufs"][ci][:, r.c0 - a:r.c0 - a + r.ncol].astype(np.float64)
            z[z == Z_NODATA] = np.nan
            z *= Z_SCALE
            vf = float(np.isfinite(z).mean())
            if vf < MIN_VALID and not r.is_target:
                continue
            try:
                rec, e, s = tile_arrays(z, r.center_lat)
            except Exception as err:  # noqa: BLE001
                log("  terrain fail", r.patch_id, err)
                continue
            rec["patch_id"] = r.patch_id
            rec.update(rock_feats(rb, r.center_lat, r.center_lon, r.ncol / 2 / PPD))
            recs.append(rec)
            E.append(e)
            S.append(s)
        (CKPT / f"band_{b:03d}.pkl").write_bytes(pickle.dumps(dict(recs=recs, E=E, S=S)))
        n_done_tiles += len(recs)
        el = time.time() - t_dem
        left = sum(int((sel["band"] == bb).sum()) for bb in todo[i + 1:]) if not stopped else 0
        rate = n_done_tiles / el
        log(f"band {i + 1}/{len(todo)} lat {band_lat(int(cur['g']['r0'].iloc[0])):+6.1f}: {len(recs)} tiles, "
            f"{len(cur['cl'])} spans (gap {cur['gap']}), est {cur['est']:.0f}s | total {n_done_tiles} tiles "
            f"{rate:.2f}/s, {STATS['bytes'] / el / 1e6:.1f} MB/s, {STATS['req'] / el:.0f} req/s | "
            f"ETA {left / max(rate, 1e-6) / 60:.1f} min | GCP {sum(f.done() for f in gfut.values())}/{len(gfut)}")
        del cur
        pending = nxt
        if nxt is None:
            break
    if stopped:
        log(f"time budget {args.time_budget_min} min reached: stopped after {i + 1} of {len(todo)} bands")
    pool.shutdown(wait=True, cancel_futures=True)

    # ---------------------------------------------------------------- gather
    recs, E, S = [], [], []
    for b in order:
        p = CKPT / f"band_{b:03d}.pkl"
        if p.exists():
            d = pickle.loads(p.read_bytes())
            recs += d["recs"]
            E += d["E"]
            S += d["S"]
    log(f"waiting for Diviner GCP ({sum(f.done() for f in gfut.values())}/{len(gfut)} done) ...")
    for k, f in gfut.items():
        try:
            gcp[k] = f.result()
        except Exception as err:  # noqa: BLE001
            gcp[k] = dict(gcp_err=str(err)[:80])
    gcp_path.write_bytes(pickle.dumps(gcp))
    dpool.shutdown(wait=False, cancel_futures=True)

    df = pd.DataFrame(recs)
    meta = sel.set_index("patch_id")
    df = df.join(meta[["center_lat", "center_lon", "is_target", "stratum"]], on="patch_id")
    geo_cols = [c for c in geo.columns if c != "geo_frac_copernican_crater"] + ["geo_frac_copernican_crater"]
    df = df.join(meta[geo_cols], on="patch_id")
    pit = df["patch_id"] == PIT["id"]
    df.loc[pit, "center_lat"], df.loc[pit, "center_lon"] = PIT["lat"], PIT["lon"]

    ill = pd.DataFrame([base.illumination(lat, "moon") for lat in df["center_lat"]])
    df = pd.concat([df.reset_index(drop=True), ill], axis=1)
    gd = pd.DataFrame([dict(gcp.get(gcp_cell(r.center_lat, r.center_lon), {})) for r in df.itertuples()])
    for c in ("dvn_tbol_min_k", "dvn_tbol_max_k", "dvn_tbol_mean_k", "dvn_tbol_noon_k", "dvn_tbol_midnight_k",
              "dvn_tbol_ltime_cov", "gcp_clat", "gcp_clon"):
        df[c] = gd[c].to_numpy() if c in gd else np.nan
    # nearest_km: 0 when the 0.5 deg cell centre lies inside the tile, else centre-to-centre distance
    dkm = base.haversine(df["center_lon"], df["center_lat"], df["gcp_clon"], df["gcp_clat"], R) / 1000
    inside = (np.abs(df["gcp_clat"] - df["center_lat"]) <= HALF_DEG) & \
             (np.abs(df["gcp_clon"] - df["center_lon"]) <= HALF_DEG / np.cos(np.radians(df["center_lat"])))
    df["dvn_tbol_n_cells"] = df["dvn_tbol_mean_k"].notna().astype(int)
    df["dvn_tbol_nearest_km"] = np.where(inside, 0.0, dkm).round(2)

    # polar-only products -> NaN / physical zeros
    for c in ("lola_avg_sun_vis_mean", "lola_avg_sun_vis_p90", "lola_avg_sun_vis_max", "lola_frac_psr",
              "lola_frac_sunlit_gt50pct", "dvn_tavg_mean_k", "dvn_tavg_min_k", "dvn_tmax_mean_k", "dvn_tmax_max_k",
              "dvn_tmax_min_k", "dvn_ice_depth_median_m", "x_m", "y_m"):
        df[c] = np.nan
    df["dvn_frac_coldtrap_110k"] = np.where(df["dvn_tbol_max_k"].notna(), (df["dvn_tbol_max_k"] < 110).astype(float), np.nan)
    for c in ("dvn_frac_ice_stable", "dvn_frac_ice_within_1m", "dvn_frac_ice_surface"):
        df[c] = 0.0
    df["dvn_n_cells"] = 0
    df["body"], df["region"] = "moon", "midlat"
    df["terrain_source"] = "SLDEM2015 LOLA+Kaguya 59 m"
    df["illum_source"] = "geometric"
    df["x_temp_max_k"] = df["dvn_tbol_max_k"]
    df["x_temp_mean_k"] = df["dvn_tbol_mean_k"]
    df["x_temp_min_k"] = df["dvn_tbol_min_k"]
    df["x_temp_swing_k"] = df["x_temp_max_k"] - df["x_temp_min_k"]
    df["x_grazing_sun_frac"] = grazing_sun_frac(df["center_lat"], 1.54)
    df["x_noon_sun_max_deg"] = df["noon_elev_max_deg"]
    df["x_polar_night_frac"] = df["frac_polar_night"]
    df["x_frozen_ground_frac"] = 0.0
    df["x_surface_ice_frac"] = 0.0
    df["x_sunlit_frac_terrain"] = df["frac_daylight"]
    df["x_shadowed_frac"] = 0.0

    ref = list(pd.read_csv(ROOT / "data" / "ml" / "moon_tiles.csv", nrows=0).columns)
    front = [c for c in ref if c in df.columns]
    if "region" not in front:
        front.insert(front.index("body") + 1, "region")
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values(["is_target", "center_lat", "center_lon"], ascending=[False, True, True])
    order_idx = df.index.to_numpy()
    df = df.reset_index(drop=True)
    E = np.stack(E)[order_idx]
    S = np.stack(S)[order_idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, npz_path = OUT_DIR / "moon_midlat_patches.csv", OUT_DIR / "moon_midlat_arrays_128.npz"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    np.savez_compressed(npz_path, patch_id=np.asarray(df["patch_id"].astype(str).tolist(), dtype="U32"),
                        center_lat=df["center_lat"].to_numpy(np.float32),
                        center_lon=df["center_lon"].to_numpy(np.float32), elev=E, slope=S)
    log(f"wrote {csv_path} ({df.shape[0]} x {df.shape[1]}), {npz_path} ({npz_path.stat().st_size / 1e6:.0f} MB)")
    summarize(df, ref)
    log(f"total runtime {(time.time() - t0) / 60:.1f} min; {STATS['req']} requests, {STATS['bytes'] / 1e6:.0f} MB")
    return 0


def summarize(df, ref):
    print(f"\ntiles: {len(df)} (targets {int(df['is_target'].sum())}); strata {df['stratum'].value_counts().to_dict()}")
    print("majority units:", df["moon_geo_unit"].value_counts().head(10).to_dict())
    missing = [c for c in ref if c not in df.columns]
    print("moon_tiles.csv columns missing here:", missing)
    cov = df.notna().mean()
    print("coverage < 100%:", {c: round(v, 3) for c, v in cov.items() if v < 1})
    key = ["center_lat", "relief_m", "slope500_mean_deg", "tri500_mean_m", "dvn_tbol_min_k", "dvn_tbol_max_k",
           "dvn_tbol_mean_k", "dvn_rock_abund_mean", "dvn_rock_abund_p95", "dvn_regolith_temp_anom_k_mean",
           "x_grazing_sun_frac", "x_noon_sun_max_deg", "geo_frac_mare"]
    print(df[key].describe(percentiles=[0.5]).T[["count", "min", "50%", "max"]].round(3).to_string())
    print("\nby stratum (median):")
    print(df.groupby("stratum")[["relief_m", "slope500_mean_deg", "dvn_rock_abund_mean", "dvn_rock_abund_p95",
                                 "dvn_regolith_temp_anom_k_mean", "dvn_tbol_max_k", "dvn_tbol_min_k"]].median().round(4).to_string())
    print("\npit tile:")
    print(df[df["is_target"] == 1].T.to_string())


if __name__ == "__main__":
    sys.exit(main())
