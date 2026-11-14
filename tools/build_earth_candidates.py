#!/usr/bin/env python
"""
Build the Earth candidate list for 10 km x 10 km analog patches.

Sources
  named_site        config/earth_sites.csv (30 curated analog sites)
  impact_structure  Wikipedia "List of impact structures on Earth" (RAW/wiki_craters.json);
                    rows without inline coordinates are filled from the Wikipedia coordinates API
  volcano           Smithsonian GVP Holocene volcanoes (RAW/gvp_holocene.csv), subaerial only
  landcover_sample  stratified random sample of analog-relevant land cover from
                    Copernicus C3S / CCI LCCS 2019 (JAXA Earth COG, level 1 = 0.025 deg)

Outputs
  data/patches/earth_candidates.csv
  data/catalogs/earth_impact_structures.csv
  data/catalogs/earth_volcanoes_gvp.csv

Usage:  python tools/build_earth_candidates.py [--seed 42] [--n-samples 3000]
"""
import argparse
import html
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract import GDAL_ENV  # noqa: E402

RAW = Path.home() / ".cache" / "earth_analog_raw"
OUT = ROOT / "data" / "patches"
CAT = ROOT / "data" / "catalogs"
R_KM = 6371.0088
MIN_SPACING_KM = 10.0

LCCS_BASE = ("https://s3.ap-northeast-1.wasabisys.com/je-pds/cog/v1/"
             "Copernicus.C3S_PROBA-V_LCCS_global_yearly/2019/1")
LCCS_TILES = [(lo, la) for lo in ("W180.00-W090.00", "W090.00-E000.00", "E000.00-E090.00", "E090.00-E180.00")
              for la in ("N00.00-N90.00", "S90.00-N00.00")]
LCCS_RES = 0.025
LCCS_NAMES = {140: "lichens_mosses", 150: "sparse_vegetation", 151: "sparse_tree", 152: "sparse_shrub",
              153: "sparse_herbaceous", 200: "bare", 201: "consolidated_bare", 202: "unconsolidated_bare",
              220: "permanent_snow_ice", 120: "shrubland", 121: "evergreen_shrubland",
              122: "deciduous_shrubland", 130: "grassland"}

# stratum -> target sample count (sum ~3000)
STRATA = {
    "bare_hot": 450,            # LCCS 200-202, |lat| < 35  (Sahara, Arabia, Atacama, Namib, Australia...)
    "bare_midlat": 400,         # LCCS 200-202, 35 <= |lat| < 60 (Gobi, Tibet, Great Basin, Patagonia...)
    "bare_polar": 250,          # LCCS 200-202, |lat| >= 60, outside Antarctica
    "sparse_hot": 300,          # LCCS 150-153 |lat| < 35
    "sparse_cold": 300,         # LCCS 150-153 |lat| >= 35
    "lichen_moss": 300,         # LCCS 140 (tundra, high Arctic)
    "ice_edge": 400,            # ice-free bare/sparse/tundra land within ~10 km of permanent ice (not Antarctica)
    "antarctic_ice_free": 450,  # any non-ice, non-water LCCS pixel south of 60 S (+ its ice-edge neighbourhood)
    "ice_sheet": 150,           # permanent ice interior (Concordia-like plateaus)
}


# ----------------------------------------------------------------------------- helpers
def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1) * R_KM


class SpacingHash:
    """Greedy minimum-spacing filter using a 3-D (ECEF km) grid hash."""

    def __init__(self, d_km):
        self.d = d_km
        self.cells = {}

    def _key(self, p):
        return tuple(np.floor(p / self.d).astype(int))

    def ok(self, p):
        k = self._key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in self.cells.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.sum((p - q) ** 2) < self.d ** 2:
                            return False
        return True

    def add(self, p):
        self.cells.setdefault(self._key(p), []).append(p)


# ----------------------------------------------------------------------------- named sites
def named_sites():
    df = pd.read_csv(ROOT / "config" / "earth_sites.csv")
    return pd.DataFrame(dict(
        patch_id=df["id"], lat=df["lat"], lon=df["lon"], source="named_site", name=df["name"],
        context=("tags=" + df["tags"].fillna("") + "; status=" + df["status"].fillna("")
                 + "; notes=" + df["notes"].fillna("") + "; country=" + df["country"].fillna("")),
        stratum="named_site"))


# ----------------------------------------------------------------------------- impact structures
def _txt(c):
    return html.unescape(re.sub(r"<[^>]+>", "", c)).replace("﻿", "").strip()


def _parse_geodec(s):
    m = re.search(r"([\d.]+)°([NS])\s+([\d.]+)°([EW])", s)
    if not m:
        return None
    lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
    lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
    return lat, lon


def _wiki_coords(titles):
    """Wikipedia API: primary coordinates for page titles (50 per request)."""
    out = {}
    s = requests.Session()
    s.headers["User-Agent"] = "space-apps-analog-finder/1.0 (research script)"
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        try:
            js = s.get("https://en.wikipedia.org/w/api.php", timeout=60, params=dict(
                action="query", prop="coordinates", titles="|".join(chunk), format="json",
                redirects=1, colimit="max")).json()
        except Exception as e:  # noqa: BLE001
            print("  wiki coords request failed:", e)
            continue
        q = js.get("query", {})
        back = {}
        for n in q.get("normalized", []) + q.get("redirects", []):
            back[n["to"]] = back.get(n["from"], n["from"])
        for p in q.get("pages", {}).values():
            if "coordinates" in p:
                c = p["coordinates"][0]
                t = p["title"]
                orig = back.get(t, t)
                orig = back.get(orig, orig)
                out[orig] = (c["lat"], c["lon"])
                out[t] = (c["lat"], c["lon"])
    return out


def impact_structures():
    t = json.loads((RAW / "wiki_craters.json").read_text(encoding="utf8"))["parse"]["text"]
    # split by section headers so we know the age bin
    sections = re.split(r'(<h[23] id="[^"]+")', t)
    recs = []
    cur = ""
    for part in sections:
        m = re.match(r'<h[23] id="([^"]+)"', part)
        if m:
            cur = m.group(1)
            continue
        if cur.startswith("Inferred") or cur.startswith("Statistics"):
            continue
        for tb in re.findall(r'<table class="wikitable.*?</table>', part, flags=re.S):
            rows = re.findall(r"<tr.*?</tr>", tb, flags=re.S)
            hdr = [_txt(c) for c in re.findall(r"<th.*?</th>", rows[0], flags=re.S)]
            if not hdr or hdr[0] != "Name":
                continue
            for r in rows[1:]:
                cells_raw = re.findall(r"<td.*?</td>", r, flags=re.S)
                if not cells_raw:
                    continue
                cells = [_txt(c) for c in cells_raw]
                a = re.search(r'<a href="/wiki/([^"#]+)"[^>]*title="([^"]+)"', cells_raw[0])
                title = html.unescape(a.group(2)) if a else cells[0]
                gd = re.search(r'class="geo-dec"[^>]*>(.*?)</span>', r, flags=re.S)
                ll = _parse_geodec(_txt(gd.group(1))) if gd else None
                rec = dict(name=cells[0], wiki_title=title, age_section=cur.replace("_", " "),
                           lat=ll[0] if ll else np.nan, lon=ll[1] if ll else np.nan,
                           coord_source="wiki_list" if ll else "")
                if len(cells) == len(hdr):  # aligned row (no rowspan shift)
                    for h, c in zip(hdr, cells):
                        if h.startswith("Location"):
                            rec["location"] = c
                        elif h.startswith("Country"):
                            rec["country"] = c
                        elif h.startswith("Diameter"):
                            rec["diameter_km_raw"] = c
                        elif h.startswith("Age"):
                            rec["age_raw"] = c
                else:
                    rec["row_raw"] = " | ".join(cells[1:-1])
                recs.append(rec)
    df = pd.DataFrame(recs)
    miss = df["lat"].isna()
    if miss.any():
        co = _wiki_coords(df.loc[miss, "wiki_title"].tolist())
        for i in df.index[miss]:
            c = co.get(df.at[i, "wiki_title"])
            if c:
                df.at[i, "lat"], df.at[i, "lon"] = c
                df.at[i, "coord_source"] = "wiki_page_api"
    for c in ("diameter_km_raw", "age_raw", "location", "country", "row_raw"):
        if c not in df:
            df[c] = ""
    df["diameter_km"] = pd.to_numeric(
        df["diameter_km_raw"].fillna("").str.replace(r"^0+(?=\d)", "", regex=True)
        .str.extract(r"([\d.]+)")[0], errors="coerce")
    df = df.drop_duplicates("wiki_title")
    CAT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAT / "earth_impact_structures.csv", index=False)
    print(f"impact structures: {len(df)} rows, {df['lat'].notna().sum()} with coordinates "
          f"({(df['coord_source'] == 'wiki_page_api').sum()} from page API)")
    d = df[df["lat"].notna()].copy()
    ctx = ("impact structure; diameter_km=" + d["diameter_km_raw"].fillna("").astype(str)
           + "; age=" + d["age_raw"].fillna("").astype(str) + " (" + d["age_section"] + ")"
           + "; location=" + d["location"].fillna("").astype(str) + ", " + d["country"].fillna("").astype(str)
           + d["row_raw"].fillna("").astype(str).map(lambda s: f"; row={s}" if s else ""))
    return pd.DataFrame(dict(patch_id="crater_" + d["name"].map(slug), lat=d["lat"], lon=d["lon"],
                             source="impact_structure", name=d["name"] + " impact structure",
                             context=ctx, stratum="impact_structure"))


# ----------------------------------------------------------------------------- volcanoes
def volcanoes():
    g = pd.read_csv(RAW / "gvp_holocene.csv")
    g.to_csv(CAT / "earth_volcanoes_gvp.csv", index=False)
    typ = g["Primary_Volcano_Type"].fillna("")
    land = ~typ.str.contains("Submarine", case=False) & (pd.to_numeric(g["Elevation"], errors="coerce") > 0)
    g = g[land & g["Latitude"].notna()].copy()
    print(f"volcanoes: {land.sum()} subaerial of {len(land)}")
    ctx = ("volcano; type=" + g["Primary_Volcano_Type"].fillna("") + "; landform=" + g["Volcanic_Landform"].fillna("")
           + "; rock=" + g["Major_Rock_Type"].fillna("") + "; tectonic=" + g["Tectonic_Setting"].fillna("")
           + "; last_eruption=" + g["Last_Eruption_Year"].astype(str) + "; elev_m=" + g["Elevation"].astype(str)
           + "; country=" + g["Country"].fillna("") + "; summary=" + g["Geological_Summary"].fillna("")
           .str.replace(r"\s+", " ", regex=True).str.slice(0, 600))
    vt = g["Primary_Volcano_Type"].fillna("other").str.lower()
    stratum = np.select([vt.str.contains("shield"), vt.str.contains("field|cone|maar|tuff|fissure|pyroclastic"),
                         vt.str.contains("caldera"), vt.str.contains("strato|complex|compound|dome")],
                        ["volcano_shield", "volcano_field_cone", "volcano_caldera", "volcano_strato_dome"],
                        "volcano_other")
    return pd.DataFrame(dict(patch_id="gvp_" + g["Volcano_Number"].astype(str), lat=g["Latitude"],
                             lon=g["Longitude"], source="volcano", name=g["Volcano_Name"], context=ctx,
                             stratum=stratum))


# ----------------------------------------------------------------------------- land cover sample
def _read_lccs_tile(spec):
    import rasterio
    lo, la = spec
    w = -180 if lo.startswith("W180") else (-90 if lo.startswith("W090") else (0 if lo.startswith("E000") else 90))
    s, n = (0, 90) if la.startswith("N00") else (-90, 0)
    fname = f"{lo.split('-')[0]}-{la.split('-')[0]}-{lo.split('-')[1]}-{la.split('-')[1]}-LCCS.tiff"
    url = f"{LCCS_BASE}/{lo}/{fname}"
    env = dict(GDAL_ENV, CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff")
    for attempt in range(4):
        try:
            with rasterio.Env(**env):
                with rasterio.open("/vsicurl/" + url) as src:
                    a = src.read(1)
                    tr = src.transform
            assert abs(tr.c - w) < 1e-6 and abs(tr.f - n) < 1e-6, (tr, w, n)
            return spec, a
        except Exception as e:  # noqa: BLE001
            print("  retry", url, e)
    raise RuntimeError(f"failed {url}")


def load_lccs():
    cache = RAW / "lccs2019_0p025.npy"
    if cache.exists():
        return np.load(cache)
    g = np.zeros((int(180 / LCCS_RES), int(360 / LCCS_RES)), dtype=np.uint8)
    with ThreadPoolExecutor(8) as ex:
        for spec, a in ex.map(_read_lccs_tile, LCCS_TILES):
            lo, la = spec
            c0 = {"W180": 0, "W090": 1, "E000": 2, "E090": 3}[lo[:4]] * a.shape[1]
            r0 = 0 if la.startswith("N00") else a.shape[0]
            g[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
    np.save(cache, g)
    return g


ANT_WMS = ("https://maps.bas.ac.uk/antarctic/wms?service=WMS&version=1.3.0&request=GetMap"
           "&layers=add:antarctic_landsat_rock&styles=&crs=EPSG:3031&bbox={b}&width={n}&height={n}"
           "&format=image/png&transparent=true")


def antarctic_rock_points(n_px=2000, half=2800000):
    """SCAR ADD Landsat rock outcrop (Burton-Johnson et al. 2016) rendered via BAS WMS in EPSG:3031
    at ~1.4 km/px; returns lat/lon of every rock pixel centre."""
    import rasterio
    from rasterio.warp import transform
    d = RAW / "ant_rock"
    d.mkdir(parents=True, exist_ok=True)
    lats, lons = [], []
    res = half / n_px
    for ix, x0 in enumerate((-half, 0)):
        for iy, y0 in enumerate((-half, 0)):
            f = d / f"rock_{ix}{iy}.png"
            if not f.exists():
                b = f"{x0},{y0},{x0 + half},{y0 + half}"
                r = requests.get(ANT_WMS.format(b=b, n=n_px), timeout=300,
                                 headers={"User-Agent": "space-apps-analog-finder/1.0"})
                r.raise_for_status()
                assert r.headers.get("content-type", "").startswith("image/png"), r.text[:300]
                f.write_bytes(r.content)
            with rasterio.open(f) as src:
                alpha = src.read(4) if src.count >= 4 else src.read(1)
            rr, cc = np.nonzero(alpha > 0)
            xs = x0 + (cc + 0.5) * res
            ys = (y0 + half) - (rr + 0.5) * res
            lo, la = transform("EPSG:3031", "EPSG:4326", xs.tolist(), ys.tolist())
            lats += la
            lons += lo
    return np.array(lats), np.array(lons)


def landcover_samples(existing, n_total, seed):
    from scipy import ndimage
    lc = load_lccs()
    H, W = lc.shape
    lat_c = 90 - (np.arange(H) + 0.5) * LCCS_RES
    lat2 = np.broadcast_to(lat_c[:, None], (H, W))
    print("LCCS loaded", lc.shape, "classes present:", np.unique(lc)[:40])

    ice = lc == 220
    water = (lc == 210) | (lc == 0)
    bare = (lc >= 200) & (lc <= 202)
    sparse = (lc >= 150) & (lc <= 153)
    lich = lc == 140
    open_land = bare | sparse | lich | ((lc >= 120) & (lc <= 130))
    near_ice = ndimage.binary_dilation(ice, structure=np.ones((3, 3), bool), iterations=4)  # ~ +-4 px (~11 km N-S)
    ice_interior = ice & ~ndimage.binary_dilation(~ice & ~water, structure=np.ones((3, 3), bool), iterations=20)
    antarc = lat2 < -60
    alat = np.abs(lat2)

    masks = {
        "antarctic_ice_free": antarc & ~ice & ~water,
        "ice_edge": ~antarc & open_land & near_ice & ~ice,
        "lichen_moss": ~antarc & lich & ~near_ice,
        "bare_polar": ~antarc & bare & (alat >= 60) & ~near_ice,
        "bare_hot": bare & (alat < 35) & ~near_ice,
        "bare_midlat": bare & (alat >= 35) & (alat < 60) & ~near_ice,
        "sparse_hot": sparse & (alat < 35) & ~near_ice,
        "sparse_cold": ~antarc & sparse & (alat >= 35) & ~near_ice,
        "ice_sheet": ice_interior,
    }
    scale = n_total / sum(STRATA.values())
    rng = np.random.default_rng(seed)
    hsh = SpacingHash(MIN_SPACING_KM)
    for p in xyz(existing["lat"].to_numpy(float), existing["lon"].to_numpy(float)):
        hsh.add(p)
    rows = []
    for st, m in masks.items():
        want = int(round(STRATA[st] * scale))
        if st == "antarctic_ice_free":
            alat, alon = antarctic_rock_points()
            print(f"  stratum {st:<20} rock pixels={alat.size:>9}  target={want}")
            got = 0
            for i in rng.permutation(alat.size):
                p = xyz(alat[i], alon[i])
                if not hsh.ok(p):
                    continue
                hsh.add(p)
                got += 1
                rows.append(dict(patch_id=f"lc_{st}_{got:04d}", lat=round(float(alat[i]), 5),
                                 lon=round(float(alon[i]), 5), source="landcover_sample",
                                 name=f"{st} sample {got}",
                                 context="SCAR ADD Landsat rock outcrop (ice-free) pixel, ~1.4 km; "
                                         f"stratum={st}", stratum=st))
                if got >= want:
                    break
            print(f"    -> {got}")
            continue
        rr, cc = np.nonzero(m)
        print(f"  stratum {st:<20} pixels={rr.size:>9}  target={want}")
        if rr.size == 0:
            continue
        # area weighting: pixel area ~ cos(lat); sample with prob ~ cos(lat) so samples are uniform in area
        wts = np.cos(np.radians(lat_c[rr]))
        wts = wts / wts.sum()
        k = min(rr.size, want * 6)
        pick = rng.choice(rr.size, size=k, replace=False, p=wts)
        got = 0
        for i in pick:
            r, c = rr[i], cc[i]
            la = 90 - (r + rng.random()) * LCCS_RES
            lo = -180 + (c + rng.random()) * LCCS_RES
            p = xyz(la, lo)
            if not hsh.ok(p):
                continue
            hsh.add(p)
            got += 1
            cls = int(lc[r, c])
            rows.append(dict(patch_id=f"lc_{st}_{got:04d}", lat=round(la, 5), lon=round(lo, 5),
                             source="landcover_sample", name=f"{st} sample {got}",
                             context=f"LCCS2019 class {cls} ({LCCS_NAMES.get(cls, 'other')}) at 0.025 deg; "
                                     f"stratum={st}", stratum=st))
            if got >= want:
                break
        print(f"    -> {got}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=3000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CAT.mkdir(parents=True, exist_ok=True)
    base = pd.concat([named_sites(), impact_structures(), volcanoes()], ignore_index=True)
    base["lon"] = (((base["lon"] + 180) % 360) - 180).round(6)
    base["lat"] = base["lat"].astype(float).round(6)
    # unique patch ids
    dup = base["patch_id"].duplicated(keep=False)
    if dup.any():
        base.loc[dup, "patch_id"] = base.loc[dup, "patch_id"] + "_" + base[dup].groupby("patch_id").cumcount().astype(str)
    samp = landcover_samples(base, a.n_samples, a.seed)
    df = pd.concat([base, samp], ignore_index=True)
    assert df["patch_id"].is_unique
    df.to_csv(OUT / "earth_candidates.csv", index=False)
    print(f"\nwrote {OUT / 'earth_candidates.csv'}: {len(df)} rows")
    print(df.groupby(["source"]).size().to_string())
    print(df.groupby(["stratum"]).size().to_string())


if __name__ == "__main__":
    main()
