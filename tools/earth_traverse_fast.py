#!/usr/bin/env python
"""
Fast, fully local Earth "landing -> ice traverse" feature (Earth counterpart of the Moon landing/ice zones).
Replaces the slow remote 30 m version (tools/earth_traverse.py) so the dataset can be completed quickly.

Per Earth tile (10 x 10 km, window +-10 km for ice):
  flat share      : slope < 8 deg from data/patches/earth_patch_arrays.npz (64 x 64, ~156 m)
  bare share      : ESA WorldCover bare/sparse + moss/lichen fractions (tile), else coarse grid share
  ice / bare grid : Copernicus C3S land cover 2019 at 0.025 deg (~2.8 km): 220 = permanent snow/ice,
                    140/150/152/153/200-202 = lichen, sparse veg, bare. Antarctica (lat <= -60): 1.4 km rock-
                    outcrop mask (EPSG:3031) -> rock = bare, rest of land = ice.
Outputs data/patches/earth_traverse.csv with
  trav_flat_bare_frac, trav_ice_frac, trav_dist_flat_to_ice_km (median), trav_dist_min_flat_to_ice_km,
  trav_dist_gt10_flag, trav_valid_frac, trav_source, trav_method
Distances are resolved only to ~1-3 km (grid size): use them as "ice within a few km?" not as exact metres.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rtransform
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.path.expanduser("~/.cache/earth_analog_raw"))
R = 6371008.8
HALF_KM, STEP_KM = 10.0, 1.0           # +-10 km window sampled on a 1 km lattice (21 x 21)
BARE = np.array([140, 150, 152, 153, 200, 201, 202])
ICE = 220


def ant_mask():
    M = np.zeros((4000, 4000), bool)
    for ix in (0, 1):
        for iy in (0, 1):
            with rasterio.open(RAW / "ant_rock" / f"rock_{ix}{iy}.png") as s:
                a = s.read(4) if s.count >= 4 else s.read(1)
            M[(1 - iy) * 2000:(2 - iy) * 2000, ix * 2000:(ix + 1) * 2000] = a > 0
    return M


def lattice(lat, lon):
    k = np.arange(-HALF_KM, HALF_KM + 1e-9, STEP_KM)
    dlat = np.degrees(k * 1000 / R)
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-3)
    LAT, LON = np.meshgrid(lat + dlat[::-1], lon + dlon)
    return LAT.T, ((LON.T + 180) % 360) - 180, k


def main():
    lc = np.load(RAW / "lccs2019_0p025.npy", mmap_mode="r")
    ant = ant_mask()
    t = pd.read_csv(ROOT / "data/patches/earth_patches_terrain.csv")
    z = np.load(ROOT / "data/patches/earth_patch_arrays.npz")
    sidx = {p: i for i, p in enumerate(z["patch_id"].astype(str))}
    slope, valid = z["slope"], (z["valid"] if "valid" in z.files else None)
    rows = []
    for r in t.itertuples(index=False):
        lat, lon = r.center_lat, r.center_lon
        LAT, LON, k = lattice(lat, lon)
        n = len(k)
        cen = np.zeros((n, n), bool)
        c0 = int(np.argmin(np.abs(k + 5))); c1 = int(np.argmin(np.abs(k - 5))) + 1
        cen[c0:c1, c0:c1] = True
        if lat <= -60:
            x, y = rtransform("EPSG:4326", "EPSG:3031", LON.ravel().tolist(), LAT.ravel().tolist())
            c = np.floor((np.array(x) + 2800000) / 1400).astype(int)
            rr = np.floor((2800000 - np.array(y)) / 1400).astype(int)
            ok = (c >= 0) & (c < 4000) & (rr >= 0) & (rr < 4000)
            rock = np.zeros(n * n, bool)
            rock[ok] = ant[rr[ok], c[ok]]
            bare, ice = rock.reshape(n, n), ~rock.reshape(n, n)
            src = "antarctic_rock_1.4km"
        else:
            ri = np.clip(((90 - LAT) / 0.025).astype(int), 0, lc.shape[0] - 1)
            ci = np.clip(((LON + 180) / 0.025).astype(int), 0, lc.shape[1] - 1)
            cls = np.asarray(lc[ri, ci])
            bare, ice = np.isin(cls, BARE), cls == ICE
            src = "c3s_lccs_2.8km"
        # flat share of the 10 km tile from the 64 x 64 slope array
        i = sidx.get(r.patch_id)
        flat = np.nan
        if i is not None:
            s = slope[i].astype(float)
            m = np.isfinite(s) if valid is None else (valid[i] > 0) & np.isfinite(s)
            flat = float((s[m] < 8).mean()) if m.any() else np.nan
        wc = getattr(r, "wc_frac_bare_sparse", np.nan)
        bare_share = (wc + getattr(r, "wc_frac_moss_lichen", 0.0)) if pd.notna(wc) else float(bare[cen].mean())
        res = dict(patch_id=r.patch_id, trav_flat_bare_frac=flat * bare_share if pd.notna(flat) else np.nan,
                   trav_ice_frac=float(ice.mean()), trav_valid_frac=1.0, trav_source=src, trav_method="fast_local",
                   trav_dist_flat_to_ice_km=np.nan, trav_dist_min_flat_to_ice_km=np.nan, trav_dist_gt10_flag=0)
        start = bare & cen
        if not start.any() and bare_share > 0.05:
            start = cen & ~ice                 # bare ground exists at sub-grid scale: start from any non-ice cell
        if start.any():
            if ice.any():
                d = ndimage.distance_transform_edt(~ice) * STEP_KM
                dd = d[start]
                res.update(trav_dist_flat_to_ice_km=float(min(np.median(dd), 10.0)),
                           trav_dist_min_flat_to_ice_km=float(min(dd.min(), 10.0)))
            else:
                res.update(trav_dist_flat_to_ice_km=10.0, trav_dist_min_flat_to_ice_km=10.0, trav_dist_gt10_flag=1)
        rows.append(res)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/patches/earth_traverse.csv", index=False)
    d = out.trav_dist_flat_to_ice_km
    print(f"tiles: {len(out)}   with a distance: {d.notna().sum()}   ice within 3 km: {(d <= 3).sum()}   "
          f"no ice within 10 km: {(out.trav_dist_gt10_flag == 1).sum()}")
    print(out.trav_source.value_counts().to_dict())
    chk = ["dry_valleys_wright", "beacon_valley", "haughton", "ellesmere_axel_heiberg", "svalbard_bockfjorden",
           "concordia", "kamil", "atacama_yungay", "meteor_crater", "tso_kar_ladakh"]
    print(out[out.patch_id.isin(chk)].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
