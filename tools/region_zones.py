#!/usr/bin/env python
"""
What is inside each Artemis region? Pixel-level (80 m) zone analysis within R km of each anchor:
  landing zone = slope < 8 deg (80 m DEM) AND average sun visibility >= 0.3
  ice zone     = permanently shadowed (average sun visibility == 0)
plus the distance from landing ground to the nearest shadowed ground, and Diviner temperatures per zone.

Inputs (local, from the patch build): RAW/LDEM_80S_80MPP_ADJ.TIF, RAW/AVGVISIB_75S_120M_201608.TIF,
RAW/dlre_prp_south.npz ; config/targets.csv ; data/patches/moon_patches.csv (single-tile values for contrast)
Output: data/patches/artemis_region_zones.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.path.expanduser("~/.cache/earth_analog_raw"))
R_MOON = 1737400.0
RADIUS = float(sys.argv[1]) * 1000 if len(sys.argv) > 1 else 15000.0
SLOPE_MAX, VIS_MIN = 8.0, 0.3


def stereo_south(lat, lon):
    rho = 2 * R_MOON * np.tan(np.radians(90 + np.asarray(lat, float)) / 2)
    lam = np.radians(np.asarray(lon, float))
    return rho * np.sin(lam), rho * np.cos(lam)


def main():
    from scipy.ndimage import distance_transform_edt
    t = pd.read_csv(ROOT / "config/targets.csv")
    t = t[(t.body == "moon") & (t.lat <= -80)]
    tiles = pd.read_csv(ROOT / "data/patches/moon_patches.csv").set_index("patch_id")
    z = np.load(RAW / "dlre_prp_south.npz")
    tx, ty = stereo_south(z["lat"], z["lon"])
    t_tmax, t_tavg, t_ice = z["tmax"].astype(float), z["tavg"].astype(float), z["ice"].astype(float)
    t_ice = np.where(t_ice == -1, 0.0, t_ice)
    rows = []
    with rasterio.open(RAW / "LDEM_80S_80MPP_ADJ.TIF") as dem, rasterio.open(RAW / "AVGVISIB_75S_120M_201608.TIF") as vis:
        for _, r in t.iterrows():
            cx, cy = stereo_south(r.lat, r.lon)
            b = (cx - RADIUS - 500, cy - RADIUS - 500, cx + RADIUS + 500, cy + RADIUS + 500)
            w = from_bounds(*b, transform=dem.transform).round_offsets().round_lengths()
            h = dem.read(1, window=w, out_dtype="float64")
            if dem.nodata is not None:
                h[h == dem.nodata] = np.nan
            h = h * (dem.scales[0] or 1) + (dem.offsets[0] or 0)
            if np.nanmax(np.abs(h)) < 20:      # km -> m if the product stores km
                h *= 1000
            res = abs(dem.transform.a)
            gy, gx = np.gradient(h, res, res)
            slope = np.degrees(np.arctan(np.hypot(gx, gy)))
            # pixel centres of the DEM window in projected metres
            wt = dem.window_transform(w)
            cols, rws = np.meshgrid(np.arange(h.shape[1]) + 0.5, np.arange(h.shape[0]) + 0.5)
            X, Y = wt * (cols, rws)
            vr, vc = rasterio.transform.rowcol(vis.transform, X.ravel(), Y.ravel())
            vr, vc = np.asarray(vr), np.asarray(vc)
            vwin = rasterio.windows.Window(vc.min(), vr.min(), vc.max() - vc.min() + 1, vr.max() - vr.min() + 1)
            va = vis.read(1, window=vwin, out_dtype="float64")
            if vis.nodata is not None:
                va[va == vis.nodata] = np.nan
            va = va * (vis.scales[0] or 1) + (vis.offsets[0] or 0)
            V = va[vr - vr.min(), vc - vc.min()].reshape(h.shape)
            inside = np.hypot(X - cx, Y - cy) <= RADIUS
            px_km2 = (res / 1000) ** 2
            land = inside & (slope < SLOPE_MAX) & (V >= VIS_MIN)
            land50 = inside & (slope < SLOPE_MAX) & (V >= 0.5)
            psr = inside & (V <= 0)
            dist = distance_transform_edt(~(V <= 0), sampling=res) / 1000   # km to nearest PSR pixel
            # Diviner triangles inside the circle, classified by the zone of their pixel
            sel = np.hypot(tx - cx, ty - cy) <= RADIUS
            ti = ((ty[sel] - wt.f) / wt.e).astype(int), ((tx[sel] - wt.c) / wt.a).astype(int)
            ok = (ti[0] >= 0) & (ti[0] < h.shape[0]) & (ti[1] >= 0) & (ti[1] < h.shape[1])
            zl = np.zeros(sel.sum(), bool)
            zp = np.zeros(sel.sum(), bool)
            zl[ok] = land[ti[0][ok], ti[1][ok]]
            zp[ok] = psr[ti[0][ok], ti[1][ok]]
            tm, ta, ic = t_tmax[sel], t_tavg[sel], t_ice[sel]
            tile = tiles.loc[r.id] if r.id in tiles.index else None
            rows.append(dict(
                target=r.id, name=r["name"], radius_km=RADIUS / 1000,
                area_km2=round(inside.sum() * px_km2, 1),
                landing_km2=round(land.sum() * px_km2, 1), landing_vis50_km2=round(land50.sum() * px_km2, 1),
                psr_km2=round(psr.sum() * px_km2, 1),
                best_landing_vis=round(float(np.nanmax(np.where(land, V, np.nan))), 2) if land.any() else np.nan,
                min_km_landing_to_psr=round(float(dist[land].min()), 2) if land.any() and psr.any() else np.nan,
                median_km_landing_to_psr=round(float(np.median(dist[land])), 2) if land.any() and psr.any() else np.nan,
                landing_tmax_k=round(float(np.nanmean(tm[zl])), 0) if zl.any() else np.nan,
                landing_tavg_k=round(float(np.nanmean(ta[zl])), 0) if zl.any() else np.nan,
                psr_tmax_k=round(float(np.nanmean(tm[zp])), 0) if zp.any() else np.nan,
                psr_tavg_k=round(float(np.nanmean(ta[zp])), 0) if zp.any() else np.nan,
                psr_ice_stable_frac=round(float(np.mean(ic[zp] >= 0)), 2) if zp.any() else np.nan,
                single_tile_psr_frac=round(float(tile.lola_frac_psr), 2) if tile is not None else np.nan,
                single_tile_sunvis_max=round(float(tile.lola_avg_sun_vis_max), 2) if tile is not None else np.nan,
            ))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/patches/artemis_region_zones.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["name"]).to_string(index=False))


if __name__ == "__main__":
    main()
