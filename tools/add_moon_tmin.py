#!/usr/bin/env python
"""
Add day/night temperature extremes to the Moon patches from the Diviner Global Cumulative Product
(80-90S slice: 0.5 deg cells x 96 local-time bins of bolometric temperature).
The Polar Resource Product used in moon_patches.csv has only average + max, so this supplies the
MINIMUM (night) temperature and the full diurnal swing.

Input : ~/.cache/earth_analog_raw/global_cumul_avg_cyl_90s80s_002.tab  (156 MB, PDS)
        data/patches/moon_patches.csv (patch_id, center_lat, center_lon, x_m, y_m)
Output: data/patches/moon_patches_tmin.csv  (patch_id, dvn_tbol_min_k, dvn_tbol_max_k, dvn_tbol_mean_k,
        dvn_tbol_noon_k, dvn_tbol_midnight_k, dvn_tbol_ltime_cov, dvn_tbol_n_cells, dvn_tbol_nearest_km)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.path.expanduser("~/.cache/earth_analog_raw"))
R = 1737400.0
PATCH = 10000.0


def stereo_south(lat, lon):
    """Spherical south polar stereographic, true scale at the pole (LOLA polar products)."""
    rho = 2 * R * np.tan(np.radians(90 + np.asarray(lat)) / 2)
    lam = np.radians(np.asarray(lon))
    return rho * np.sin(lam), rho * np.cos(lam)


def main():
    m = pd.read_csv(ROOT / "data/patches/moon_patches.csv", usecols=["patch_id", "center_lat", "center_lon", "x_m", "y_m", "is_target"])
    # verify the projection convention against the patch table (sign of y can differ by product)
    px, py = stereo_south(m.center_lat.values, m.center_lon.values)
    ysign = 1.0
    if np.nanmedian(np.abs(py - m.y_m)) > np.nanmedian(np.abs(-py - m.y_m)):
        ysign = -1.0
    err = np.nanmedian(np.hypot(px - m.x_m, ysign * py - m.y_m))
    print(f"projection check: median |xy error| = {err:.0f} m (y sign {ysign:+.0f})")

    tabs = [RAW / f for f in ("global_cumul_avg_cyl_90s80s_002.tab", "global_cumul_avg_cyl_80s70s_002.tab")]
    tabs = [t for t in tabs if t.exists()]
    print("slices:", [t.name for t in tabs])
    d = pd.concat([pd.read_csv(t, skipinitialspace=True, usecols=["clon", "clat", "ltim", "tbol"], dtype="float32")
                   for t in tabs], ignore_index=True)
    d.loc[d.tbol < 0, "tbol"] = np.nan
    g = d.groupby(["clon", "clat"])
    cells = pd.DataFrame({
        "tmin": g.tbol.min(), "tmax": g.tbol.max(), "tmean": g.tbol.mean(), "cov": g.tbol.apply(lambda s: s.notna().mean()),
        "noon": d[(d.ltim > 11) & (d.ltim < 13)].groupby(["clon", "clat"]).tbol.mean(),
        "midnight": d[(d.ltim < 1) | (d.ltim > 23)].groupby(["clon", "clat"]).tbol.mean(),
    }).reset_index()
    cells = cells[cells["cov"] > 0.25]
    print(f"Diviner GCP cells with data: {len(cells)}")
    cx, cy = stereo_south(cells.clat.values, cells.clon.values)
    cy = ysign * cy

    # each patch: mean over cells whose centre falls in the 10 x 10 km square; else nearest cell
    out = []
    for _, p in m.iterrows():
        inside = (np.abs(cx - p.x_m) <= PATCH / 2) & (np.abs(cy - p.y_m) <= PATCH / 2)
        if p.is_target == 1 or not inside.any():
            dist = np.hypot(cx - p.x_m, cy - p.y_m)
            inside = dist <= max(PATCH / 2, dist.min() + 1)  # nearest cell(s)
            near = dist.min() / 1000
        else:
            near = 0.0
        c = cells[inside]
        out.append(dict(patch_id=p.patch_id, dvn_tbol_min_k=c.tmin.mean(), dvn_tbol_max_k=c.tmax.mean(),
                        dvn_tbol_mean_k=c.tmean.mean(), dvn_tbol_noon_k=c.noon.mean(),
                        dvn_tbol_midnight_k=c.midnight.mean(), dvn_tbol_ltime_cov=c["cov"].mean(),
                        dvn_tbol_n_cells=int(inside.sum()), dvn_tbol_nearest_km=round(near, 2)))
    o = pd.DataFrame(out)
    o.to_csv(ROOT / "data/patches/moon_patches_tmin.csv", index=False)
    print(o.describe().T[["count", "min", "50%", "max"]].round(1).to_string())
    print(o[o.patch_id.str.startswith("moon_")].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
