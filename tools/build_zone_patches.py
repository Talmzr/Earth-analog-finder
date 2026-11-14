#!/usr/bin/env python
"""
Terrain image patch + window terrain metrics for each Artemis zone (data/ml/moon_zones.csv).

A zone (landing / ice) is an irregular set of pixels inside a 15 km region box, so it has no tile image.
For the terrain encoder each zone gets the 10 x 10 km window centred on the ZONE CENTROID, cut exactly like
the Moon tiles (LOLA 80 m DEM -> 128 x 128 at 78 m/px, elevation minus mean, slope in degrees).

Outputs (small, committed to the repo):
  data/patches/moon_zone_arrays_128.npz   patch_id = "<region_id>__<zone>", elev, slope (float16)
  data/ml/moon_zone_patch_terrain.csv     terrain_metrics of the same 10 km window (relief_m, slope500_*, ...)
Needs the local LOLA DEM (~/.cache/earth_analog_raw/LDEM_80S_80MPP_ADJ.TIF; downloaded if missing).
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import extract as base  # noqa: E402

RAW = Path(os.environ.get("ANALOG_RAW_DIR", Path.home() / ".cache" / "earth_analog_raw"))
DEM = RAW / "LDEM_80S_80MPP_ADJ.TIF"
DEM_URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF"
R_MOON, PATCH_M, N = 1737400.0, 10_000.0, 128


def stereo_south(lat, lon):
    rho = 2 * R_MOON * math.tan(math.radians(90 + lat) / 2)
    return rho * math.sin(math.radians(lon)), rho * math.cos(math.radians(lon))


def main():
    if not DEM.exists():
        from pdownload import main as pdl
        RAW.mkdir(parents=True, exist_ok=True)
        pdl(DEM_URL, str(DEM), 12)
    z = pd.read_csv(ROOT / "data/ml/moon_zones.csv")
    ids, E, S, rows = [], [], [], []
    with rasterio.open(DEM) as dem:
        sc, of = (dem.scales or (1,))[0] or 1, (dem.offsets or (0,))[0] or 0
        for r in z.itertuples(index=False):
            pid = f"{r.region_id}__{r.zone}"
            x, y = stereo_south(r.zone_centroid_lat, r.zone_centroid_lon)
            b = (x - PATCH_M / 2, y - PATCH_M / 2, x + PATCH_M / 2, y + PATCH_M / 2)
            win = from_bounds(*b, transform=dem.transform)

            def read(shape, resampling):
                a = dem.read(1, window=win, out_shape=shape, resampling=resampling, boundless=True,
                             fill_value=np.nan, out_dtype="float64")
                if dem.nodata is not None and np.isfinite(dem.nodata):
                    a[a == dem.nodata] = np.nan
                a = a * sc + of
                return a * 1000 if np.nanmax(np.abs(a)) < 20 else a

            gscale = math.cos(math.radians(90.0 + r.zone_centroid_lat) / 2) ** 2
            zn = read((125, 125), Resampling.nearest)                 # native 80 m window for the metrics
            rec = dict(entity_id=pid, region_id=r.region_id, zone=r.zone)
            rec.update({f"patch_{k}": v for k, v in base.terrain_metrics(zn, 80.0 * gscale, 80.0 * gscale).items()})
            za = read((N, N), Resampling.bilinear)                    # same recipe as tools/build_arrays_128.py
            px = PATCH_M / N * gscale
            s = base._slope_deg(za, px, px)
            za = za - np.nanmean(za)
            ids.append(pid); E.append(np.nan_to_num(za)); S.append(np.nan_to_num(s)); rows.append(rec)
    np.savez_compressed(ROOT / "data/patches/moon_zone_arrays_128.npz", patch_id=np.asarray(ids, dtype="U48"),
                        center_lat=z.zone_centroid_lat.values.astype(np.float32),
                        center_lon=z.zone_centroid_lon.values.astype(np.float32),
                        elev=np.asarray(E, np.float16), slope=np.asarray(S, np.float16))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/ml/moon_zone_patch_terrain.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[["entity_id", "patch_relief_m", "patch_slope500_mean_deg", "patch_frac_slope500_gt10", "patch_tri500_mean_m"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
