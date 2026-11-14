#!/usr/bin/env python
"""
Artemis region zones v2: place a 15 x 15 km box per region, split it into zones at 80 m and describe each zone
with the SAME feature names as data/ml/moon_tiles.csv so the tile models can score zones like tiles.

Region placement (placement_method):
  official           box centred on NASA's Artemis III candidate-region outline from NASA Moon Trek
                     (layer "Artemis III Candidate Landing Regions", A3_Named_regions, Aug 2022; the region name is
                     kept in NASA's Oct 2024 list of 9). The 2022 outlines are 16-21 km squares (Haworth larger);
                     the 2024 regions are ~15 x 15 km, so we use a 15 x 15 km box on the outline centre.
  criteria_fallback  crater-centre anchor without an official outline: search centres in a ring 0.6-1.6 crater
                     radii (radius = rim crest of the azimuthal-median DEM profile) for the 15 x 15 km box with the
                     most sunlit flat ground (slope < 8 deg, vis >= 0.3) lying within 2 km of permanent shadow.
  anchor             feature-anchored region (massif / plateau), box centred on the config/targets.csv anchor.

Zones inside the box (80 m LOLA grid; visibility 120 m resampled nearest-neighbour):
  landing = slope(80 m) < 8 deg AND average sun visibility >= 0.3   (landing_vis50_km2 also reported at >= 0.5)
  ice     = permanently shadowed (average sun visibility == 0)
Per-zone features: terrain_metrics-style columns computed over zone pixels (500 m baseline slope/TRI computed on the
full box DEM with extract.py's helpers, then sampled at zone pixels), Diviner tmax/tavg/ice-depth triangles whose
centre falls in a zone pixel, LOLA visibility stats, latitude-only illumination at the zone centroid.
x_temp_min_k / x_temp_swing_k: pixel-weighted 10 km tile value (data/patches/moon_patches_tmin.csv, Diviner GCP);
NOT zone-resolved (landing and ice zones of one region get nearly the same value).

Inputs: RAW/LDEM_80S_80MPP_ADJ.TIF, RAW/AVGVISIB_75S_120M_201608.TIF, RAW/dlre_prp_south.npz, config/targets.csv,
        data/external/artemis3_regions_2022_moontrek.geojson (downloaded from TREK_URL if missing),
        data/patches/moon_patches.csv + moon_patches_tmin.csv
Outputs: data/ml/moon_zones.csv (region x zone), data/patches/artemis_region_zones.csv (region summary),
         data/patches/artemis_region_boxes.geojson (the boxes used)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from extract import SLOPE_BASELINE_M, _block_mean, _slope_deg, _tri, illumination  # noqa: E402
from build_ml_tables import grazing_sun_frac  # noqa: E402

RAW = Path(os.path.expanduser("~/.cache/earth_analog_raw"))
R_MOON = 1737400.0
HALF = 7500.0                    # 15 x 15 km box
MARGIN = 3000.0                  # extra DEM read around the box (500 m blocks, edges)
SLOPE_MAX, VIS_MIN, VIS_GOOD = 8.0, 0.3, 0.5
NEAR_PSR_KM, RING = 2.0, (0.6, 1.6)
TREK_URL = "https://trek.nasa.gov/moon/TrekWS/rest/cat/data/stream?uuid=82d31599-9957-4d79-b30a-720c918d1515"
TREK_FILE = ROOT / "data/external/artemis3_regions_2022_moontrek.geojson"
SRC_OFFICIAL = f"NASA Moon Trek layer 'Artemis III Candidate Landing Regions' (A3_Named_regions, 2022 outline; name kept in NASA Oct 2024 list) {TREK_URL}"
SRC_TARGETS = "config/targets.csv anchor"

# region_id, region_name, anchor target id, placement_method, Moon Trek label
REGIONS = [
    ("shackleton_rim", "Shackleton rim (not an Artemis III region)", "moon_shackleton", "criteria_fallback", None),
    ("peak_near_cabeus_b", "Peak near Cabeus B", "moon_peak_cabeus_b", "criteria_fallback", None),
    ("haworth", "Haworth", "moon_haworth", "official", "Haworth"),
    ("malapert_massif", "Malapert Massif", "moon_malapert_massif", "official", "Malapert Massif"),
    ("mons_mouton", "Mons Mouton", "moon_mons_mouton", "anchor", None),
    ("mons_mouton_plateau", "Mons Mouton Plateau", "moon_mons_mouton_plateau", "anchor", None),
    ("nobile_rim_1", "Nobile Rim 1", "moon_nobile_rim", "official", "Nobile Rim 1"),
    ("nobile_rim_2", "Nobile Rim 2", "moon_nobile_rim", "official", "Nobile Rim 2"),
    ("de_gerlache_rim_2", "de Gerlache Rim 2", "moon_de_gerlache_rim2", "official", "de Gerlache Rim 2"),
    ("slater_plain", "Slater Plain", "moon_slater_plain", "criteria_fallback", None),
]


def stereo_south(lat, lon):
    rho = 2 * R_MOON * np.tan(np.radians(90 + np.asarray(lat, float)) / 2)
    lam = np.radians(np.asarray(lon, float))
    return rho * np.sin(lam), rho * np.cos(lam)


def inv_stereo_south(x, y):
    rho = np.hypot(x, y)
    return -90 + 2 * np.degrees(np.arctan(rho / (2 * R_MOON))), np.degrees(np.arctan2(x, y))


def trek_regions():
    if not TREK_FILE.exists():
        import requests
        TREK_FILE.parent.mkdir(parents=True, exist_ok=True)
        TREK_FILE.write_bytes(requests.get(TREK_URL, timeout=60).content)
    out = {}
    for f in json.loads(TREK_FILE.read_text())["features"]:
        c = np.asarray(f["geometry"]["coordinates"][0], float)[:-1]
        x, y = stereo_south(c[:, 1], c[:, 0])
        out[f["properties"]["label"]] = dict(cx=x.mean(), cy=y.mean(), w_km=(x.max() - x.min()) / 1000,
                                             h_km=(y.max() - y.min()) / 1000)
    return out


class Rasters:
    def __init__(self):
        self.dem = rasterio.open(RAW / "LDEM_80S_80MPP_ADJ.TIF")
        self.vis = rasterio.open(RAW / "AVGVISIB_75S_120M_201608.TIF")
        self.res = abs(self.dem.transform.a)

    def read(self, cx, cy, half):
        dem, vis = self.dem, self.vis
        w = from_bounds(cx - half, cy - half, cx + half, cy + half, transform=dem.transform).round_offsets().round_lengths()
        h = dem.read(1, window=w, out_dtype="float64")
        if dem.nodata is not None:
            h[h == dem.nodata] = np.nan
        h = h * (dem.scales[0] or 1) + (dem.offsets[0] or 0)
        if np.nanmax(np.abs(h)) < 20:          # km -> m if the product stores km
            h *= 1000
        wt = dem.window_transform(w)
        cols, rws = np.meshgrid(np.arange(h.shape[1]) + 0.5, np.arange(h.shape[0]) + 0.5)
        X, Y = wt * (cols, rws)
        vr, vc = (np.asarray(a) for a in rasterio.transform.rowcol(vis.transform, X.ravel(), Y.ravel()))
        vwin = rasterio.windows.Window(vc.min(), vr.min(), vc.max() - vc.min() + 1, vr.max() - vr.min() + 1)
        va = vis.read(1, window=vwin, out_dtype="float64")
        if vis.nodata is not None:
            va[va == vis.nodata] = np.nan
        va = va * (vis.scales[0] or 1) + (vis.offsets[0] or 0)
        V = va[vr - vr.min(), vc - vc.min()].reshape(h.shape)
        gy, gx = np.gradient(h, self.res, self.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        return dict(h=h, V=V, X=X, Y=Y, wt=wt, slope=slope)


def crater_radius_km(w, cx, cy, rmin=3.0, rmax=40.0):
    """Rim crest = first maximum of the azimuthal-median elevation profile that stands >= 100 m above the
    profile minimum inside it and is followed by a >= 50 m drop (else global max)."""
    r = np.hypot(w["X"] - cx, w["Y"] - cy).ravel() / 1000
    ok = (r >= rmin) & (r <= rmax) & np.isfinite(w["h"].ravel())
    bins = np.floor(r[ok] / 0.5).astype(int)
    prof = pd.Series(w["h"].ravel()[ok]).groupby(bins).median().rolling(3, center=True, min_periods=1).mean()
    rr, pv = prof.index.values * 0.5 + 0.25, prof.values
    for i in range(1, len(pv) - 1):
        if pv[i] >= pv[max(0, i - 2):i + 3].max() and pv[i] - pv[:i + 1].min() >= 100 and pv[i] - pv[i:].min() >= 50:
            return float(rr[i])
    return float(rr[np.argmax(pv)])


def place_fallback(ras, ax, ay):
    w0 = ras.read(ax, ay, 42000)
    R = crater_radius_km(w0, ax, ay)
    w = ras.read(ax, ay, RING[1] * R * 1000 + HALF + MARGIN)
    res = ras.res
    land = (w["slope"] < SLOPE_MAX) & (w["V"] >= VIS_MIN)
    psr = w["V"] <= 0
    near = distance_transform_edt(~psr, sampling=res) / 1000 <= NEAR_PSR_KM
    good = (land & near).astype(np.float32)
    nbox = int(round(2 * HALF / res))
    score = uniform_filter(good, size=nbox, mode="constant") * nbox * nbox * (res / 1000) ** 2   # km2 in box
    rr = np.hypot(w["X"] - ax, w["Y"] - ay) / 1000
    edge = nbox // 2 + 2
    cand = (rr >= RING[0] * R) & (rr <= RING[1] * R)
    cand[:edge], cand[-edge:], cand[:, :edge], cand[:, -edge:] = False, False, False, False
    k = np.argmax(np.where(cand, score, -1))
    i, j = np.unravel_index(k, score.shape)
    return float(w["X"][i, j]), float(w["Y"][i, j]), R, float(score[i, j])


def zone_features(mask, w, s5p, t5p, dv, tile_tmin, res):
    px_km2 = (res / 1000) ** 2
    n = int(mask.sum())
    d = dict(area_km2=round(n * px_km2, 2), n_px=n)
    if n == 0:
        return d
    zv = w["h"][mask]
    s5 = s5p[mask]
    s5 = s5[np.isfinite(s5)]
    vv = w["V"][mask]
    clat, clon = inv_stereo_south(w["X"][mask].mean(), w["Y"][mask].mean())
    d.update(zone_centroid_lat=round(float(clat), 4), zone_centroid_lon=round(float(clon), 4),
             native_res_m=res, valid_frac=round(float(np.isfinite(zv).mean()), 3),
             elev_mean_m=float(np.nanmean(zv)), elev_std_m=float(np.nanstd(zv)),
             relief_m=float(np.nanpercentile(zv, 95) - np.nanpercentile(zv, 5)),
             slope_native_mean_deg=float(np.nanmean(w["slope"][mask])))
    if s5.size:
        d.update(slope500_mean_deg=float(s5.mean()), slope500_p90_deg=float(np.percentile(s5, 90)),
                 frac_slope500_gt10=float(np.mean(s5 > 10)), frac_slope500_gt20=float(np.mean(s5 > 20)),
                 tri500_mean_m=float(np.nanmean(t5p[mask])))
    d.update(lola_avg_sun_vis_mean=float(np.nanmean(vv)), lola_avg_sun_vis_p90=float(np.nanpercentile(vv, 90)),
             lola_avg_sun_vis_max=float(np.nanmax(vv)), lola_frac_psr=float(np.mean(vv <= 0)),
             lola_frac_sunlit_gt50pct=float(np.mean(vv > 0.5)))
    # Diviner triangles whose centre falls in a zone pixel
    r, c, tm, ta, ic = dv
    sel = mask[r, c]
    d["dvn_n_cells"] = int(sel.sum())
    if sel.any():
        tm, ta, ic = tm[sel], ta[sel], ic[sel]
        stable = ic >= 0
        d.update(dvn_tavg_mean_k=float(np.nanmean(ta)), dvn_tavg_min_k=float(np.nanmin(ta)),
                 dvn_tmax_mean_k=float(np.nanmean(tm)), dvn_tmax_max_k=float(np.nanmax(tm)),
                 dvn_tmax_min_k=float(np.nanmin(tm)), dvn_frac_coldtrap_110k=float(np.mean(tm[np.isfinite(tm)] < 110)),
                 dvn_frac_ice_stable=float(stable.mean()), dvn_frac_ice_within_1m=float((stable & (ic <= 1.0)).mean()),
                 dvn_frac_ice_surface=float((stable & (ic <= 0.0)).mean()),
                 dvn_ice_depth_median_m=float(np.median(ic[stable])) if stable.any() else np.nan)
    tmin = tile_tmin[mask]
    tmin = tmin[np.isfinite(tmin)]
    ill = illumination(float(clat), "moon")
    d.update(noon_elev_max_deg=ill["noon_elev_max_deg"], frac_polar_night=ill["frac_polar_night"],
             x_temp_max_k=d.get("dvn_tmax_max_k", np.nan), x_temp_mean_k=d.get("dvn_tavg_mean_k", np.nan),
             x_temp_min_k=float(tmin.mean()) if tmin.size else np.nan,
             x_grazing_sun_frac=float(grazing_sun_frac([clat], 1.54)[0]),
             x_noon_sun_max_deg=ill["noon_elev_max_deg"], x_polar_night_frac=ill["frac_polar_night"],
             x_frozen_ground_frac=d.get("dvn_frac_ice_within_1m", np.nan),
             x_surface_ice_frac=d.get("dvn_frac_coldtrap_110k", np.nan),
             x_sunlit_frac_terrain=d["lola_avg_sun_vis_mean"], x_shadowed_frac=d["lola_frac_psr"])
    d["x_temp_swing_k"] = d["x_temp_max_k"] - d["x_temp_min_k"]
    return d


def main():
    t = pd.read_csv(ROOT / "config/targets.csv").set_index("id")
    trek = trek_regions()
    z = np.load(RAW / "dlre_prp_south.npz")
    tx, ty = stereo_south(z["lat"], z["lon"])
    d_tmax, d_tavg, d_ice = (z[k].astype(float) for k in ("tmax", "tavg", "ice"))
    d_tmax[d_tmax <= 0] = np.nan
    d_tavg[d_tavg <= 0] = np.nan
    d_ice[d_ice == -1] = 0.0          # -1 = ice stable AT the surface; -999 = not stable (stays < 0)
    tiles = pd.read_csv(ROOT / "data/patches/moon_patches.csv", usecols=["patch_id", "x_m", "y_m"])
    tmin_csv = ROOT / "data/patches/moon_patches_tmin.csv"
    if tmin_csv.exists():
        tiles = tiles.merge(pd.read_csv(tmin_csv, usecols=["patch_id", "dvn_tbol_min_k"]), on="patch_id", how="left")
    else:
        tiles["dvn_tbol_min_k"] = np.nan
    tile_lut = {(int(a), int(b)): v for a, b, v in zip(tiles.x_m, tiles.y_m, tiles.dvn_tbol_min_k)}

    ras = Rasters()
    res = ras.res
    rows, summ, boxes = [], [], []
    for rid, rname, anchor, method, label in REGIONS:
        a = t.loc[anchor]
        ax, ay = stereo_south(a.lat, a.lon)
        crater_r, fb_score, source, note = np.nan, np.nan, SRC_TARGETS, ""
        if method == "official":
            o = trek[label]
            cx, cy, source = o["cx"], o["cy"], SRC_OFFICIAL
            note = f"2022 outline {o['w_km']:.0f}x{o['h_km']:.0f} km; 15x15 km box on its centre"
        elif method == "criteria_fallback":
            cx, cy, crater_r, fb_score = place_fallback(ras, float(ax), float(ay))
            source = f"criteria search around {anchor} crater-centre anchor (ring {RING[0]}-{RING[1]} R_crater)"
            note = f"crater radius {crater_r:.1f} km (DEM rim crest); {fb_score:.1f} km2 sunlit-flat ground <=2 km from PSR"
        else:
            cx, cy = float(ax), float(ay)
        w = ras.read(cx, cy, HALF + MARGIN)
        inbox = (np.abs(w["X"] - cx) <= HALF) & (np.abs(w["Y"] - cy) <= HALF) & np.isfinite(w["h"])
        land = inbox & (w["slope"] < SLOPE_MAX) & (w["V"] >= VIS_MIN)
        land50 = land & (w["V"] >= VIS_GOOD)
        ice = inbox & (w["V"] <= 0)
        # 500 m baseline slope / TRI on the whole window (extract.py helpers), mapped back to 80 m pixels
        f = max(1, int(round(SLOPE_BASELINE_M / res)))
        z5 = _block_mean(w["h"], f, f)
        s5p = np.full(w["h"].shape, np.nan)
        t5p = np.full(w["h"].shape, np.nan)
        H5, W5 = z5.shape
        s5p[:H5 * f, :W5 * f] = np.kron(_slope_deg(z5, res * f, res * f), np.ones((f, f)))
        t5p[:H5 * f, :W5 * f] = np.kron(_tri(z5), np.ones((f, f)))
        # tile-level Tmin (10 km tiles centred at 5000 + 10000 k)
        kx = (np.floor(w["X"] / 10000) * 10000 + 5000).astype(int)
        ky = (np.floor(w["Y"] / 10000) * 10000 + 5000).astype(int)
        tile_tmin = np.vectorize(lambda p, q: tile_lut.get((p, q), np.nan), otypes=[float])(kx, ky)
        # Diviner triangles in the window
        wt = w["wt"]
        sel = (np.abs(tx - cx) <= HALF + MARGIN) & (np.abs(ty - cy) <= HALF + MARGIN)
        ti_r = np.floor((ty[sel] - wt.f) / wt.e).astype(int)
        ti_c = np.floor((tx[sel] - wt.c) / wt.a).astype(int)
        ok = (ti_r >= 0) & (ti_r < w["h"].shape[0]) & (ti_c >= 0) & (ti_c < w["h"].shape[1])
        dv = (ti_r[ok], ti_c[ok], d_tmax[sel][ok], d_tavg[sel][ok], d_ice[sel][ok])
        # landing -> ice distance (to the region's own ice zone)
        dist = distance_transform_edt(~ice, sampling=res) / 1000 if ice.any() else None
        clat, clon = inv_stereo_south(cx, cy)
        links = dict(
            landing_to_ice_min_km=round(float(dist[land].min()), 2) if land.any() and dist is not None else np.nan,
            landing_to_ice_median_km=round(float(np.median(dist[land])), 2) if land.any() and dist is not None else np.nan,
            best_landing_vis=round(float(np.nanmax(w["V"][land])), 3) if land.any() else np.nan,
            landing_vis50_km2=round(land50.sum() * (res / 1000) ** 2, 2))
        base = dict(region_id=rid, region_name=rname, placement_method=method,
                    center_lat=round(float(clat), 4), center_lon=round(float(clon), 4))
        zf = {}
        for zname, m in (("landing", land), ("ice", ice)):
            zf[zname] = zone_features(m, w, s5p, t5p, dv, tile_tmin, res)
            rows.append({**base, "zone": zname, **zf[zname], **links, "anchor_target": anchor, "source": source})
        L, I = zf["landing"], zf["ice"]
        summ.append(dict(
            region_id=rid, target=anchor, name=rname, placement_method=method, source=source, placement_note=note,
            center_lat=round(float(clat), 4), center_lon=round(float(clon), 4), box_km=2 * HALF / 1000,
            offset_from_anchor_km=round(float(np.hypot(cx - ax, cy - ay)) / 1000, 1),
            crater_radius_km=crater_r, area_km2=round(inbox.sum() * (res / 1000) ** 2, 1),
            landing_km2=L["area_km2"], landing_vis50_km2=links["landing_vis50_km2"], psr_km2=I["area_km2"],
            best_landing_vis=links["best_landing_vis"],
            min_km_landing_to_psr=links["landing_to_ice_min_km"], median_km_landing_to_psr=links["landing_to_ice_median_km"],
            landing_tmax_mean_k=L.get("dvn_tmax_mean_k"), landing_tmax_max_k=L.get("dvn_tmax_max_k"),
            landing_tavg_k=L.get("dvn_tavg_mean_k"),
            psr_tmax_mean_k=I.get("dvn_tmax_mean_k"), psr_tmax_max_k=I.get("dvn_tmax_max_k"),
            psr_tavg_k=I.get("dvn_tavg_mean_k"),
            psr_ice_stable_frac=I.get("dvn_frac_ice_stable"), psr_ice_within_1m_frac=I.get("dvn_frac_ice_within_1m"),
            landing_slope500_p90_deg=L.get("slope500_p90_deg"), tile_tmin_k=I.get("x_temp_min_k")))
        corners = [(cx - HALF, cy - HALF), (cx + HALF, cy - HALF), (cx + HALF, cy + HALF), (cx - HALF, cy + HALF)]
        ring = []
        for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):   # densify edges (curved in lon/lat)
            for s in np.linspace(0, 1, 8, endpoint=False):
                la, lo = inv_stereo_south(x0 + s * (x1 - x0), y0 + s * (y1 - y0))
                ring.append([round(float(lo), 5), round(float(la), 5)])
        ring.append(ring[0])
        boxes.append(dict(type="Feature", properties=dict(region_id=rid, name=rname, placement_method=method),
                          geometry=dict(type="Polygon", coordinates=[ring])))
        print(f"{rid:22s} {method:17s} lat {clat:8.3f} lon {clon:8.2f}  {note}", flush=True)

    zones = pd.DataFrame(rows)
    lead = ["region_id", "region_name", "zone", "placement_method", "center_lat", "center_lon",
            "zone_centroid_lat", "zone_centroid_lon", "area_km2", "n_px"]
    zones = zones[lead + [c for c in zones.columns if c not in lead]]
    zones.round(4).to_csv(ROOT / "data/ml/moon_zones.csv", index=False)
    s = pd.DataFrame(summ)
    s.round(2).to_csv(ROOT / "data/patches/artemis_region_zones.csv", index=False)
    (ROOT / "data/patches/artemis_region_boxes.geojson").write_text(
        json.dumps(dict(type="FeatureCollection", features=boxes)))
    pd.set_option("display.width", 250)
    cols = ["region_id", "placement_method", "offset_from_anchor_km", "landing_km2", "landing_vis50_km2", "psr_km2",
            "median_km_landing_to_psr", "landing_tmax_mean_k", "landing_tavg_k", "psr_tmax_mean_k", "psr_tavg_k"]
    print(s[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
