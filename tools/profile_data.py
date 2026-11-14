#!/usr/bin/env python
"""
Decision-support profile of the Moon vs Earth data, per analog category.
Uses Earth patch tables when they exist (data/patches/earth_patches_*.csv), otherwise the
site-level caches from extract.py / extract_extra.py. Re-run any time; it only reads.

Writes docs/DATA_PROFILE.md and docs/profile_figs/*.png
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, DOCS = ROOT / "data", ROOT / "docs"
FIGS = DOCS / "profile_figs"

# category -> list of (moon column, earth column, label, unit, log-scale?)
# A pair with both columns names the SAME physical quantity on both bodies (comparable directly).
PAIRS = {
    "terrain": [
        ("relief_m", "relief_m", "Relief (p95-p5)", "m", True),
        ("slope500_mean_deg", "slope500_mean_deg", "Mean slope @500 m", "deg", False),
        ("slope500_p90_deg", "slope500_p90_deg", "90th pct slope @500 m", "deg", False),
        ("frac_slope500_gt10", "frac_slope500_gt10", "Share of ground >10 deg", "frac", False),
        ("tri500_mean_m", "tri500_mean_m", "Roughness (TRI @500 m)", "m", True),
    ],
    "temperature_illumination": [
        ("dvn_tmax_max_k", "lst_day_max_k", "Hottest surface temp", "K", False),
        ("dvn_tavg_mean_k", "lst_mean_k", "Mean surface temp", "K", False),
        ("dvn_tbol_min_k", "lst_night_min_k", "Coldest surface temp", "K", False),
        ("x_temp_swing_k", "x_temp_swing_k", "Max-min temperature swing", "K", False),
        ("x_grazing_sun_frac", "x_grazing_sun_frac", "Share of daylight with sun < 12 deg", "frac", False),
        ("frac_polar_night", "frac_polar_night", "Fraction of year in polar night", "frac", False),
        ("noon_elev_max_deg", "noon_elev_max_deg", "Highest noon sun", "deg", False),
    ],
    "water_ice": [
        ("dvn_frac_ice_within_1m", "x_frozen_ground_frac", "Ice-stable / frozen ground share", "frac", False),
        ("dvn_frac_coldtrap_110k", "wc_frac_snow_ice", "Surface ice share (cold trap vs snow/ice)", "frac", False),
    ],
}
# features that exist on one body only (reported for coverage, not overlap)
ONE_SIDED = {
    "geology": ["moon_geo_unit", "geo_frac_crater", "geo_frac_basin", "geo_frac_terra", "geo_frac_plains",
                "mstrat_lith", "mstrat_descrip", "sg_sand_pct", "sg_silt_pct", "sg_clay_pct", "sg_bdod_g_cm3", "sg_cfvo_pct"],
    "biology_habitability": ["ndvi_mean", "ndvi_max", "precip_best_mm_yr", "soil_moist_mean", "power_uv_index", "wc_frac_tree"],
    "water_ice_earth": ["pf_frac_pct", "pf_magt_2m_c", "wc_frac_snow_ice", "power_frost_days"],
    "dust_surface": ["aot_mean", "sg_silt_pct", "sg_clay_pct", "wc_frac_bare_sparse"],
    "operations": ["slope500_p90_deg", "frac_slope500_gt20", "lola_frac_psr", "lola_frac_sunlit_gt50pct"],
}


def grazing_sun_frac(lat, obliquity, low=12.0):
    """Share of daytime (sun above horizon) with the sun below `low` degrees, over a full year.
    Lunar-pole light is ~100% grazing; this is the comparable Earth quantity (analytic, flat horizon)."""
    eps, phi = np.radians(obliquity), np.radians(np.asarray(lat, float))[:, None, None]
    t = np.linspace(0, 1, 120, endpoint=False)[None, :, None]
    h = np.linspace(-np.pi, np.pi, 96, endpoint=False)[None, None, :]
    dec = np.arcsin(np.sin(eps) * np.sin(2 * np.pi * t))
    el = np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)))
    day = el > 0
    return ((day & (el < low)).sum(axis=(1, 2)) / np.maximum(day.sum(axis=(1, 2)), 1)).astype(float)


def load_moon():
    m = pd.read_csv(DATA / "patches" / "moon_patches.csv")
    m["x_grazing_sun_frac"] = grazing_sun_frac(m["center_lat"], 1.54)
    gcp = DATA / "patches" / "moon_patches_tmin.csv"
    if gcp.exists():
        m = m.merge(pd.read_csv(gcp), on="patch_id", how="left")
    if "dvn_tbol_min_k" in m:
        m["x_temp_swing_k"] = m["dvn_tmax_max_k"] - m["dvn_tbol_min_k"]
    return m


def load_earth():
    """Prefer patch-level tables; fall back to site-level caches."""
    t = DATA / "patches" / "earth_patches_terrain.csv"
    e = DATA / "patches" / "earth_patches_env.csv"
    if t.exists():
        df = pd.read_csv(t)
        if e.exists():
            env = pd.read_csv(e)
            df = df.merge(env.drop(columns=[c for c in env.columns if c in df.columns and c != "patch_id"]),
                          on="patch_id", how="left")
        src = f"patch-level ({len(df)} Earth patches)"
    else:
        recs = {}
        for f in glob.glob(str(DATA / "cache" / "earth_*.json")):
            j = json.loads(Path(f).read_text())
            recs[j["id"]] = j
        for f in glob.glob(str(DATA / "cache" / "extra_earth_*.json")):
            j = json.loads(Path(f).read_text())
            recs.setdefault(j["id"], {}).update(j)
        df = pd.DataFrame(recs.values()).rename(columns={"id": "patch_id"})
        # site-level names -> patch-level names
        df["lst_day_max_k"] = df.get("lst_day_k")
        df["lst_night_min_k"] = df.get("lst_night_k")
        if "lst_day_k" in df:
            df["lst_mean_k"] = (df["lst_day_k"] + df["lst_night_k"]) / 2
        src = f"SITE-LEVEL fallback ({len(df)} named sites; annual-mean LST, so extremes are understated)"
    lat_col = "lat" if "lat" in df else ("center_lat" if "center_lat" in df else None)
    if lat_col:
        df["x_grazing_sun_frac"] = grazing_sun_frac(df[lat_col], 23.44)
    if {"lst_day_max_k", "lst_night_min_k"} <= set(df.columns):
        df["x_temp_swing_k"] = df["lst_day_max_k"] - df["lst_night_min_k"]
    # frozen-ground proxy: permafrost fraction if present, else POWER mean air temp < -2 C
    if "pf_frac_pct" in df:   # ESA CCI permafrost fraction (%), proxy where outside the CCI domain
        proxy = (df["power_t2m_c"] < -2).astype(float) if "power_t2m_c" in df else np.nan
        df["x_frozen_ground_frac"] = (df["pf_frac_pct"] / 100).fillna(proxy)
    elif "power_t2m_c" in df:
        df["x_frozen_ground_frac"] = (df["power_t2m_c"] < -2).astype(float)
    elif "t_mean_c" in df:
        df["x_frozen_ground_frac"] = (df["t_mean_c"] < -2).astype(float)
    return df, src


def overlap(mv, ev):
    mv, ev = mv[np.isfinite(mv)], ev[np.isfinite(ev)]
    if not len(mv) or not len(ev):
        return dict(moon_in_earth_range=np.nan, earth_in_moon_range=np.nan, gap=np.nan)
    e_lo, e_hi = np.percentile(ev, [1, 99])
    m_lo, m_hi = np.percentile(mv, [1, 99])
    gap = max(0.0, m_lo - e_hi, e_lo - m_hi)
    return dict(moon_in_earth_range=float(((mv >= e_lo) & (mv <= e_hi)).mean()),
                earth_in_moon_range=float(((ev >= m_lo) & (ev <= m_hi)).mean()), gap=gap)


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:,.{nd}f}" if abs(x) < 1e4 else f"{x:,.0f}"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    moon, earth_all = load_moon(), None
    earth, esrc = load_earth()
    mt = moon[moon.get("is_target", 0) == 1]
    lines = ["# Data profile: Moon vs Earth, by analog category", "",
             f"- **Moon:** {len(moon)} south-pole patches (10 x 10 km, lat <= -80), incl. {len(mt)} Artemis target patches.",
             f"- **Earth:** {esrc}.", "",
             "Read this before choosing models. *Overlap* = share of one body's patches whose value falls inside the "
             "other body's 1st-99th percentile range. **Overlap near 0 means a direct distance is meaningless** "
             "for that feature: the score must instead be a calibrated 'closest approach' (how near Earth gets), "
             "and the gap itself is a finding to report.", ""]

    # ---- cross-body overlap table + figure
    rows = []
    n_pairs = sum(len(v) for v in PAIRS.values())
    fig, axes = plt.subplots(int(np.ceil(n_pairs / 3)), 3, figsize=(15, 3.1 * np.ceil(n_pairs / 3)))
    axes = axes.ravel()
    k = 0
    for cat, plist in PAIRS.items():
        for mc, ec, label, unit, logx in plist:
            mv = moon[mc].to_numpy(float) if mc in moon else np.array([])
            ev = earth[ec].to_numpy(float) if ec in earth else np.array([])
            o = overlap(mv, ev)
            fm, fe = np.isfinite(mv), np.isfinite(ev)
            rows.append(dict(category=cat, feature=label, unit=unit, moon_col=mc, earth_col=ec,
                             moon_cov=fm.mean() if len(mv) else 0, earth_cov=fe.mean() if len(ev) else 0,
                             moon_p5=np.percentile(mv[fm], 5) if fm.any() else np.nan,
                             moon_p50=np.percentile(mv[fm], 50) if fm.any() else np.nan,
                             moon_p95=np.percentile(mv[fm], 95) if fm.any() else np.nan,
                             earth_p5=np.percentile(ev[fe], 5) if fe.any() else np.nan,
                             earth_p50=np.percentile(ev[fe], 50) if fe.any() else np.nan,
                             earth_p95=np.percentile(ev[fe], 95) if fe.any() else np.nan, **o))
            ax = axes[k]
            k += 1
            if fm.any() or fe.any():
                allv = np.concatenate([mv[fm], ev[fe]])
                if logx:
                    allv = allv[allv > 0]
                    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 40) if len(allv) else 20
                    ax.set_xscale("log")
                else:
                    bins = np.linspace(allv.min(), allv.max(), 40) if len(allv) and allv.min() < allv.max() else 20
                if fm.any():
                    ax.hist(mv[fm], bins=bins, alpha=.55, density=True, label=f"Moon (n={fm.sum()})", color="#7a7a7a")
                if fe.any():
                    ax.hist(ev[fe], bins=bins, alpha=.55, density=True, label=f"Earth (n={fe.sum()})", color="#2f7ed8")
                for _, r in mt.iterrows():
                    if mc in r and np.isfinite(r[mc]):
                        ax.axvline(r[mc], color="k", lw=.6, alpha=.5)
            ax.set_title(f"[{cat}] {label}", fontsize=9)
            ax.set_xlabel(unit, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Moon south-pole patches vs Earth (black lines = Artemis target patches)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "moon_vs_earth_distributions.png", dpi=110)
    plt.close(fig)
    R = pd.DataFrame(rows)
    R.to_csv(DOCS / "data_profile_overlap.csv", index=False)

    lines += ["## 1. Shared features: do Moon and Earth values even overlap?", "",
              "![distributions](profile_figs/moon_vs_earth_distributions.png)", "",
              "| Category | Feature | Moon p5 / p50 / p95 | Earth p5 / p50 / p95 | Moon in Earth range | Earth in Moon range | Gap | Coverage M / E |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        lines.append(f"| {r.category} | {r.feature} ({r.unit}) | {fmt(r.moon_p5)} / {fmt(r.moon_p50)} / {fmt(r.moon_p95)} | "
                     f"{fmt(r.earth_p5)} / {fmt(r.earth_p50)} / {fmt(r.earth_p95)} | {fmt(100 * r.moon_in_earth_range, 0)}% | "
                     f"{fmt(100 * r.earth_in_moon_range, 0)}% | {fmt(r.gap)} | {fmt(100 * r.moon_cov, 0)}% / {fmt(100 * r.earth_cov, 0)}% |")

    # ---- one-sided coverage
    lines += ["", "## 2. One-sided / text features: coverage", "",
              "| Category | Feature | Moon coverage | Earth coverage |", "|---|---|---|---|"]
    for cat, cols in ONE_SIDED.items():
        for c in cols:
            mc = f"{100 * moon[c].notna().mean():.0f}%" if c in moon else "n/a"
            ecv = f"{100 * earth[c].notna().mean():.0f}%" if c in earth else "not yet"
            lines.append(f"| {cat} | `{c}` | {mc} | {ecv} |")

    # ---- Moon structure: constants, redundancy, intrinsic dimension
    num = moon.select_dtypes("number").drop(columns=[c for c in ("is_target", "x_m", "y_m", "center_lat", "center_lon",
                                                                 "dvn_n_cells", "dem_valid_frac", "valid_frac",
                                                                 "native_res_m", "geo_valid_frac") if c in moon])
    const = [c for c in num if num[c].nunique(dropna=True) <= 1]
    num = num.drop(columns=const)
    corr = num.corr().abs()
    red = sorted({(a, b, round(corr.loc[a, b], 3)) for a in corr for b in corr if a < b and corr.loc[a, b] > 0.95},
                 key=lambda x: -x[2])
    Z = ((num - num.mean()) / num.std(ddof=0)).fillna(0).to_numpy()
    ev = np.linalg.svd(Z, compute_uv=False) ** 2
    cum = np.cumsum(ev) / ev.sum()
    lines += ["", "## 3. Moon patch structure (what the encoders will see)", "",
              f"- Numeric Moon features: {num.shape[1]} (after dropping constants: {', '.join(const) or 'none'}).",
              f"- PCA: {int(np.searchsorted(cum, .80) + 1)} components explain 80% of variance, "
              f"{int(np.searchsorted(cum, .95) + 1)} explain 95%. Small intrinsic dimension = small encoders are enough.",
              f"- Near-duplicate pairs (|r| > 0.95), keep one of each: " +
              (", ".join(f"`{a}`~`{b}` ({r})" for a, b, r in red[:12]) or "none")]

    # ---- quick terrain nearest-neighbour preview (sanity check only, not the model)
    tf = [p[0] for p in PAIRS["terrain"]]
    if set(tf) <= set(earth.columns) and len(mt):
        both = pd.concat([moon[tf], earth[tf]], ignore_index=True).astype(float)
        both[["relief_m", "tri500_mean_m"]] = np.log10(1 + both[["relief_m", "tri500_mean_m"]].clip(lower=0))
        Zb = (both - both.mean()) / both.std(ddof=0)
        Zm = Zb.iloc[:len(moon)].set_index(moon["patch_id"])
        Ze = Zb.iloc[len(moon):].set_index(earth["patch_id"])
        name = earth.set_index("patch_id")["name"] if "name" in earth else pd.Series(earth["patch_id"].values, index=earth["patch_id"])
        lines += ["", "## 4. Sanity preview: nearest Earth terrain for each Artemis target (plain z-score distance, not the model)", "",
                  "| Moon target | Nearest Earth (terrain only) | 2nd | 3rd |", "|---|---|---|---|"]
        for pid in mt["patch_id"]:
            d = np.sqrt(((Ze - Zm.loc[pid]) ** 2).sum(axis=1, min_count=3)).dropna().nsmallest(3)
            lines.append(f"| {pid} | " + " | ".join(f"{name.get(i, i)} ({v:.2f})" for i, v in d.items()) + " |")

    # ---- decisions this implies
    lines += ["", "## 5. What this means for the ML design", ""]
    for _, r in R.iterrows():
        if np.isfinite(r.moon_in_earth_range):
            verdict = ("direct similarity OK" if r.moon_in_earth_range > .5 else
                       "partial overlap: similarity OK inside overlap, closest-approach outside" if r.moon_in_earth_range > .05 else
                       "NO overlap: use calibrated closest-approach + report the gap")
            lines.append(f"- **{r.feature}** ({r.category}): Moon-in-Earth-range {100 * r.moon_in_earth_range:.0f}% -> {verdict}.")
    (DOCS / "DATA_PROFILE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
