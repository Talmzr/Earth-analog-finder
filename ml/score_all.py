"""Steps 4-6: category scores for every (Moon entity, Earth tile) pair, anchor calibration, final outputs.

  python -m ml.score_all                 # uses embeddings/ from terrain_ssl + geology_ssl when present
  python -m ml.score_all --emb-suffix _smoke

Every category is a small linear model over interpretable TERMS in [-1, 1]:
    calibrated = clip(b + sum_k c_k * term_k, -1, 1), then the category's caps
with c_k >= 0 and b fitted on the expert anchors of that category (ridge towards the prior weights), so the anchors
set both the term weights and what the -1..+1 scale means. raw_score is the same sum with the PRIOR weights.
Terms (see docs/ML_RESULTS.md):
  terrain     ssl_cos (DINOv2 embedding cosine), kernel (relief, slope, roughness, steep share)
  geology     text_cos (BGE embedding cosine), lith_match (discrete lithology match)
  thermal     temperature swing / min / max, grazing-sun share, noon sun height, polar night   [closest approach]
  water_ice   frozen ground, surface ice (both one-sided), landing -> ice traverse distance
  dust        Earth soil vs a fixed lunar-regolith reference (fines, density, coarse, bare, dryness, tephra, aerosol)
  biology     1 - (NDVI, precipitation, soil moisture, tree cover)        Earth-only ranking, capped
  caves       rule: known lava-tube region / basaltic volcano             Tranquillitatis pit target only
  operations  calibrated terrain + thermal + water_ice, slope-hazard kernel, traverse, forest/urban penalty
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from . import common as C
from . import geo_text as G

SUB = C.subscore
TUBE_REGIONS = re.compile(
    r"hawai|kilauea|mauna loa|hualalai|iceland|lanzarote|canary|tenerife|jeju|halla\b|galapagos|gal.pagos|azores|"
    r"\betna\b|undara|mcbride|medicine lake|lava beds|newberry|st\. helens|craters of the moon|snake river|"
    r"zuni-bandera|el malpais|khaybar|chyulu|suswa|\bfuji\b|r.union|piton de la fournaise|rapa nui|easter island")
CAPS = dict(terrain=0.9, geology=0.8, thermal=0.4, water_ice=0.7, dust=0.7, biology=0.6, caves=0.5, operations=0.8)


def col(df, name):
    return df[name].to_numpy(float) if name in df else np.full(len(df), np.nan)


def M_(ent, name):
    return col(ent, name)[:, None]


def E_(E, name):
    return col(E, name)[None, :]


def nanmean0(stack):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(stack), axis=0)


# ============================================================================= terms per category
def cross_body_agreement(Em, Ee, fm, fe, k=10):
    """Mean |difference| in measured terrain between a Moon tile and its k most similar Earth tiles, relative to
    random Earth tiles (averaged over features). Lower is better; 1.0 = similarity carries no terrain information."""
    A = Em / np.linalg.norm(Em, axis=1, keepdims=True)
    B = Ee / np.linalg.norm(Ee, axis=1, keepdims=True)
    top = np.argsort(-(A @ B.T), axis=1)[:, :k]
    near = np.abs(fm[:, None, :] - fe[top]).mean(axis=(0, 1))
    rand = np.abs(fm[:, None, :] - fe[None, :, :]).mean(axis=(0, 1))
    return float(np.mean(near / rand)), [round(float(v), 3) for v in near / rand]


def inlp(Xtr, ytr, k=8):
    """Iterative null-space projection: remove the k directions that best tell Moon from Earth."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    P = np.eye(Xtr.shape[1])
    Z = (Xtr - mu) / sd
    for _ in range(k):
        w = LogisticRegression(max_iter=200, C=0.1).fit(Z @ P, ytr).coef_[0]
        w = P @ w
        w /= np.linalg.norm(w) + 1e-12
        P = P - np.outer(P @ w, w)
    return lambda X: ((X - mu) / sd) @ P


def terrain_terms(ent, E, moon, earth, zones, suffix, info):
    feats = [("relief_m", True, 0.15), ("slope500_mean_deg", False, 3.0), ("tri500_mean_m", True, 0.15),
             ("frac_slope500_gt10", False, 0.15)]
    subs = []
    for name, logt, tau in feats:
        m, e = M_(ent, name), E_(E, name)
        if logt:
            m, e = np.log10(1 + np.clip(m, 0, None)), np.log10(1 + np.clip(e, 0, None))
        subs.append(SUB(m, e, tau))
    terms = {"kernel": nanmean0(subs)}
    prior = {"kernel": 0.4}

    # ---- SSL embeddings: choose the variant whose cross-body neighbours agree best with measured terrain
    f_cols = ["relief_m", "slope500_mean_deg", "tri500_mean_m"]
    def feat(df):
        f = df[f_cols].to_numpy(float)
        return np.column_stack([np.log10(1 + np.clip(f[:, 0], 0, None)), f[:, 1], np.log10(1 + np.clip(f[:, 2], 0, None))])
    fm_all, fe_all = feat(moon), feat(earth)
    held_m = C.holdout_mask(C.spatial_blocks(moon["center_lat"], moon["center_lon"], np.full(len(moon), "moon")), 0.15, 0)
    held_e = C.holdout_mask(C.spatial_blocks(earth["center_lat"], earth["center_lon"], np.full(len(earth), "earth")), 0.15, 0)
    zone_ix = np.where(ent["kind"].to_numpy() == "zone")[0]
    tile_pos = {p: i for i, p in enumerate(moon["patch_id"])}
    best, table = None, []
    for tag in ("", "_coral"):
        fm, fe, fz = (C.EMB / f"{n}_terrain_384{tag}{suffix}.npy" for n in ("moon", "earth", "moon_zone"))
        if not (fm.exists() and fe.exists() and fz.exists()):
            continue
        Mall, Eall, Z = np.load(fm), np.load(fe), np.load(fz)
        Ment = np.full((len(ent), Mall.shape[1]), np.nan, np.float32)
        Ment[zone_ix] = Z[:len(zone_ix)]
        for i, (kind, tid) in enumerate(zip(ent["kind"], ent["target_id"])):
            if kind == "tile" and tid in tile_pos:
                Ment[i] = Mall[tile_pos[tid]]
        okm, oke = np.isfinite(Mall).all(1), np.isfinite(Eall).all(1)
        pooled_mu = np.vstack([Mall[okm], Eall[oke]]).mean(0)
        variants = {"pooled": (lambda X, b: X - pooled_mu),
                    "per_body": (lambda X, b, mm=Mall[okm].mean(0), me=Eall[oke].mean(0): X - (mm if b == "m" else me))}
        try:
            trm, tre = okm & ~held_m, oke & ~held_e
            proj = inlp(np.vstack([Mall[trm], Eall[tre]]), np.r_[np.zeros(trm.sum()), np.ones(tre.sum())])
            mu_p = proj(np.vstack([Mall[okm], Eall[oke]])).mean(0)
            variants["inlp8"] = lambda X, b, proj=proj, mu_p=mu_p: proj(X) - mu_p
        except Exception as ex:  # noqa: BLE001
            info.setdefault("notes", []).append(f"INLP variant skipped: {type(ex).__name__}: {str(ex)[:80]}")
        vm, ve = okm & held_m & np.isfinite(fm_all).all(1), oke & held_e & np.isfinite(fe_all).all(1)
        for vname, fn in variants.items():
            if vm.sum() < 5 or ve.sum() < 15:
                score, per = float("nan"), []
            else:
                score, per = cross_body_agreement(fn(Mall[vm], "m"), fn(Eall[ve], "e"), fm_all[vm], fe_all[ve])
            table.append(dict(model="vicreg" + tag, variant=vname, agreement=round(score, 3), per_feature=per,
                              n_moon=int(vm.sum()), n_earth=int(ve.sum())))
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, tag, vname, fn(Ment, "m"), fn(Eall, "e"))
        if best is None:                           # too few held-out embeddings (smoke run): take the first variant
            best = (float("nan"), tag, "pooled", variants["pooled"](Ment, "m"), variants["pooled"](Eall, "e"))
    info["terrain_ssl_variants"] = table
    if best is not None:
        _, tag, vname, A, B = best
        with np.errstate(all="ignore"):
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            B = B / np.linalg.norm(B, axis=1, keepdims=True)
            terms["ssl_cos"] = A @ B.T
        prior = {"ssl_cos": 0.6, "kernel": 0.4}
        info["terrain_ssl_chosen"] = dict(model="vicreg" + tag, variant=vname, agreement=None if not np.isfinite(best[0]) else round(best[0], 3))
    else:
        info["terrain_ssl_chosen"] = "none (no embeddings found: kernel only)"
    return terms, prior


def geology_terms(ent, E, earth, suffix, info):
    cat = pd.read_csv(C.DATA / "catalogs" / "moon_geologic_units.csv")
    _, zc = G.moon_texts_for(ent["moon_geo_unit"], ent["geo_frac_mare"], cat)
    ice = col(earth, "x_surface_ice_frac")
    et = [G.earth_text(r) for r in earth.to_dict("records")]
    ec = [G.earth_classes(t, s, i) for t, s, i in zip(et, earth["source"], ice)]
    terms = {"lith_match": np.array([[G.lith_match(m, e) for e in ec] for m in zc], float)}
    prior = {"lith_match": 1.0}
    fm, fe = C.EMB / f"moon_entity_geology_384{suffix}.npy", C.EMB / f"earth_geology_384{suffix}.npy"
    if fm.exists() and fe.exists():
        A, B = np.load(fm), np.load(fe)
        cos = A @ B.T
        cos[:, ~np.isfinite(B).all(1)] = np.nan
        terms["text_cos"] = cos
        prior = {"text_cos": 1.0, "lith_match": 1.0}
        info["geology_embeddings"] = "fine-tuned BGE"
    else:
        info["geology_embeddings"] = "none (lithology match only)"
    unknown = np.array([not e.get("known") for e in ec])
    for k in terms:
        terms[k][:, unknown] = np.nan
    return terms, prior


def thermal_terms(ent, E):
    spec = [("swing", "x_temp_swing_k", 50.0), ("t_min", "x_temp_min_k", 60.0), ("t_max", "x_temp_max_k", 40.0),
            ("grazing_sun", "x_grazing_sun_frac", 0.3), ("noon_sun", "x_noon_sun_max_deg", 10.0),
            ("polar_night", "x_polar_night_frac", 0.15)]
    terms = {n: SUB(M_(ent, c), E_(E, c), tau) for n, c, tau in spec}
    return terms, {n: 1 / len(spec) for n in terms}


def traverse_term(ent, E):
    """Earth distance from flat bare ground to ice vs the Moon's landing -> shadow distance (only LONGER Earth
    distances are penalised). An Earth tile with no flat bare ground at all fails the traverse outright."""
    d_e = np.where(col(E, "x_frozen_ground_frac") >= 0.5, 0.0, col(E, "trav_dist_flat_to_ice_km"))  # ground ice counts
    t = SUB(d_e[None, :], M_(ent, "landing_to_ice_median_km"), 3.0, one_sided=True)
    no_ground = (col(E, "trav_flat_bare_frac") < 0.02)[None, :] & np.isfinite(M_(ent, "landing_to_ice_median_km"))
    return np.where(no_ground, -1.0, t)


def water_terms(ent, E):
    terms = {"frozen_ground": SUB(M_(ent, "x_frozen_ground_frac"), E_(E, "x_frozen_ground_frac"), 0.25, one_sided=True),
             "surface_ice": SUB(M_(ent, "x_surface_ice_frac"), E_(E, "x_surface_ice_frac"), 0.4, one_sided=True),
             "traverse": traverse_term(ent, E)}
    with np.errstate(all="ignore"):                                      # liquid water: the Moon has none
        wet = np.nanmean(np.vstack([np.clip(col(E, "precip") / 1000.0, 0, 1), np.clip(col(E, "soil_moist_frac") / 0.4, 0, 1)]), axis=0)
    terms["no_liquid_water"] = SUB(0.0, wet[None, :], 0.15) * np.ones((len(ent), 1))
    return terms, {"frozen_ground": 0.35, "surface_ice": 0.2, "traverse": 0.25, "no_liquid_water": 0.2}


def dust_terms(ent, E, info):
    ref = dict(fines_pct=45.0, bulk_density=1.5, coarse_pct=5.0)        # fixed lunar-regolith reference
    info["dust_reference"] = dict(ref, note="fines = % finer than 50 um; lunar soil median grain size ~70 um "
                                  "(the brief's '70-80% sand-sized' would be fines ~25%; kept the 45% of categories.json)")
    one = np.ones((len(ent), 1))
    t = {"fines": SUB(ref["fines_pct"], E_(E, "fines_pct"), 15.0),
         "bulk_density": SUB(ref["bulk_density"], E_(E, "sg_bdod_g_cm3"), 0.2),
         "coarse": SUB(ref["coarse_pct"], E_(E, "sg_cfvo_pct"), 15.0),
         "bare": SUB(1.0, E_(E, "bare_f"), 0.25),
         "dryness": SUB(0.0, E_(E, "soil_moist_frac"), 0.08),
         "aerosol": 2 * np.clip(E_(E, "aot_mean") / 0.8, 0, 1) - 1}
    phys = np.isfinite(np.stack(list(t.values()))).sum(0)                # physical terms available per Earth tile
    t["tephra"] = np.where(E["flag_tephra"].to_numpy(bool), 1.0, 0.0)[None, :]
    terms = {k: np.where(phys >= 1, v, np.nan) * one for k, v in t.items()}
    prior = dict(fines=0.25, bulk_density=0.15, coarse=0.1, bare=0.2, dryness=0.1, aerosol=0.05, tephra=0.15)
    return terms, prior


def biology_terms(ent, E):
    one = np.ones((len(ent), 1))
    t = {"ndvi": col(E, "ndvi_f"), "precip": np.clip(col(E, "precip") / 1000.0, 0, 2),
         "soil_moisture": col(E, "soil_moist_frac"), "tree_cover": col(E, "tree_f")}
    terms = {k: -v[None, :] * one for k, v in t.items()}                 # more life -> lower score
    return terms, dict(ndvi=1.0, precip=0.5, soil_moisture=1.0, tree_cover=0.5)


def caves_terms(ent, E, earth):
    """Rule levels (brief + categories.json), used as the score itself:
       +0.5 known lava tubes: named lava-tube site, or a volcanic tile whose catalog text mentions lava tubes / caves
            or that lies in a known lava-tube region (Hawaii, Iceland, Canaries, Jeju, ...)
       +0.3 basaltic volcano / shield / volcanic field: tubes plausible, not documented in the tile text
       -0.5 sedimentary, evaporite or ice-covered ground with no volcanic evidence: no lava tubes
        0.0 everything else.  Only the Tranquillitatis pit target is scored."""
    ctx = C.text_blob(earth, ["name", "context"])                      # catalog / tag text only, not the bedrock map
    has = lambda rx: ctx.str.contains(rx, regex=True).to_numpy()       # noqa: E731
    volc_src = earth["source"].eq("volcano").to_numpy()
    named_volc = earth["source"].eq("named_site").to_numpy() & has(r"basalt|volcanic|lava")
    volcanic = volc_src | named_volc
    known = has(r"lava_tube") | (volcanic & (has(r"lava[ _-]?tubes?|lava caves?|\bcaves?\b") | has(TUBE_REGIONS.pattern)))
    plausible = (volc_src & has(r"shield|basalt")) | (named_volc & ~E["flag_aqueous_sed"].to_numpy())
    no_tubes = (E["flag_aqueous_sed"].to_numpy() | E["flag_ice_surface"].to_numpy()) & ~volc_src & ~known & ~plausible
    rule = np.where(known, 0.5, np.where(plausible, 0.3, np.where(no_tubes, -0.5, 0.0)))
    t = np.full((len(ent), len(earth)), np.nan)
    t[ent["is_pit"].to_numpy()] = rule
    return {"rule": t}, {"rule": 1.0}, dict(known_tube_tiles=int(known.sum()), plausible_tiles=int((plausible & ~known).sum()),
                                           no_tube_tiles=int(no_tubes.sum()))


def hazard_term(ent, E):
    return nanmean0([SUB(M_(ent, "haz_gt10"), E_(E, "frac_slope500_gt10"), 0.10),
                     SUB(M_(ent, "haz_gt20"), E_(E, "frac_slope500_gt20"), 0.08),
                     SUB(M_(ent, "haz_p90"), E_(E, "slope500_p90_deg"), 4.0)])


# ============================================================================= calibration
def calibrate(name, terms, prior, anchors, ent_pos, earth_pos, prior_b=0.0, lam=1.0, fit=True):
    names = list(terms)
    S = np.stack([terms[k] for k in names])                              # (k, M, E)
    n_terms = np.isfinite(S).sum(0).astype(np.int8)
    with np.errstate(all="ignore"):
        fill = np.nanmean(S, axis=0)
    S = np.where(np.isfinite(S), S, fill[None])                          # missing term -> mean of the pair's others
    p = np.array([prior[k] for k in names], float)
    raw = np.clip(prior_b + np.tensordot(p, S, axes=1), -1, 1)
    a = anchors[anchors["cat"] == name].copy()
    a["mi"], a["ei"] = a["entity_id"].map(ent_pos), a["earth_site"].map(earth_pos)
    a = a.dropna(subset=["mi", "ei"])
    mi, ei = a["mi"].to_numpy(int), a["ei"].to_numpy(int)
    T = S[:, mi, ei].T
    ok = np.isfinite(T).all(1)
    a, T = a[ok], T[ok]
    y, w = a["label"].to_numpy(float), a["w"].to_numpy(float)
    stats = dict(category=name, n_anchors=int(len(a)), terms=names, prior=dict(zip(names, p.round(3).tolist())))
    if len(a) >= 4 and not fit:                    # rule-defined score: anchors check it, they do not rescale it
        c, b = p, prior_b
        pred, loo = np.clip(b + T @ c, -1, 1), np.clip(b + T @ c, -1, 1)
        stats.update(weights=dict(zip(names, p.round(4).tolist())), intercept=b, note="rule levels used as the score (not rescaled)",
                     mae=round(float(np.abs(pred - y).mean()), 3), mae_loo=round(float(np.abs(pred - y).mean()), 3),
                     spearman=round(C.spearman(pred, y), 3), spearman_raw=round(C.spearman(pred, y), 3), spearman_loo=round(C.spearman(pred, y), 3))
    elif len(a) >= 4:
        c, b = C.fit_terms(T, y, w, prior=p, lam=lam)
        pred = np.clip(b + T @ c, -1, 1)
        loo = np.full(len(a), np.nan)
        for i in range(len(a)):
            m = np.arange(len(a)) != i
            ci, bi = C.fit_terms(T[m], y[m], w[m], prior=p, lam=lam)
            loo[i] = np.clip(bi + T[i] @ ci, -1, 1)
        stats.update(weights=dict(zip(names, np.round(c, 4).tolist())), intercept=round(b, 4),
                     mae=round(float(np.abs(pred - y).mean()), 3), mae_loo=round(float(np.abs(loo - y).mean()), 3),
                     spearman=round(C.spearman(pred, y), 3), spearman_raw=round(C.spearman(raw[mi[ok], ei[ok]], y), 3),
                     spearman_loo=round(C.spearman(loo, y), 3))
    else:
        c, b = p, prior_b
        pred, loo = np.clip(b + T @ c, -1, 1) if len(a) else np.array([]), np.full(len(a), np.nan)
        stats.update(weights=dict(zip(names, np.round(c, 4).tolist())), intercept=b, note="too few anchors: prior weights kept")
    cal = np.clip(b + np.tensordot(np.asarray(c, float), S, axes=1), -1, 1)
    res = a[["moon_target", "moon_zone", "entity_id", "earth_site", "category", "label", "confidence"]].copy()
    res["raw_score"], res["calibrated_uncapped"], res["loo_pred"] = raw[mi[ok], ei[ok]], pred, loo
    return raw, cal, n_terms, stats, res


def apply_caps(name, cal, ent, E):
    cal = np.minimum(cal, CAPS[name])
    if name == "terrain":
        cal = np.where((col(E, "obstacle_frac") >= 0.5)[None, :], np.minimum(cal, 0.0), cal)
    if name == "thermal":
        psr = (col(ent, "dvn_frac_coldtrap_110k") >= 0.5) | (col(ent, "lola_frac_psr") >= 0.5)
        cal = np.where(psr[:, None], np.minimum(cal, 0.0), cal)
    if name == "water_ice":
        cold = col(ent, "dvn_frac_coldtrap_110k") >= 0.5
        cal = np.where(cold[:, None], np.minimum(cal, 0.3), cal)
    if name == "operations":
        pen = ((col(E, "tree_f") > 0.3) | (col(E, "wc_frac_built") > 0.1))[None, :]
        cal = np.where(pen, cal - 0.5, cal)
        cal = np.where((col(E, "obstacle_frac_ops") >= 0.5)[None, :], np.minimum(cal, -0.3), cal)
        cal = np.clip(cal, -1, CAPS[name])
    return cal


# ============================================================================= outputs
def long_table(ent, earth, raw, cal, nt):
    M, Ecount = len(ent), len(earth)
    mz = np.repeat(ent["entity_id"].to_numpy(), Ecount)
    et = np.tile(earth["patch_id"].to_numpy(), M)
    parts = []
    for c in cal:
        d = pd.DataFrame(dict(moon_zone_id=mz, earth_tile_id=et, category=c,
                              raw_score=raw[c].ravel().astype(np.float32),
                              calibrated_score=cal[c].ravel().astype(np.float32), n_terms=nt[c].ravel()))
        parts.append(d[np.isfinite(d["calibrated_score"])])
    out = pd.concat(parts, ignore_index=True)
    for c in ("moon_zone_id", "earth_tile_id", "category"):
        out[c] = out[c].astype("category")
    return out


def precondition(expr, ent):
    if not expr:
        return np.ones(len(ent), bool)
    ns = {c: ent[c] for c in ent.columns}
    ns["contains"] = lambda s, sub: s.fillna("").astype(str).str.contains(sub, regex=False)
    return pd.Series(eval(expr, {"__builtins__": {}}, ns)).fillna(False).to_numpy(bool)  # noqa: S307 (own config)


def readiness(ent, earth, cal, exps, avoid_below):
    Ecount = len(earth)
    rows, app_rows = [], []
    counts = {s: np.zeros((len(ent), Ecount), np.int16) for s in ("sufficient", "partial", "avoid")}
    wsum = {c: np.zeros(len(ent)) for c in cal}
    for ex in exps:
        req = {C.CFG2OUT[k]: v for k, v in ex["required"].items()}
        sup = {C.CFG2OUT[k]: v for k, v in (ex.get("supporting") or {}).items()}
        app = precondition(ex.get("moon_precondition"), ent)
        for c in req:
            wsum[c] += app * 1.0
        for c in sup:
            wsum[c] += app * 0.5
        names = list(req)
        S = np.stack([cal[c] for c in names])
        mins = np.array([req[c] for c in names])[:, None, None]
        nan = np.isnan(S).any(0)
        avoid = (S < avoid_below).any(0)
        suff = (S >= mins).all(0)
        margin = np.where(np.isnan(S), np.inf, S - mins)
        lim = np.array(names, dtype=object)[margin.argmin(0)]
        below = S < mins
        weak = np.full(S.shape[1:], "", dtype=object)
        for i, c in enumerate(names):
            weak = np.where(below[i], np.where(weak == "", c, weak + "+" + c), weak)
        weak_sup = np.full(S.shape[1:], "", dtype=object)
        for c, v in sup.items():
            weak_sup = np.where(cal[c] < v, np.where(weak_sup == "", c, weak_sup + "+" + c), weak_sup)
        status = np.where(nan, "not_applicable", np.where(avoid, "avoid", np.where(suff, "sufficient", "partial")))
        for i, m in enumerate(ent["entity_id"]):
            app_rows.append(dict(experiment=ex["id"], moon_zone_id=m, applicable=bool(app[i])))
            if not app[i]:
                continue
            for s in counts:
                counts[s][i] += status[i] == s
            rows.append(pd.DataFrame(dict(
                experiment=ex["id"], moon_zone_id=m, earth_tile_id=earth["patch_id"].to_numpy(), status=status[i],
                limiting_category=lim[i], min_margin=np.where(np.isfinite(margin.min(0)[i]), margin.min(0)[i], np.nan).astype(np.float32),
                weak_required=weak[i], weak_supporting=weak_sup[i])))
    out = pd.concat(rows, ignore_index=True)
    for c in ("experiment", "moon_zone_id", "earth_tile_id", "status", "limiting_category", "weak_required", "weak_supporting"):
        out[c] = out[c].astype("category")
    return out, pd.DataFrame(app_rows), counts, wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-suffix", default="", help="e.g. _smoke to read smoke-test embeddings")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    C.ensure_dirs()
    sfx = a.out_suffix
    moon, earth, zones = C.load_tables()
    ent = C.build_entities(moon, zones)
    E = C.prepare_earth(earth)
    anchors = C.load_anchors()
    exps_cfg = json.loads((C.CFG / "experiments.json").read_text(encoding="utf-8"))
    ent_pos = {e: i for i, e in enumerate(ent["entity_id"])}
    earth_pos = {e: i for i, e in enumerate(earth["patch_id"])}
    info, raw, cal, nt, stats, resid = {}, {}, {}, {}, {}, []

    def run(name, terms, prior, prior_b=0.0, fit=True):
        r, c, n, st, res = calibrate(name, terms, prior, anchors, ent_pos, earth_pos, prior_b=prior_b, fit=fit)
        c = apply_caps(name, c, ent, E)
        raw[name], cal[name], nt[name], stats[name] = r, c, n, st
        res["calibrated_score"] = c[res["entity_id"].map(ent_pos).to_numpy(int), res["earth_site"].map(earth_pos).to_numpy(int)]
        resid.append(res)

    t, p = terrain_terms(ent, E, moon, earth, zones, a.emb_suffix, info)
    run("terrain", t, p)
    t, p = geology_terms(ent, E, earth, a.emb_suffix, info)
    run("geology", t, p)
    run("thermal", *thermal_terms(ent, E))
    run("water_ice", *water_terms(ent, E))
    run("dust", *dust_terms(ent, E, info))
    run("biology", *biology_terms(ent, E), prior_b=1.0)
    t, p, cinfo = caves_terms(ent, E, earth)
    info["caves"] = cinfo
    run("caves", t, p, fit=False)
    ops = {"terrain": cal["terrain"], "thermal": cal["thermal"], "water_ice": cal["water_ice"],
           "slope_hazard": hazard_term(ent, E), "traverse": traverse_term(ent, E)}
    run("operations", ops, dict(terrain=0.35, thermal=0.10, water_ice=0.15, slope_hazard=0.25, traverse=0.15))

    # ---- calibration report
    R = pd.concat(resid, ignore_index=True)
    R["residual"] = R["calibrated_score"] - R["label"]
    R["flag_gt_0.3"] = R["residual"].abs() > 0.3
    R.to_csv(C.RES / f"anchor_residuals{sfx}.csv", index=False)
    summ = pd.DataFrame([dict(category=k, n_anchors=v["n_anchors"], mae=v.get("mae"), mae_loo=v.get("mae_loo"),
                              spearman_raw=v.get("spearman_raw"), spearman=v.get("spearman"), spearman_loo=v.get("spearman_loo"),
                              n_flagged=int(R.loc[R.category == C.CATS[k], "flag_gt_0.3"].sum()),
                              intercept=v.get("intercept"), weights=json.dumps(v.get("weights"))) for k, v in stats.items()])
    summ.to_csv(C.RES / f"calibration_summary{sfx}.csv", index=False)

    # ---- score tables
    L = long_table(ent, earth, raw, cal, nt)
    C.save_table(L, C.RES / f"category_scores{sfx}.parquet")
    for c in ("terrain", "geology"):
        d = L[L.category == c][["moon_zone_id", "earth_tile_id", "calibrated_score"]].rename(columns={"calibrated_score": f"{c}_score"})
        C.save_table(d, C.RES / f"{c}_scores{sfx}.parquet")

    # ---- experiments + composite
    RD, APP, counts, wsum = readiness(ent, earth, cal, exps_cfg["experiments"], exps_cfg.get("default_avoid_below", 0.0))
    C.save_table(RD, C.RES / f"experiment_readiness{sfx}.parquet")
    APP.to_csv(C.RES / f"experiment_applicability{sfx}.csv", index=False)
    Wm = np.stack([wsum[c] for c in cal])                                       # (cat, M)
    Sc = np.stack([cal[c] for c in cal])                                        # (cat, M, E)
    wgt = np.where(np.isfinite(Sc), Wm[:, :, None], 0.0)
    comp = np.where(wgt.sum(0) > 0, (np.nan_to_num(Sc) * wgt).sum(0) / np.maximum(wgt.sum(0), 1e-9), np.nan)
    tops = []
    for i, m in enumerate(ent["entity_id"]):
        order = np.argsort(-np.nan_to_num(comp[i], nan=-9))[:20]
        d = earth.iloc[order][["patch_id", "name", "source", "stratum", "center_lat", "center_lon"]].rename(columns={"patch_id": "earth_tile_id"}).copy()
        d.insert(0, "rank", np.arange(1, len(d) + 1))
        d.insert(0, "moon_kind", ent["kind"].iloc[i])
        d.insert(0, "moon_zone_id", m)
        d["composite"] = comp[i, order]
        for c in cal:
            d[c] = cal[c][i, order]
        for s in counts:
            d[f"n_{s}"] = counts[s][i, order]
        tops.append(d)
    TOP = pd.concat(tops, ignore_index=True)
    TOP.round(3).to_csv(C.RES / f"top_analogs{sfx}.csv", index=False)

    params = dict(score_form="calibrated = clip(intercept + sum_k weight_k * term_k, -1, 1), then caps", caps=CAPS,
                  subscore="1 - |d|/tol for |d| <= tol; -(1 - exp(-(|d|/tol - 1))) beyond (soft floor at -1)",
                  categories=stats, info=info, composite_weights={c: np.round(wsum[c] / max(Wm.sum(0).max(), 1e-9), 3).tolist() for c in cal},
                  entities=ent["entity_id"].tolist(), always_attached=exps_cfg.get("always_include"),
                  non_reproducible=[g.get("id") for g in exps_cfg.get("global_non_reproducible", [])])
    (C.MODELS / f"calibration_params{sfx}.json").write_text(json.dumps(params, indent=1, default=str))

    # ---- console report
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print("\n== terrain SSL variants (held-out blocks; lower = neighbours agree better with measured terrain) ==")
    print(pd.DataFrame(info.get("terrain_ssl_variants", [])).to_string(index=False) if info.get("terrain_ssl_variants") else "no embeddings")
    print("chosen:", info.get("terrain_ssl_chosen"), "| geology:", info.get("geology_embeddings"), "| caves:", info.get("caves"))
    print("\n== calibration on anchors ==")
    print(summ.drop(columns=["weights"]).to_string(index=False))
    for k, v in stats.items():
        print(f"  {k:<11} weights {v.get('weights')}  intercept {v.get('intercept')}")
    fl = R[R["flag_gt_0.3"]]
    print(f"\n== anchors off by more than 0.3: {len(fl)} of {len(R)} ==")
    print(fl[["entity_id", "earth_site", "category", "label", "calibrated_score", "residual"]].round(2).to_string(index=False))
    print("\n== score ranges (calibrated) ==")
    print(L.groupby("category", observed=True)["calibrated_score"].describe()[["count", "min", "50%", "max"]].round(2).to_string())
    print("\n== experiment readiness (pairs) ==")
    print(RD.groupby(["experiment", "status"], observed=True).size().unstack(fill_value=0).to_string())
    show = [e for e in ("haworth__landing", "haworth__ice", "shackleton_rim__landing", "mons_mouton__landing", "moon_tranquillitatis_pit__tile") if e in ent_pos]
    for m in show:
        d = TOP[(TOP.moon_zone_id == m) & (TOP["rank"] <= 5)]
        print(f"\n-- top 5 for {m} --")
        print(d[["rank", "name", "stratum", "center_lat", "center_lon", "composite"] + list(cal)].round(2).to_string(index=False))
    print(f"\nwrote results/: category_scores ({len(L):,} rows), experiment_readiness ({len(RD):,} rows), top_analogs ({len(TOP)} rows), "
          f"anchor_residuals, calibration_summary; models/calibration_params{sfx}.json")


if __name__ == "__main__":
    main()
