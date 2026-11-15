"""Regenerate docs/ML_RESULTS.md from the pipeline outputs.

Inputs : results/terrain_ssl_metrics.json, results/geology_ssl_metrics.json, results/calibration_summary.csv,
         results/anchor_residuals.csv, results/score_matrices.npz, results/top_analogs.csv,
         models/calibration_params.json
Usage  : python tools/make_results_doc.py            (run after ml.score_all)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, MOD, OUT = ROOT / "results", ROOT / "models", ROOT / "docs" / "ML_RESULTS.md"
FEATS = ("log_relief", "slope_mean", "log_tri")
SHOW_CATS = [("terrain", "Terrain"), ("geology", "Geology"), ("thermal", "Thermal"), ("water_ice", "Water/ice"),
             ("dust", "Dust"), ("operations", "Ops")]

LIMITS = """## 6. Deviations from the brief and known limits

- **Data:** the brief's counts were stale. Used 5,610 Moon tiles (3,609 polar + 2,001 mid-latitude), 151 anchors, 128 x 128 patches and 30 Moon targets.
- **Sub-score floor:** the brief's hard floor (`max(-1, 1 - |d|/tol)`) made every Earth site tie at -1 for temperature. Same formula within one tolerance, soft approach to -1 beyond, so closest-approach order survives.
- **Terrain:** added a CORAL alignment term, random-blur views and early stopping on held-out cross-body agreement (longer training made Moon-to-Earth matching worse).
- **Geology:** an Earth tile counts as anorthositic only when anorthosite/norite/troctolite is its first-listed Macrostrat lithology, its name, or a curated site tag. Granite/gneiss units that mention anorthosite later in the description are a partial match only.
- **Thermal:** six terms instead of four (added max temperature and noon sun height from `categories.json`). No Earth site reaches lunar temperatures, so the absolute thermal score is always low; `thermal_percentile` in the web bundle gives the closest-approach rank among Earth tiles instead. Scores for the Tranquillitatis pit are extrapolated: all thermal anchors are polar.
- **Water/ice:** permafrost under a tile counts as ice at distance zero; a liquid-water term was added.
- **Dust reference:** 45% fines (lunar soil median grain size ~70 um) instead of the brief's '70-80% sand-sized'.
- **Caves:** the rule levels (+0.5 / +0.3 / 0 / -0.5) are the score; anchors check it rather than rescale it.
- **Files:** `category_scores` and `experiment_readiness` are gzip CSV when no parquet engine is installed; `results/score_matrices.npz` holds the same numbers as arrays.
- **Not scored, always a caveat:** 1/6 gravity, vacuum, radiation, 25-110 K shadowed temperatures.
"""


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def triple(d):
    return " / ".join(f3(d.get(k)) for k in FEATS)


def models_section(tm, gm, info):
    b, a = tm["before"], tm["after"]
    ta = tm["args"]
    ga = gm["args"]
    rows = [
        f"| Terrain: `facebook/dinov2-small`, VICReg + CORAL, {ta['epochs']} epochs, best epoch {tm.get('best_epoch', '?')} | two-view instance retrieval (top-1) | {f3(b['instance_top1'])} | {f3(a['instance_top1'])} |",
        f"| | cross-body terrain agreement, relief / slope / roughness (1.0 = random, lower is better) | {triple(b['xbody_top10_vs_random']['pooled'])} | {triple(a['xbody_top10_vs_random']['pooled'])} |",
        f"| | linear probe R2 to measured relief / slope / roughness | {triple(b['probe_r2'])} | {triple(a['probe_r2'])} |",
        f"| | Moon-vs-Earth separability (AUC; 0.5 would be body-agnostic) | {b['moon_vs_earth_auc']:.4f} | {a['moon_vs_earth_auc']:.4f} |",
        f"| Geology: `BAAI/bge-small-en-v1.5`, InfoNCE, {ga['epochs']} epochs | retrieval of lithology matches (mAP) | {f3(gm['before']['heldout_map'])} | {f3(gm['after']['heldout_map'])} |",
        f"| | rank agreement with the {gm['after'].get('n_anchor_pairs', '')} geology anchors (never used in training) | {f3(gm['before']['anchor_spearman'])} | {f3(gm['after']['anchor_spearman'])} |",
    ]
    out = ["## 1. Fine-tuned models (evaluated on held-out spatial blocks)", "",
           "| Model | Measure | Pretrained | Fine-tuned |", "|---|---|---|---|", *rows, ""]
    ch = info.get("terrain_ssl_chosen")
    chosen = f"{ch.get('variant')} (agreement {ch.get('agreement')})" if isinstance(ch, dict) else str(ch)
    out.append("Terrain embeddings stay separable by body, so pooled training alone does not give body-agnostic features. "
               f"The scorer compares post-hoc alignments on held-out blocks and uses the best: **{chosen}**.")
    out += ["", "| Alignment | agreement (lower is better) |", "|---|---|"]
    out += [f"| {v['variant']} | {f3(v['agreement'])} |" for v in info.get("terrain_ssl_variants", [])]
    return out + [""]


def calibration_section(summ, R):
    out = ["## 2. Calibration against the expert anchors", "",
           "`calibrated = clip(intercept + sum(weight x term), -1, 1)` with non-negative weights fitted on the anchors of each "
           f"category (ridge towards prior weights). All {len(R)} anchors are still drafts (`needs_review=1`).", "",
           "| Category | Anchors | Mean abs. error | Leave-one-out error | Rank agreement | LOO rank agreement | Off by > 0.3 | Fitted term weights |",
           "|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        out.append(f"| {r.category} | {r.n_anchors} | {f3(r.mae)} | {f3(r.mae_loo)} | {f3(r.spearman)} | {f3(r.spearman_loo)} | "
                   f"{r.n_flagged} | `{r.weights}` |")
    fl = R[R["flag_gt_0.3"]]
    out += ["", f"### Anchors that disagree with the model by more than 0.3 ({len(fl)} of {len(R)}): review these first", "",
            "| Moon | Earth | Category | Label | Model | Difference |", "|---|---|---|---|---|---|"]
    for r in fl.itertuples():
        out.append(f"| {r.entity_id} | {r.earth_site} | {r.category} | {r.label:g} | {r.calibrated_score:.2f} | {r.residual:+.2f} |")
    return out + [""]


def ranges_section(Z):
    cats, cal, rd = list(Z["categories"]), Z["calibrated"], Z["readiness"]
    out = ["## 3. Score ranges and experiment readiness", "", "| Category | Min | Median | Max | Earth-tile pairs >= 0.5 |",
           "|---|---|---|---|---|"]
    for i, c in enumerate(cats):
        v = cal[i][np.isfinite(cal[i])]
        out.append(f"| {c} | {v.min():.2f} | {np.median(v):.2f} | {v.max():.2f} | {int((v >= 0.5).sum()):,} |")
    out += ["", "| Experiment | Sufficient | Partial | Avoid |", "|---|---|---|---|"]
    for j, e in enumerate(Z["experiments"]):
        r = rd[j]
        out.append(f"| {e} | {int((r == 3).sum()):,} | {int((r == 2).sum()):,} | {int((r == 1).sum()):,} |")
    zero = [str(e) for j, e in enumerate(Z["experiments"]) if int((rd[j] == 3).sum()) == 0]
    out += ["", "Counts are (Moon target, Earth tile) pairs."
            + (f" No Earth tile is sufficient for {', '.join(zero)}: water/ice is capped at 0.3 against cold-trap zones "
               "and no Earth site reaches lunar thermal conditions." if zero else ""), ""]
    return out


def top_section(top):
    out = ["## 4. Best Earth analog per Moon target (composite of the categories its experiments need)", "",
           "| Moon target | Best Earth tile | Type | Lat, lon | Composite | " + " | ".join(n for _, n in SHOW_CATS) + " |",
           "|---|---|---|---|---|" + "---|" * len(SHOW_CATS)]
    for r in top[top["rank"] == 1].itertuples():
        vals = " | ".join("n/a" if pd.isna(getattr(r, c)) else f"{getattr(r, c):.2f}" for c, _ in SHOW_CATS)
        out.append(f"| {r.moon_zone_id} | {r.name} | {r.stratum} | {r.center_lat:.2f}, {r.center_lon:.2f} | {r.composite:.2f} | {vals} |")
    return out + [""]


def composite_section(params):
    w = params.get("composite_weights", {})
    ents = params.get("entities", [])
    if not w or not ents:
        return []
    out = ["## 5. Composite weights", "",
           "Composite = weighted mean of category scores; each category's weight is the number of experiments relevant to "
           "that Moon target that require (1.0) or support (0.5) it, normalised. Typical weights (median over targets):", "",
           "| " + " | ".join(w) + " |", "|" + "---|" * len(w),
           "| " + " | ".join(f"{np.median(v):.2f}" for v in w.values()) + " |", ""]
    return out


def main():
    tm = json.loads((RES / "terrain_ssl_metrics.json").read_text())
    gm = json.loads((RES / "geology_ssl_metrics.json").read_text())
    params = json.loads((MOD / "calibration_params.json").read_text())
    summ = pd.read_csv(RES / "calibration_summary.csv")
    R = pd.read_csv(RES / "anchor_residuals.csv")
    top = pd.read_csv(RES / "top_analogs.csv")
    Z = np.load(RES / "score_matrices.npz", allow_pickle=True)
    n_m, n_e = Z["calibrated"].shape[1:]
    lines = ["# ML results: Earth-Moon analog scores", "",
             "Generated by `tools/make_results_doc.py` from the outputs of `ml/` (see the `ml/score_all.py` docstring for the score "
             f"definition). {n_m} Moon targets x {n_e:,} Earth tiles x {len(Z['categories'])} categories. Scores are in [-1, 1]: "
             ">= 0.5 sufficient, 0-0.5 partial, < 0 avoid.", ""]
    lines += models_section(tm, gm, params.get("info", {}))
    lines += calibration_section(summ, R)
    lines += ranges_section(Z)
    lines += top_section(top)
    lines += composite_section(params)
    lines += [LIMITS]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
