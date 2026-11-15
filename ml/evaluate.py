"""Scorecard: one fixed benchmark that every scoring or model change must beat.

  python -W ignore -m ml.evaluate                    # after ml.score_all; writes results/scorecard.json
  python -W ignore -m ml.evaluate --out-suffix _v2   # score a variant run (ml.score_all --out-suffix _v2)
  python -W ignore -m ml.evaluate --compare results/scorecard.json

Two independent checks:
1. Category accuracy: site-grouped 5-fold cross-validation on the expert anchors (cv_pred in anchor_residuals.csv,
   from ml.score_all). Every anchor is predicted by a calibration that never saw its Earth site in any category, so
   a site cannot leak in through another Moon target. Folds are fixed by a hash of the site id (score_all.site_fold).
   Rule-defined scores (caves) and rule-capped anchors (KNOWN_LIFE) are not fitted, so their CV value is the score.
   Operations is built from the full-data terrain and water_ice scores (one level of leakage, kept for simplicity).
2. Overall-score sanity: the four Earth sites named as lunar south-pole analogs before any data was examined
   (POLAR_ANALOGS: Dry Valleys, Beacon Valley, Haughton, Svalbard) should outrank the other named sites for the
   29 south-pole targets. AUC = share of (polar, other) pairs ranked the right way; 0.5 = chance. There are no expert
   labels for the overall score, so this is a sanity check, not an accuracy measure.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import common as C

POLAR_ANALOGS = ("dry_valleys_wright", "beacon_valley", "haughton", "svalbard_bockfjorden")   # fixed 2026-11-15


def category_cv(R: pd.DataFrame) -> dict:
    out = {}
    for cat, d in R.groupby("category", sort=True):
        err = (d["cv_pred"] - d["label"]).abs()
        out[C.CFG2OUT.get(cat, cat)] = dict(n=int(len(d)), cv_mae=round(float(err.mean()), 3),
                                             cv_rank=round(C.spearman(d["cv_pred"], d["label"]), 3),
                                             cv_off_by_0_3=int((err > 0.3).sum()))
    return out


def polar_auc(npz, earth: pd.DataFrame) -> dict:
    ents, pid = list(npz["entity_id"]), list(npz["earth_tile_id"])
    comp = npz["composite"]
    named = [j for j, p in enumerate(pid) if earth.loc[p, "source"] == "named_site"]
    pos = [j for j in named if pid[j] in POLAR_ANALOGS]
    neg = [j for j in named if pid[j] not in POLAR_ANALOGS]
    polar_t = [i for i, m in enumerate(ents) if "tranquillitatis" not in m]
    aucs, ranks = [], []
    for i in polar_t:
        a, b = comp[i, pos][:, None], comp[i, neg][None, :]
        aucs.append(float(((a > b) + 0.5 * (a == b)).mean()))
        order = np.argsort(np.argsort(-np.nan_to_num(comp[i], nan=-9))) + 1
        ranks += order[pos].tolist()
    return dict(auc=round(float(np.mean(aucs)), 3), polar_sites_median_rank=int(np.median(ranks)),
                n_targets=len(polar_t), n_tiles=len(pid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--compare", default=None, help="an earlier scorecard JSON to print side by side")
    a = ap.parse_args()
    R = pd.read_csv(C.RES / f"anchor_residuals{a.out_suffix}.csv")
    if "cv_pred" not in R:
        raise SystemExit("anchor_residuals has no cv_pred column: re-run ml.score_all")
    npz = np.load(C.RES / f"score_matrices{a.out_suffix}.npz", allow_pickle=True)
    earth = C.load_tables()[1].set_index("patch_id")
    cats = category_cv(R)
    err = (R["cv_pred"] - R["label"]).abs()
    card = dict(categories=cats, all_anchors=dict(n=int(len(R)), cv_mae=round(float(err.mean()), 3),
                                                   cv_off_by_0_3=int((err > 0.3).sum())),
                overall_sanity=polar_auc(npz, earth))
    (C.RES / f"scorecard{a.out_suffix}.json").write_text(json.dumps(card, indent=1))
    old = json.loads(open(a.compare).read()) if a.compare else None
    print(f"{'category':13s} {'n':>3s}  {'CV error':>8s} {'CV rank':>8s} {'off>0.3':>7s}" + ("   | before: error rank off" if old else ""))
    for c, v in cats.items():
        line = f"{c:13s} {v['n']:3d}  {v['cv_mae']:8.3f} {v['cv_rank']:8.3f} {v['cv_off_by_0_3']:7d}"
        if old and c in old["categories"]:
            o = old["categories"][c]
            line += f"   | {o['cv_mae']:.3f} {o['cv_rank']:.3f} {o['cv_off_by_0_3']}"
        print(line)
    al, ov = card["all_anchors"], card["overall_sanity"]
    print(f"{'ALL':13s} {al['n']:3d}  {al['cv_mae']:8.3f} {'':8s} {al['cv_off_by_0_3']:7d}"
          + (f"   | {old['all_anchors']['cv_mae']:.3f} {'':5s} {old['all_anchors']['cv_off_by_0_3']}" if old else ""))
    print(f"overall score: polar-analog AUC {ov['auc']:.3f}, polar sites median rank {ov['polar_sites_median_rank']} "
          f"of {ov['n_tiles']}" + (f"   | before: AUC {old['overall_sanity']['auc']:.3f}, rank "
                                   f"{old['overall_sanity']['polar_sites_median_rank']}" if old else ""))


if __name__ == "__main__":
    main()
