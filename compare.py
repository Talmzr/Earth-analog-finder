#!/usr/bin/env python
"""
Score every Earth site against every Moon/Mars target, per category and per mission task.

Category scores (0..1, higher = more analogous)
  terrain       exp(-d/sqrt(k)), d = Euclidean distance in z-scored feature space
                (relief and TRI log-transformed; slopes at a common 500 m baseline)
  illumination  same method on solar-geometry features
  environment   'closest approach': Earth can never be as cold/dry/airless as the Moon or Mars,
                so each Earth site gets the mean percentile (among Earth sites) in the target's
                direction - drier, colder, less vegetated, larger day/night swing
  geology       0.8 * recall(target traits present at site) + 0.2 * Jaccard; traits no Earth site
                can have (vacuum, thin CO2 atmosphere) are excluded
Task scores = weighted mean of category scores (weights in config/scoring.json; NaN categories
are dropped and the remaining weights renormalised).

Outputs in results/:
  scores_long.csv           one row per (target, Earth site)
  top5_<task>.csv           best 5 Earth sites per target for each task
  best_by_category.csv      best Earth site per target per category
  heatmap_<task>.png        targets x Earth sites
  report.md                 inference summary: best analogs, new candidates that beat
                            established ones, and gaps no Earth site covers
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "config"
DATA = ROOT / "data"
RES = ROOT / "results"
UNMATCHABLE = {"vacuum", "thin_co2_atmosphere"}
LOG_FEATURES = {"relief_m", "tri500_mean_m"}
CATS = ["terrain", "environment", "illumination", "geology"]


def zspace(all_df, feats):
    X = all_df[feats].astype(float).copy()
    for f in feats:
        if f in LOG_FEATURES:
            X[f] = np.log10(1 + X[f].clip(lower=0))
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    return (X - mu) / sd


def distance_score(zt, ze):
    """zt: (k,) target vector, ze: (n,k) Earth matrix -> (n,) similarity, NaN-aware."""
    diff = ze - zt
    mask = np.isfinite(diff)
    k = mask.sum(axis=1)
    d = np.sqrt(np.where(mask, diff, 0) ** 2 @ np.ones(diff.shape[1]))
    with np.errstate(all="ignore"):
        s = np.exp(-d / np.sqrt(np.maximum(k, 1)))
    s[k < max(2, diff.shape[1] // 2)] = np.nan  # too few shared features
    return s


def tags(s):
    return {t.strip() for t in str(s).split(";") if t.strip() and str(s) != "nan"}


def geology_score(ttags, etags):
    t = ttags - UNMATCHABLE
    if not t:
        return np.nan
    inter = len(t & etags)
    return 0.8 * inter / len(t) + 0.2 * inter / len(t | etags)


def env_scores(earth, directions):
    parts = []
    for f, d in directions.items():
        if f not in earth or d == "ignore":
            continue
        r = earth[f].rank(pct=True, ascending=(d == "high"))  # 1 = most extreme in target dir
        parts.append(r)
    if not parts:
        return pd.Series(np.nan, index=earth.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def light_twin_month(earth_row, target_noon_mean):
    cols = [f"noon_elev_m{m:02d}" for m in range(1, 13)]
    if not all(c in earth_row for c in cols):
        return ""
    v = np.array([earth_row[c] for c in cols], dtype=float)
    m = int(np.argmin(np.abs(v - target_noon_mean)))
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m]


def heatmap(pivot, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(10, 0.38 * pivot.shape[1] + 4), 0.42 * pivot.shape[0] + 2.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        j = np.nanargmax(pivot.values[i]) if np.isfinite(pivot.values[i]).any() else None
        if j is not None:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.5))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, label="analog score (0-1)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    cfg = json.loads((CFG / "scoring.json").read_text())
    tcfg = pd.read_csv(CFG / "targets.csv")
    ecfg = pd.read_csv(CFG / "earth_sites.csv")
    tf = pd.read_csv(DATA / "target_features.csv")
    ef = pd.read_csv(DATA / "earth_features.csv")
    T = tcfg.merge(tf.drop(columns=["name", "lat", "lon", "body"], errors="ignore"), on="id")
    E = ecfg.merge(ef.drop(columns=["name", "lat", "lon", "body"], errors="ignore"), on="id")
    RES.mkdir(exist_ok=True)

    # --- terrain & illumination: shared z-space across targets + Earth
    sim = {}
    for cat, key in (("terrain", "terrain_features"), ("illumination", "illumination_features")):
        feats = [f for f in cfg[key] if f in T.columns or f in E.columns]
        for f in feats:
            for df in (T, E):
                if f not in df:
                    df[f] = np.nan
        Z = zspace(pd.concat([T[feats], E[feats]], ignore_index=True), feats)
        ZT, ZE = Z.iloc[:len(T)].values, Z.iloc[len(T):].values
        sim[cat] = np.vstack([distance_score(ZT[i], ZE) for i in range(len(T))])

    env = env_scores(E, cfg["environment"]["directions"]).values
    sim["environment"] = np.tile(env, (len(T), 1))
    sim["geology"] = np.array([[geology_score(tags(t), tags(e)) for e in E["tags"]] for t in T["tags"]])

    # --- long table
    rows = []
    for i, t in T.iterrows():
        for j, e in E.iterrows():
            r = dict(target=t["id"], target_name=t["name"], body=t["body"], target_role=t["role"],
                     site=e["id"], site_name=e["name"], country=e["country"], site_status=e["status"],
                     site_lat=e["lat"], site_lon=e["lon"])
            for c in CATS:
                r[c] = sim[c][i, j]
            for task, w in cfg["tasks"].items():
                num = sum(w[c] * r[c] for c in CATS if np.isfinite(r[c]) and w[c] > 0)
                den = sum(w[c] for c in CATS if np.isfinite(r[c]) and w[c] > 0)
                r[f"score_{task}"] = num / den if den else np.nan
            r["light_twin_month"] = light_twin_month(e, t.get("noon_elev_mean_deg", np.nan))
            rows.append(r)
    L = pd.DataFrame(rows)
    L.to_csv(RES / "scores_long.csv", index=False, float_format="%.4f")

    # --- per-task top 5 + heatmaps
    tasks = list(cfg["tasks"])
    for task in tasks:
        col = f"score_{task}"
        top = (L.sort_values(col, ascending=False).groupby("target").head(5)
               .sort_values(["target", col], ascending=[True, False]))
        top[["target", "target_name", "site", "site_name", "country", "site_status", col] + CATS +
            ["light_twin_month"]].to_csv(RES / f"top5_{task}.csv", index=False, float_format="%.3f")
        pv = L.pivot(index="target", columns="site", values=col).loc[T["id"], E["id"]]
        heatmap(pv, f"Earth analog score - task: {task}", RES / f"heatmap_{task}.png")

    # --- best per category
    best = []
    for c in CATS:
        g = L.dropna(subset=[c]).sort_values(c, ascending=False).groupby("target").head(1)
        for _, r in g.iterrows():
            best.append(dict(target=r["target"], category=c, best_site=r["site_name"], score=r[c]))
    B = pd.DataFrame(best)
    B.to_csv(RES / "best_by_category.csv", index=False, float_format="%.3f")

    # --- report
    lines = ["# Earth analog inference report", "",
             f"{len(T)} targets x {len(E)} Earth sites "
             f"({(E['status'] == 'established').sum()} established, {(E['status'] == 'candidate').sum()} candidate).", "",
             "Scores are 0-1 and relative to this site set; environment is a 'closest approach' rank because "
             "no Earth site reaches lunar or Martian cold, dryness or pressure.", ""]
    lines += ["## Best overall analog per target", "",
              "| Target | Best Earth site | Score | Weakest category (best available) | Light-twin month |",
              "|---|---|---|---|---|"]
    for tid in T["id"]:
        sub = L[L["target"] == tid].sort_values("score_overall", ascending=False)
        if sub.empty or not np.isfinite(sub["score_overall"].iloc[0]):
            continue
        b = sub.iloc[0]
        cat_best = {c: L.loc[L["target"] == tid, c].max() for c in CATS}
        cat_best = {c: v for c, v in cat_best.items() if np.isfinite(v)}
        weak = min(cat_best, key=cat_best.get) if cat_best else "-"
        lines.append(f"| {b['target_name']} | {b['site_name']} | {b['score_overall']:.2f} | "
                     f"{weak} ({cat_best.get(weak, float('nan')):.2f}) | {b['light_twin_month']} |")

    lines += ["", "## New candidates that beat every established analog", "",
              "Per target and task, candidate sites scoring above the best *established* site.", ""]
    found = False
    for task in tasks:
        col = f"score_{task}"
        for tid in T["id"]:
            sub = L[L["target"] == tid]
            est = sub.loc[sub["site_status"] == "established", col].max()
            cand = sub[(sub["site_status"] == "candidate") & (sub[col] > est)].sort_values(col, ascending=False)
            if len(cand):
                found = True
                names = ", ".join(f"{r.site_name} ({r[col]:.2f})" for _, r in cand.head(3).iterrows())
                lines.append(f"- **{task}**, {sub['target_name'].iloc[0]}: {names}  - best established {est:.2f}")
    if not found:
        lines.append("- none: established analogs win everywhere in this run")

    lines += ["", "## Category coverage gaps", "",
              "Mean over targets of the best available Earth score per category (low = Earth analogs fall short).", ""]
    for c in CATS:
        v = B[B["category"] == c]["score"]
        if len(v):
            lines.append(f"- {c}: {v.mean():.2f} (min {v.min():.2f})")
    lines += ["", "## Per-body view", ""]
    for body in sorted(L["body"].unique()):
        sub = L[L["body"] == body].groupby("site_name")["score_overall"].mean().sort_values(ascending=False)
        lines.append(f"- **{body}** - top 5 Earth sites on average: " +
                     ", ".join(f"{k} ({v:.2f})" for k, v in sub.head(5).items()))
    (RES / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote results to {RES}")


if __name__ == "__main__":
    sys.exit(main())
