"""Step 3: geology text encoder. Contrastive fine-tuning of BAAI/bge-small-en-v1.5 on Moon and Earth descriptions.

  python -m ml.geology_ssl            # full run (Colab T4: a few minutes)
  python -m ml.geology_ssl --smoke    # tiny end-to-end check (CPU ok)

No expert anchors are used for training. Pairs come from lithology heuristics (ml/geo_text.py):
  pull together   mare unit <-> basaltic volcano / flood basalt;  highland unit <-> anorthosite / norite / troctolite;
                  crater or basin material <-> impact structure in a non-sedimentary target
  push apart      Moon unit <-> sedimentary, evaporite, silicic volcanic, granitic / metamorphic, ice
  left alone      partial matches (impact in sedimentary target, gabbro, basalt vs highland, loose sediment, no text)
Loss: multi-positive InfoNCE (temperature 0.07) Moon<->Earth both ways + a same-class Earth<->Earth term, on a
384->128 projection head (discarded) and on the CLS embedding itself. Last 4 of 12 layers trained.
Earth tiles named in any anchor and all tiles in held-out spatial blocks are excluded from training pairs.
Outputs: models/geology_bge_ssl.pt, embeddings/{moon,earth,moon_zone}_geology_384.npy, results/geology_ssl_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C
from . import geo_text as G

BASE = "BAAI/bge-small-en-v1.5"
NEG_PRIMARY = {"sed", "silicic", "granmet", "ice"}
EE_CLASSES = ["anorth", "impact", "basalt", "silicic", "sed", "granmet", "ice"]


def build_corpus():
    moon, earth, zones = C.load_tables()
    ent = C.build_entities(moon, zones)
    cat = pd.read_csv(C.DATA / "catalogs" / "moon_geologic_units.csv")
    mt, mc = G.moon_texts_for(moon["moon_geo_unit"], moon.get("geo_frac_mare", pd.Series(0.0, index=moon.index)), cat)
    zt, zc = G.moon_texts_for(ent["moon_geo_unit"], ent["geo_frac_mare"], cat)
    ice = earth["x_surface_ice_frac"] if "x_surface_ice_frac" in earth else earth["wc_frac_snow_ice"]
    et = [G.earth_text(r) for r in earth.to_dict("records")]
    ec = [G.earth_classes(t, s, i, G.earth_primary_text(r)) for t, s, i, r in zip(et, earth["source"], ice, earth.to_dict("records"))]
    return moon, earth, zones, ent, mt, mc, zt, zc, et, ec


def augment(text, rng):
    """Light text augmentation: drop / reorder sentences so the model cannot key on position or one phrase."""
    parts = [p.strip() for p in text.replace(";", ".").split(".") if p.strip()]
    if len(parts) <= 2:
        return text
    keep = [p for p in parts if rng.random() > 0.25] or parts[:1]
    if rng.random() < 0.5:
        head, tail = keep[:1], keep[1:]
        rng.shuffle(tail)
        keep = head + tail
    return ". ".join(keep) + "."


def nce(sim, pos, neg, tau):
    """Multi-positive InfoNCE over rows of sim. pos / neg: bool masks; everything else is ignored."""
    logits = sim / tau
    valid = pos | neg
    rows = pos.any(1) & neg.any(1)
    if not bool(rows.any()):
        return sim.sum() * 0
    den = torch.logsumexp(logits.masked_fill(~valid, -1e4), dim=1)
    num = torch.logsumexp(logits.masked_fill(~pos, -1e4), dim=1)
    return -(num - den)[rows].mean()


class Encoder(nn.Module):
    def __init__(self, base=BASE, train_last=4):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        self.bert = AutoModel.from_pretrained(base)
        self.head = nn.Linear(self.bert.config.hidden_size, 128)
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-train_last:]:
            for p in layer.parameters():
                p.requires_grad = True

    max_len = 256

    def encode(self, texts, device, max_len=None):
        max_len = max_len or self.max_len
        b = self.tok(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        return self.bert(**b).last_hidden_state[:, 0]

    @torch.no_grad()
    def embed(self, texts, device, amp, bs=128):
        self.eval()
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                h = self.encode([t if t.strip() else "unknown" for t in texts[i:i + bs]], device)
            out.append(F.normalize(h.float(), dim=1).cpu())
        return torch.cat(out).numpy()


def evaluate(enc, device, amp, u_text, R, e_text, va_idx, ent_text, anchors, earth_ids):
    """Held-out-block retrieval (mean average precision of heuristic positives) + anchor rank agreement."""
    Em = enc.embed(u_text, device, amp)
    Ee = enc.embed([e_text[i] for i in va_idx], device, amp)
    S = Em @ Ee.T
    aps = []
    for r in range(len(u_text)):
        rel = R[r, va_idx]
        keep = rel != 0
        if (rel == 1).sum() == 0 or (rel == -1).sum() == 0:
            continue
        order = np.argsort(-S[r, keep])
        y = (rel[keep][order] == 1).astype(float)
        aps.append(float((np.cumsum(y) / (np.arange(len(y)) + 1) * y).sum() / y.sum()))
    a = anchors[anchors["cat"] == "geology"]
    pos = {p: i for i, p in enumerate(earth_ids)}
    rows = [(ent_text[m], e_text[pos[e]], y) for m, e, y in zip(a["entity_id"], a["earth_site"], a["label"])
            if m in ent_text and e in pos and e_text[pos[e]].strip()]
    rho = float("nan")
    if len(rows) >= 4:
        A = enc.embed([r[0] for r in rows], device, amp)
        B = enc.embed([r[1] for r in rows], device, amp)
        rho = C.spearman((A * B).sum(1), [r[2] for r in rows])
    return dict(heldout_map=round(float(np.mean(aps)), 3) if aps else None, n_queries=len(aps),
                anchor_spearman=None if not np.isfinite(rho) else round(rho, 3), n_anchor_pairs=len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--bm", type=int, default=16, help="Moon texts per batch")
    ap.add_argument("--be", type=int, default=40, help="Earth texts per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-len", type=int, default=0, help="token limit while TRAINING (0 = 256); embeddings always use 256")
    ap.add_argument("--eval-every", type=int, default=5, help="held-out evaluation every N epochs (and after the last)")
    ap.add_argument("--before", type=json.loads, default=None,
                    help="reuse a known pretrained-baseline metrics dict (JSON) instead of re-evaluating; saves a full pass on CPU")
    a = ap.parse_args()
    C.ensure_dirs()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    moon, earth, zones, ent, mt, mc, zt, zc, et, ec = build_corpus()
    anchors = C.load_anchors()
    # unique Moon texts (tiles + entities share them)
    u_text, u_cls, seen = [], [], {}
    for t, c in list(zip(mt, mc)) + list(zip(zt, zc)):
        if t not in seen:
            seen[t] = len(u_text)
            u_text.append(t)
            u_cls.append(c)
    R = np.array([[G.relation(m, e) for e in ec] for m in u_cls], dtype=np.int8)        # (unique Moon, Earth)
    prim = np.array([e.get("primary", "unknown") for e in ec])
    known = np.array([bool(e.get("known")) for e in ec])
    blocks = C.spatial_blocks(earth["center_lat"], earth["center_lon"], np.full(len(earth), "earth"))
    held = C.holdout_mask(blocks, 0.15, a.seed)
    in_anchor = earth["patch_id"].isin(set(anchors["earth_site"])).to_numpy()
    tr_idx = np.where(known & ~held & ~in_anchor)[0]
    va_idx = np.where(known & held)[0]
    if a.smoke:                                   # CPU-sized check: a few texts per class, short sequences
        a.bm, a.be = 4, 10
        va_idx = np.random.RandomState(0).permutation(va_idx)[:40]
    fam = np.array([c["family"] + ("_impact" if c["impact"] else "") for c in u_cls])
    C.log(f"device={device} | Moon texts: {len(u_text)} unique {dict(pd.Series(fam).value_counts())}")
    C.log(f"Earth texts: {int(known.sum())} with geology text | train {len(tr_idx)} | held-out blocks {len(va_idx)} "
          f"| excluded as anchor sites {int((in_anchor & known).sum())}")
    C.log("Earth primary class (train):", dict(pd.Series(prim[tr_idx]).value_counts()))
    C.log("pairs (unique Moon x Earth): pull", int((R == 1).sum()), "| push", int((R == -1).sum()), "| ignore", int((R == 0).sum()))
    C.log("example Moon text:", u_text[0][:300])
    ex = int(tr_idx[np.argmax(prim[tr_idx] == "basalt")])
    C.log("example Earth text:", et[ex][:300])

    buckets = {k: tr_idx[prim[tr_idx] == k] for k in pd.unique(prim[tr_idx])}
    groups = [["anorth"], ["impact", "impact_sed"], ["basalt"], ["sed", "silicic", "granmet", "ice"],
              ["intermediate", "uncons", "other", "basalt_weak", "mafic_plut"]]
    groups = [[k for k in g if k in buckets and len(buckets[k])] for g in groups]
    groups = [g for g in groups if g]
    fam_idx = {f: np.where(fam == f)[0] for f in np.unique(fam)}

    enc = Encoder().to(device)
    if a.smoke and device.type == "cpu":
        enc.max_len = 96
    full_len = enc.max_len
    if a.max_len:
        enc.max_len = a.max_len
    params = [p for p in enc.bert.parameters() if p.requires_grad]
    C.log(f"params {sum(p.numel() for p in enc.bert.parameters()) / 1e6:.1f}M, trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW([dict(params=params, lr=a.lr), dict(params=enc.head.parameters(), lr=a.lr * 20)], weight_decay=0.01)
    steps_per_epoch = 3 if a.smoke else max(20, len(tr_idx) // a.be)
    epochs = 1 if a.smoke else a.epochs
    total = epochs * steps_per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ent_text = dict(zip(ent["entity_id"], zt))
    ev = lambda: evaluate(enc, device, amp, u_text, R, et, va_idx, ent_text, anchors, earth["patch_id"].tolist())  # noqa: E731
    metrics = dict(args=vars(a), n_train=int(len(tr_idx)), n_val=int(len(va_idx)), history=[])
    metrics["before"] = a.before if a.before else ev()
    C.log("BEFORE fine-tuning (pretrained BGE):", metrics["before"])
    ee_code = {k: i for i, k in enumerate(EE_CLASSES)}
    t0 = time.time()
    for ep in range(epochs):
        enc.train()
        agg = np.zeros(3)
        for _ in range(steps_per_epoch):
            mi = np.array([rng.choice(list(fam_idx[f])) for f in rng.choices(list(fam_idx), k=a.bm)])
            per = max(1, a.be // len(groups))
            ei = []
            for g in groups:
                pool = np.concatenate([buckets[k] for k in g])
                ei += [int(rng.choice(pool)) for _ in range(per)]
            ei = np.array(ei)
            texts = [augment(u_text[i], rng) for i in mi] + [augment(et[i], rng) for i in ei]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                h = enc.encode(texts, device)
                z = enc.head(h)
            rel = torch.from_numpy(R[np.ix_(mi, ei)]).to(device)
            code = torch.tensor([ee_code.get(prim[i], -1) for i in ei], device=device)
            same = (code[:, None] == code[None, :]) & (code[:, None] >= 0) & ~torch.eye(len(ei), dtype=torch.bool, device=device)
            diff = (code[:, None] != code[None, :]) & (code[:, None] >= 0) & (code[None, :] >= 0)
            loss, parts = 0, []
            for rep in (z, h):                                   # projection head and the CLS embedding itself
                r = F.normalize(rep.float(), dim=1)
                m, e = r[:len(mi)], r[len(mi):]
                s = m @ e.T
                l_me = nce(s, rel == 1, rel == -1, a.tau)
                l_em = nce(s.T, (rel == 1).T, (rel == -1).T, a.tau)
                l_ee = nce(e @ e.T, same, diff, a.tau)
                loss = loss + l_me + l_em + 0.5 * l_ee
                parts = [float(l_me), float(l_em), float(l_ee)]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg += np.array(parts)
        agg /= steps_per_epoch
        rec = dict(epoch=ep + 1, moon_to_earth=round(float(agg[0]), 4), earth_to_moon=round(float(agg[1]), 4),
                   earth_earth=round(float(agg[2]), 4), min=round((time.time() - t0) / 60, 1))
        if ep + 1 == epochs:
            enc.max_len = full_len                                  # evaluate + embed at the full length
        if (ep + 1) % a.eval_every == 0 or ep + 1 == epochs:
            rec["val"] = ev()
        metrics["history"].append(rec)
        C.log(rec)
    metrics["after"] = metrics["history"][-1]["val"]
    metrics["train_minutes"] = round((time.time() - t0) / 60, 1)

    # ---- embeddings aligned to the tables (unique texts embedded once)
    tag = "_smoke" if a.smoke else ""
    Eu = enc.embed(u_text, device, amp)
    np.save(C.EMB / f"moon_geology_384{tag}.npy", Eu[[seen[t] for t in mt]].astype(np.float32))
    np.save(C.EMB / f"moon_entity_geology_384{tag}.npy", Eu[[seen[t] for t in zt]].astype(np.float32))
    zone_rows = ent[ent["kind"] == "zone"].index.to_numpy()
    np.save(C.EMB / f"moon_zone_geology_384{tag}.npy", Eu[[seen[zt[i]] for i in zone_rows]].astype(np.float32))
    e_sel = np.arange(len(et)) if not a.smoke else np.unique(np.r_[tr_idx[:20], va_idx[:20], np.where(in_anchor)[0]])
    Ee = np.full((len(et), Eu.shape[1]), np.nan, np.float32)
    Ee[e_sel] = enc.embed([et[i] for i in e_sel], device, amp)
    Ee[~known] = np.nan                                         # no geology text -> no geology embedding
    np.save(C.EMB / f"earth_geology_384{tag}.npy", Ee)
    (C.EMB / f"geology_ids{tag}.json").write_text(json.dumps(dict(
        moon=moon["patch_id"].tolist(), earth=earth["patch_id"].tolist(), moon_entity=ent["entity_id"].tolist(),
        moon_zone=ent.loc[zone_rows, "entity_id"].tolist())))
    state = {k: v.half() if v.dtype == torch.float32 else v for k, v in enc.bert.state_dict().items()}
    torch.save(dict(base_model=BASE, pooling="cls, L2-normalised", weights_dtype="float16", train_last=4,
                    bert=state, head=enc.head.state_dict(), metrics=metrics), C.MODELS / f"geology_bge_ssl{tag}.pt")
    (C.RES / f"geology_ssl_metrics{tag}.json").write_text(json.dumps(metrics, indent=1, default=str))
    C.log("AFTER fine-tuning:", metrics["after"])
    C.log(f"saved embeddings/*_geology_384{tag}.npy (moon {len(mt)}, earth {len(et)}, entities {len(zt)}) "
          f"and models/geology_bge_ssl{tag}.pt in {metrics['train_minutes']} min")


if __name__ == "__main__":
    main()
