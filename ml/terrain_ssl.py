"""Step 2: terrain encoder. Self-supervised fine-tuning of facebook/dinov2-small on Moon + Earth DEM patches.

  python -m ml.terrain_ssl            # full run (Colab T4: ~25-35 min)
  python -m ml.terrain_ssl --smoke    # tiny end-to-end check (CPU ok)

Method (VICReg, no labels, anchors never used):
  input      2 channels per 10 km tile: elevation minus tile mean (asinh-compressed), slope in degrees;
             z-scored with statistics pooled over Moon + Earth; 128 px -> 224 px
  views      random 90-degree rotations / flips, random-resized crop, small Gaussian noise (no photometric aug)
  model      DINOv2 ViT-S/14, patch embedding adapted 3 -> 2 channels and frozen, first 6 blocks frozen,
             last 6 blocks + final norm trained, 2-layer projector (256 -> 128) used only for the loss
  pooling    one training set of Moon (polar + mid-latitude) and Earth tiles, so features are body-agnostic
  evaluation on HELD-OUT SPATIAL BLOCKS, before vs after fine-tuning: VICReg loss, two-view instance retrieval,
             linear probe to measured terrain (relief, slope, roughness), Moon-vs-Earth separability
Outputs: models/terrain_dinov2_ssl.pt, embeddings/{moon,earth,moon_zone}_terrain_384.npy, results/terrain_ssl_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common as C

BASE = "facebook/dinov2-small"


# ----------------------------------------------------------------------------- model
class TerrainEncoder(nn.Module):
    def __init__(self, base=BASE, n_frozen=6, in_ch=2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(base)
        pe = self.backbone.embeddings.patch_embeddings
        old = pe.projection
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride)
        with torch.no_grad():                       # reuse pretrained RGB filters for the 2 physical channels
            new.weight.copy_(old.weight[:, :in_ch] * (3.0 / in_ch))
            new.bias.copy_(old.bias)
        pe.projection = new
        pe.num_channels = in_ch
        self.backbone.config.num_channels = in_ch
        self.n_frozen = n_frozen
        self.dim = self.backbone.config.hidden_size
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[:n_frozen]:
            for p in blk.parameters():
                p.requires_grad = False

    @staticmethod
    def _blk(blk, h):
        out = blk(h)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, x):
        layers = self.backbone.encoder.layer
        with torch.no_grad():
            h = self.backbone.embeddings(x)
            for blk in layers[:self.n_frozen]:
                h = self._blk(blk, h)
        for blk in layers[self.n_frozen:]:
            h = self._blk(blk, h)
        h = self.backbone.layernorm(h)
        return 0.5 * (h[:, 0] + h[:, 1:].mean(1))          # CLS + mean patch token, 384-d

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def coral(a, b):
    """CORAL: squared distance between the means and between the covariances of two feature sets."""
    if len(a) < 4 or len(b) < 4:
        return a.sum() * 0
    ma, mb = a.mean(0), b.mean(0)
    ca = (a - ma).T @ (a - ma) / (len(a) - 1)
    cb = (b - mb).T @ (b - mb) / (len(b) - 1)
    return (ma - mb).pow(2).mean() + (ca - cb).pow(2).mean()


def projector(dim, hidden=256, out=128):
    return nn.Sequential(nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))


def vicreg(z1, z2, sim=25.0, var=25.0, cov=1.0):
    z1, z2 = z1.float(), z2.float()
    inv = F.mse_loss(z1, z2)
    z1, z2 = z1 - z1.mean(0), z2 - z2.mean(0)
    v = F.relu(1 - torch.sqrt(z1.var(0) + 1e-4)).mean() + F.relu(1 - torch.sqrt(z2.var(0) + 1e-4)).mean()
    n, d = z1.shape

    def offdiag(z):
        c = (z.T @ z) / max(n - 1, 1)
        return (c.pow(2).sum() - c.diagonal().pow(2).sum()) / d
    co = offdiag(z1) + offdiag(z2)
    return sim * inv + var * v + cov * co, (float(inv), float(v), float(co))


# ----------------------------------------------------------------------------- data on device
class Prep:
    """Physical channels -> network input. Statistics pooled over both bodies (saved in the checkpoint)."""

    def __init__(self, stats):
        self.s = stats

    @staticmethod
    def fit(X, idx, device):
        x = X[idx].to(device).float()
        e, s = torch.asinh(x[:, 0] / 50.0), x[:, 1]
        return dict(e_mean=float(e.mean()), e_std=float(e.std()), s_mean=float(s.mean()), s_std=float(s.std()))

    def __call__(self, xb):
        xb = xb.float()
        e = (torch.asinh(xb[:, 0] / 50.0) - self.s["e_mean"]) / self.s["e_std"]
        s = (xb[:, 1] - self.s["s_mean"]) / self.s["s_std"]
        return torch.stack([e, s], 1).clamp(-6, 6)


def gaussian_blur(x, sigma, k=7):
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    c = x.shape[1]
    x = F.pad(x, (k // 2,) * 4, mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def views(x, out, gen=None, min_area=0.45, noise=(0.02, 0.05)):
    """One random view per sample: random resolution (blur), dihedral transform + random-resized crop
    (single grid_sample), small noise. The blur makes the encoder ignore pixel-scale sharpness, which differs
    between the LOLA and Copernicus DEMs for sensor reasons rather than terrain reasons."""
    B, dev = x.shape[0], x.device
    r = lambda *shape: torch.rand(*shape, device=dev, generator=gen)  # noqa: E731
    u = r(B)
    x = x.clone()
    for sigma, mask in ((0.8, (u >= 0.5) & (u < 0.75)), (1.5, u >= 0.75)):
        if bool(mask.any()):
            x[mask] = gaussian_blur(x[mask], sigma)
    side = (min_area + (1 - min_area) * r(B)).sqrt()
    tx, ty = (2 * r(B) - 1) * (1 - side), (2 * r(B) - 1) * (1 - side)
    fx = torch.where(r(B) < 0.5, -1.0, 1.0) * side
    fy = torch.where(r(B) < 0.5, -1.0, 1.0) * side
    swap = r(B) < 0.5
    zero = torch.zeros(B, device=dev)
    row0 = torch.stack([torch.where(swap, zero, fx), torch.where(swap, fx, zero), tx], 1)
    row1 = torch.stack([torch.where(swap, fy, zero), torch.where(swap, zero, fy), ty], 1)
    theta = torch.stack([row0, row1], 1)
    grid = F.affine_grid(theta, (B, x.shape[1], out, out), align_corners=False)
    v = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    sigma = noise[0] + (noise[1] - noise[0]) * r(B, 1, 1, 1)
    return v + sigma * torch.randn(v.shape, device=dev, generator=gen)


def dihedral(x, k):
    x = torch.rot90(x, k % 4, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if k >= 4 else x


@torch.no_grad()
def embed(enc, prep, X, idx, img, device, amp, tta=1, bs=256):
    enc.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = prep(X[idx[i:i + bs]].to(device))
        acc = 0
        for k in range(tta):
            xk = F.interpolate(dihedral(xb, k), size=(img, img), mode="bilinear", align_corners=False)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                acc = acc + enc(xk).float()
        out.append((acc / tta).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- evaluation (held-out blocks)
def ridge_r2(Etr, Ytr, Eva, Yva, lam=1.0):
    mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
    A, B = (Etr - mu) / sd, (Eva - mu) / sd
    ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) / 100, A.T @ (Ytr - ym))
    P = B @ W + ym
    return (1 - ((Yva - P) ** 2).sum(0) / ((Yva - Yva.mean(0)) ** 2).sum(0)).tolist()


def domain_auc(Etr, btr, Eva, bva):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Etr)
        clf = LogisticRegression(max_iter=300, C=0.1).fit(sc.transform(Etr), btr)
        return float(roc_auc_score(bva, clf.predict_proba(sc.transform(Eva))[:, 1]))
    except Exception as ex:  # noqa: BLE001
        return f"n/a ({type(ex).__name__})"


@torch.no_grad()
def evaluate(enc, proj, prep, X, meta, tr, va, img, device, amp, seed=123):
    enc.eval()
    proj.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    vi = va[:1536]
    z1, z2, h1, h2 = [], [], [], []
    for i in range(0, len(vi), 128):
        xb = prep(X[vi[i:i + 128]].to(device))
        a, b = views(xb, img, gen), views(xb, img, gen)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            ha, hb = enc(a).float(), enc(b).float()
            za, zb = proj(ha).float(), proj(hb).float()
        h1.append(ha), h2.append(hb), z1.append(za), z2.append(zb)
    h1, h2, z1, z2 = torch.cat(h1), torch.cat(h2), torch.cat(z1), torch.cat(z2)
    loss, _ = vicreg(z1, z2)
    sim = F.normalize(h1 - h1.mean(0), dim=1) @ F.normalize(h2 - h1.mean(0), dim=1).T
    top1 = float((sim.argmax(1) == torch.arange(len(vi), device=device)).float().mean())
    tri = tr[np.random.RandomState(0).permutation(len(tr))[:4000]]
    Etr, Eva = embed(enc, prep, X, tri, img, device, amp), embed(enc, prep, X, va, img, device, amp)
    cols = ["log_relief", "slope_mean", "log_tri"]
    r2 = ridge_r2(Etr, meta.loc[tri, cols].to_numpy(float), Eva, meta.loc[va, cols].to_numpy(float))
    auc = domain_auc(Etr, (meta.loc[tri, "body"] == "earth").to_numpy(int), Eva, (meta.loc[va, "body"] == "earth").to_numpy(int))
    return dict(val_vicreg=round(float(loss), 3), instance_top1=round(top1, 3),
                probe_r2=dict(zip(cols, [round(v, 3) for v in r2])), moon_vs_earth_auc=auc,
                xbody_top10_vs_random=cross_body_agreement(Eva, meta.loc[va], cols))


def cross_body_agreement(E, m, cols, k=10):
    """For each held-out Moon tile: mean |difference| in measured terrain between it and its k most similar
    held-out Earth tiles, divided by the same for random Earth tiles. < 1 means cross-body similarity carries
    real terrain information; 1 means none. Reported for pooled and for per-body centring of the embeddings."""
    bm = (m["body"] != "earth").to_numpy()
    if bm.sum() < 5 or (~bm).sum() <= k:
        return "n/a"
    f = m[cols].to_numpy(float)
    fm, fe = f[bm], f[~bm]
    rand = np.abs(fm[:, None, :] - fe[None, :, :]).mean(axis=(0, 1))
    out = {}
    for name, (cm, ce) in dict(pooled=(E.mean(0), E.mean(0)), per_body=(E[bm].mean(0), E[~bm].mean(0))).items():
        A, B = E[bm] - cm, E[~bm] - ce
        A, B = A / np.linalg.norm(A, axis=1, keepdims=True), B / np.linalg.norm(B, axis=1, keepdims=True)
        top = np.argsort(-(A @ B.T), axis=1)[:, :k]
        near = np.abs(fm[:, None, :] - fe[top]).mean(axis=(0, 1))
        out[name] = dict(zip(cols, [round(float(v), 3) for v in near / rand]))
    return out


# ----------------------------------------------------------------------------- main
def build_meta(ids, body):
    moon, earth, zones = C.load_tables()
    cols = ["patch_id", "center_lat", "center_lon", "relief_m", "slope500_mean_deg", "tri500_mean_m"]
    t = pd.concat([moon[cols], earth[cols]], ignore_index=True)
    zp = pd.read_csv(C.ML / "moon_zone_patch_terrain.csv")
    zz = pd.DataFrame(dict(patch_id=zp["entity_id"], center_lat=zones["zone_centroid_lat"].values,
                           center_lon=zones["zone_centroid_lon"].values, relief_m=zp["patch_relief_m"],
                           slope500_mean_deg=zp["patch_slope500_mean_deg"], tri500_mean_m=zp["patch_tri500_mean_m"]))
    t = pd.concat([t, zz], ignore_index=True).drop_duplicates("patch_id").set_index("patch_id")
    m = t.reindex(ids).reset_index()
    m["body"] = body
    m["log_relief"] = np.log10(1 + m["relief_m"].clip(lower=0))
    m["slope_mean"] = m["slope500_mean_deg"]
    m["log_tri"] = np.log10(1 + m["tri500_mean_m"].clip(lower=0))
    m["block"] = C.spatial_blocks(m["center_lat"], m["center_lon"], np.where(m["body"] == "earth", "earth", "moon"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--frozen", type=int, default=6)
    ap.add_argument("--tta", type=int, default=8, help="dihedral test-time views averaged in the final embeddings")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--coral", type=float, default=0.0,
                    help="weight of the Moon/Earth alignment term (CORAL on the 384-d features); 0 = plain VICReg")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. 'coral'")
    ap.add_argument("--eval-every", type=int, default=2,
                    help="held-out evaluation every N epochs; the checkpoint with the best cross-body agreement is kept")
    a = ap.parse_args()
    C.ensure_dirs()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if a.smoke:
        a.epochs, a.batch, a.tta = 1, 16, 2
        a.img = a.img if device.type == "cuda" else 112
    C.log(f"device={device} amp={amp} epochs={a.epochs} batch={a.batch} img={a.img} smoke={a.smoke}")

    ids, body, Xnp, vf = C.load_terrain_arrays()
    meta = build_meta(ids, body)
    usable = (meta["body"] != "zone").to_numpy() & (vf >= 0.6) & meta["relief_m"].notna().to_numpy()
    held = C.holdout_mask(meta["block"].to_numpy(), 0.15, a.seed)
    tr, va = np.where(usable & ~held)[0], np.where(usable & held)[0]
    if a.smoke:
        rs = np.random.RandomState(0)
        tr, va = rs.permutation(tr)[:160], rs.permutation(va)[:64]
    C.log(f"patches {len(ids)} | train {len(tr)} | held-out blocks {len(va)} "
          f"(moon {int((meta.loc[va, 'body'] == 'moon').sum())}, earth {int((meta.loc[va, 'body'] == 'earth').sum())})")
    X = torch.from_numpy(Xnp)
    X = X.to(device) if device.type == "cuda" else X
    prep = Prep(Prep.fit(X, tr[:3000], device))
    C.log("input stats", {k: round(v, 3) for k, v in prep.s.items()})

    enc = TerrainEncoder(n_frozen=a.frozen).to(device)
    proj = projector(enc.dim).to(device)
    n_tr = sum(p.numel() for p in enc.trainable())
    C.log(f"backbone params {sum(p.numel() for p in enc.parameters()) / 1e6:.1f}M, trainable {n_tr / 1e6:.1f}M (+ projector)")
    opt = torch.optim.AdamW([dict(params=enc.trainable(), lr=a.lr), dict(params=proj.parameters(), lr=a.lr * 10)],
                            weight_decay=0.04)
    steps_per_epoch = max(1, len(tr) // a.batch)
    total = a.epochs * steps_per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    earth_np = (meta["body"] == "earth").to_numpy()
    metrics = dict(args=vars(a), n_train=int(len(tr)), n_val=int(len(va)), history=[])
    metrics["before"] = evaluate(enc, proj, prep, X, meta, tr, va, a.img, device, amp)
    C.log("BEFORE fine-tuning (pretrained DINOv2, untrained projector):", metrics["before"])
    t0 = time.time()
    best = None
    for ep in range(a.epochs):
        enc.train()
        proj.train()
        enc.backbone.embeddings.eval()
        perm = torch.randperm(len(tr)).numpy()
        agg = np.zeros(4)
        for s in range(steps_per_epoch):
            idx = tr[perm[s * a.batch:(s + 1) * a.batch]]
            xb = prep(X[idx].to(device))
            v = torch.cat([views(xb, a.img), views(xb, a.img)])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                h = enc(v)
                z = proj(h)
            loss, parts = vicreg(z[:len(idx)], z[len(idx):])
            if a.coral > 0:                       # make Moon and Earth feature statistics match
                is_e = torch.from_numpy(np.tile(earth_np[idx], 2)).to(device)
                loss = loss + a.coral * coral(h.float()[~is_e], h.float()[is_e])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg += np.array([float(loss), *parts])
        agg /= steps_per_epoch
        rec = dict(epoch=ep + 1, loss=round(float(agg[0]), 4), inv=round(float(agg[1]), 4), var=round(float(agg[2]), 4), cov=round(float(agg[3]), 4),
                   lr=round(sched.get_last_lr()[0], 7), min=round((time.time() - t0) / 60, 1))
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            rec["val"] = evaluate(enc, proj, prep, X, meta, tr, va, a.img, device, amp)
            xb = rec["val"].get("xbody_top10_vs_random")
            score = float(np.mean(list(xb["pooled"].values()))) if isinstance(xb, dict) else float("inf")
            if best is None or score < best[0]:      # unsupervised early stopping: held-out Moon->Earth agreement
                best = (score, ep + 1, {k: v.detach().clone() for k, v in enc.state_dict().items()},
                        {k: v.detach().clone() for k, v in proj.state_dict().items()}, rec["val"])
        metrics["history"].append(rec)
        C.log(rec)
    if best is not None:
        enc.load_state_dict(best[2])
        proj.load_state_dict(best[3])
        metrics["best_epoch"], metrics["after"] = best[1], best[4]
        C.log(f"kept checkpoint from epoch {best[1]} (cross-body agreement {best[0]:.3f})")
    else:
        metrics["after"] = metrics["history"][-1]["val"]
    metrics["train_minutes"] = round((time.time() - t0) / 60, 1)

    # ---- embeddings for every tile, aligned to the tables
    C.log("extracting embeddings (dihedral TTA x%d) ..." % a.tta)
    sel = np.arange(len(ids)) if not (a.smoke and device.type == "cpu") else np.unique(np.r_[tr, va, np.where(body == "zone")[0]])
    Eall = embed(enc, prep, X, sel, a.img, device, amp, tta=a.tta)
    pos = {ids[j]: i for i, j in enumerate(sel)}
    moon, earth, zones = C.load_tables()
    zone_ids = (zones["region_id"] + "__" + zones["zone"]).tolist()
    tag = ("_" + a.tag if a.tag else "") + ("_smoke" if a.smoke else "")
    for name, id_list in (("moon", moon["patch_id"].tolist()), ("earth", earth["patch_id"].tolist()), ("moon_zone", zone_ids)):
        M = np.stack([Eall[pos[p]] if p in pos else np.full(Eall.shape[1], np.nan, np.float32) for p in id_list])
        np.save(C.EMB / f"{name}_terrain_384{tag}.npy", M.astype(np.float32))
        C.log(f"saved embeddings/{name}_terrain_384{tag}.npy {M.shape}")
    (C.EMB / f"terrain_ids{tag}.json").write_text(json.dumps(dict(moon=moon["patch_id"].tolist(), earth=earth["patch_id"].tolist(), moon_zone=zone_ids)))
    torch.save(dict(base_model=BASE, in_channels=2, n_frozen=a.frozen, img=a.img, pooling="0.5*(cls+mean_patch)",
                    backbone=enc.backbone.state_dict(), projector=proj.state_dict(), input_stats=prep.s,
                    metrics=metrics), C.MODELS / f"terrain_dinov2_ssl{tag}.pt")
    (C.RES / f"terrain_ssl_metrics{tag}.json").write_text(json.dumps(metrics, indent=1, default=str))
    C.log("AFTER fine-tuning:", metrics["after"])
    C.log(f"done in {metrics['train_minutes']} min -> models/terrain_dinov2_ssl{tag}.pt")


if __name__ == "__main__":
    main()
