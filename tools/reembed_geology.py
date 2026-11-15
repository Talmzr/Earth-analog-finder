"""Re-embed Earth geology texts with the EXISTING fine-tuned geology model (no training).

Use after changing ml/geo_text.py text construction. Only rows whose text changed are embedded, so it takes seconds on
CPU; the texts behind the current embeddings are kept in embeddings/earth_geology_texts.json for the next diff.

  python tools/reembed_geology.py                      # diff against embeddings/earth_geology_texts.json
  python tools/reembed_geology.py --old-texts X.json   # diff against a saved list of texts (first run)
  python tools/reembed_geology.py --all                # re-embed every Earth text (~5 min on CPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml import common as C  # noqa: E402
from ml import geology_ssl as GS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-texts", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    store = C.EMB / "earth_geology_texts.json"
    moon, earth, zones, ent, mt, mc, zt, zc, et, ec = GS.build_corpus()
    known = np.array([bool(e.get("known")) for e in ec])
    if a.all:
        old = [None] * len(et)
    else:
        src = Path(a.old_texts) if a.old_texts else store
        old = json.loads(src.read_text(encoding="utf-8"))
        assert len(old) == len(et), "text list does not match the Earth table"
    changed = np.array([et[i] != old[i] for i in range(len(et))])
    E = np.load(C.EMB / "earth_geology_384.npy")
    ck = torch.load(C.MODELS / "geology_bge_ssl.pt", map_location="cpu", weights_only=False)
    enc = GS.Encoder()
    enc.bert.load_state_dict({k: v.float() if v.dtype == torch.float16 else v for k, v in ck["bert"].items()})
    enc.head.load_state_dict(ck["head"])
    idx = np.where(changed & known)[0]
    C.log(f"re-embedding {len(idx)} changed Earth texts (of {len(et)})")
    if len(idx):
        E[idx] = enc.embed([et[i] for i in idx], torch.device("cpu"), False)
    E[~known] = np.nan
    np.save(C.EMB / "earth_geology_384.npy", E.astype(np.float32))
    store.write_text(json.dumps(et), encoding="utf-8")
    C.log(f"saved embeddings/earth_geology_384.npy and {store.name}")


if __name__ == "__main__":
    main()
