#!/usr/bin/env python
"""
The 128 x 128 terrain image arrays are stored in the repo as <= 90 MB parts (GitHub's per-file limit is
100 MB). This script joins them back into data/patches/*.npz and verifies their SHA-256 checksums.

  python tools/reassemble_arrays.py            # rebuild all arrays listed in data/arrays/manifest.json
  python tools/reassemble_arrays.py --split    # (maintainers) re-split data/patches/*_128.npz into parts
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "arrays"
OUT = ROOT / "data" / "patches"
CHUNK = 90 * 1024 * 1024
FILES = ["moon_patch_arrays_128.npz", "earth_patch_arrays_128.npz", "moon_midlat_arrays_128.npz"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split():
    PARTS.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        src = OUT / name
        data = src.read_bytes()
        parts = []
        for i in range(0, len(data), CHUNK):
            p = PARTS / f"{name}.part{i // CHUNK:02d}"
            p.write_bytes(data[i:i + CHUNK])
            parts.append(p.name)
        manifest[name] = dict(size=len(data), sha256=sha256(src), parts=parts)
        print(f"split {name}: {len(data) / 1e6:.0f} MB -> {len(parts)} parts")
    (PARTS / "manifest.json").write_text(json.dumps(manifest, indent=1))


def join():
    manifest = json.loads((PARTS / "manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, m in manifest.items():
        dst = OUT / name
        if dst.exists() and dst.stat().st_size == m["size"] and sha256(dst) == m["sha256"]:
            print(f"ok (already present) {name}")
            continue
        with open(dst, "wb") as f:
            for p in m["parts"]:
                f.write((PARTS / p).read_bytes())
        ok = sha256(dst) == m["sha256"]
        print(f"{'ok' if ok else 'CHECKSUM MISMATCH'} {name} ({m['size'] / 1e6:.0f} MB from {len(m['parts'])} parts)")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true")
    split() if ap.parse_args().split else join()
