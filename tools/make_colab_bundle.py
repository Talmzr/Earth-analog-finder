#!/usr/bin/env python
"""
Lite Colab bundle (no big .npz arrays: Colab rebuilds them with tools/build_arrays_128.py).

  colab_upload/earth_analog_lite_bundle_v3.zip   (deflate, must stay < 9.5 MB)

Contents: config/*, data/ml/*.csv + manifest.json, data/patches/*.csv|*.md|*.geojson, data/catalogs/*,
data/external/*, docs/*.md|*.csv, docs/profile_figs/*.png, tools/*.py, extract.py, extract_extra.py,
DATA_SOURCES.md, notebooks/*.ipynb

Also regenerates colab_upload/colab_cell_v3_oneline.txt from colab_upload/colab_cell_v3.py
(one line, base64-wrapped, no trailing newline: paste into a single Colab cell).

  python tools/make_colab_bundle.py [--out colab_upload/earth_analog_lite_bundle_v3.zip]
"""
import argparse
import base64
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT_MB = 9.5
PATTERNS = ["config/*", "data/ml/*.csv", "data/ml/manifest.json",
            "data/patches/*.csv", "data/patches/*.md", "data/patches/*.geojson",
            "data/catalogs/*", "data/external/*",
            "docs/*.md", "docs/*.csv", "docs/profile_figs/*.png",
            "tools/*.py", "extract.py", "extract_extra.py", "DATA_SOURCES.md", "notebooks/*.ipynb"]


def files():
    seen, out = set(), []
    for pat in PATTERNS:
        for f in sorted(ROOT.glob(pat)):
            if f.is_file() and f.suffix != ".npz" and not f.name.endswith((".part", ".tmp")) and f not in seen:
                seen.add(f)
                out.append(f)
    return out


def write_oneline(cell=ROOT / "colab_upload" / "colab_cell_v3.py"):
    if not cell.exists():
        print(f"  (no {cell.name}: one-line cell not written)")
        return
    code = cell.read_text(encoding="utf-8")
    compile(code, str(cell), "exec")                     # syntax check before wrapping
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    line = f"exec(__import__('base64').b64decode('{b64}').decode())"
    out = cell.with_name(cell.stem + "_oneline.txt")
    out.write_bytes(line.encode("ascii"))                # single line, no trailing newline
    assert base64.b64decode(b64).decode("utf-8") == code
    print(f"cell -> {out} ({len(line)} chars, single line)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "colab_upload" / "earth_analog_lite_bundle_v3.zip"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.tmp")
    fl = files()
    raw = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in fl:
            zf.write(f, arcname=f.relative_to(ROOT).as_posix())
            raw += f.stat().st_size
    tmp.replace(out)
    mb = out.stat().st_size / 1e6
    by_dir = {}
    for f in fl:
        d = f.relative_to(ROOT).parent.as_posix()
        by_dir[d] = by_dir.get(d, 0) + 1
    print(f"bundle -> {out}")
    print(f"  {len(fl)} files, {raw / 1e6:.1f} MB raw -> {mb:.2f} MB zipped (limit {LIMIT_MB} MB)")
    print("  files per folder:", by_dir)
    write_oneline()
    if mb >= LIMIT_MB:
        print(f"ERROR: bundle is {mb:.2f} MB >= {LIMIT_MB} MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
