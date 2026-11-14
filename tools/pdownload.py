"""Parallel HTTP range downloader: python tools/pdownload.py URL OUTFILE [connections]"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests


def main(url, out, n=8):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("exists", out)
        return
    s = requests.Session()
    size = int(s.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])
    part = (size + n - 1) // n
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.truncate(size)

    def get(i):
        a, b = i * part, min(size, (i + 1) * part) - 1
        for attempt in range(6):
            try:
                pos = a
                with s.get(url, headers={"Range": f"bytes={pos}-{b}"}, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "r+b") as f:
                        f.seek(pos)
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            pos += len(chunk)
                if pos == b + 1:
                    return
                a = pos
            except Exception as e:  # noqa: BLE001
                print("retry", i, attempt, e, flush=True)
        raise RuntimeError(f"part {i} failed")

    with ThreadPoolExecutor(n) as ex:
        list(ex.map(get, range(n)))
    os.replace(tmp, out)
    print("done", out, size, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
