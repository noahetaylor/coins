#!/usr/bin/env python3
"""
Build a DateBands variant where the studio (2026data) coins come from the
raking-light rig captures instead of the old 2026Data photos.

    python build_raking_bands.py photos     # all-on + the four directional frames
    python build_raking_bands.py fused      # all-on + fused raking light from 4 angles

Writes data/coins/DateBands_raking_<mode>/ with a manifest.json and bands/.
Every non-2026data band is linked from data/coins/DateBands unchanged, so the
only thing that differs from the baseline cross-validation is the rig images.
Coin ids stay "2026data/<label>" - the same physical coins.

Train on it with:
    python train_date_reader_v1.py --bands-root ../../data/coins/DateBands_raking_<mode> \
        --folds 5 --seed 0 --epochs 30 --wear-aug --in-size 448x168
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from fuse_raking import load_capture, fuse
from harvest_date_bands import DATA_ROOT, band_box, make_record, parse_2026

RAW = DATA_ROOT / "RakingRaw"
BASE = DATA_ROOT / "DateBands"
FUSE_ANGLES = (45, 135, 225, 315)       # light direction, degrees (0 = from image right)
FUSE_K = 1.6
MARGIN = 300                             # fusion's 40 px blur needs context around the band


def coin_circle(gray):
    """(cx, cy, r): strongest edge ring near the frame centre, coarse-to-fine."""
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    cos, sin = np.cos(th), np.sin(th)

    def edges(scale):
        g = cv2.resize(gray, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA).astype(float)
        gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1)
        return np.hypot(gx, gy)

    def score(e, cx, cy, r):
        h, w = e.shape
        ys, xs = (cy + r * sin).astype(int), (cx + r * cos).astype(int)
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        return e[ys[ok], xs[ok]].mean() if ok.any() else 0

    e = edges(16)
    h, w = e.shape
    best = max((score(e, cx, cy, r), cx, cy, r)
               for cy in range(h // 2 - 30, h // 2 + 31, 2)
               for cx in range(w // 2 - 40, w // 2 + 41, 2)
               for r in range(int(0.25 * h), int(0.48 * h)))
    _, cx, cy, r = best
    e = edges(4)
    cx, cy, r = cx * 4, cy * 4, r * 4
    best = max((score(e, x, y, rr), x, y, rr)
               for y in range(cy - 8, cy + 9)
               for x in range(cx - 8, cx + 9)
               for rr in range(r - 8, r + 9))
    _, cx, cy, r = best
    return cx * 4.0, cy * 4.0, r * 4.0


def main():
    mode = sys.argv[1]
    assert mode in ("photos", "fused"), "mode is 'photos' or 'fused'"
    out = DATA_ROOT / f"DateBands_raking_{mode}"
    (out / "bands").mkdir(parents=True, exist_ok=True)

    base = json.loads((BASE / "manifest.json").read_text())
    keep = [r for r in base if r["source"] != "2026data"]
    for r in keep:
        dst = out / "bands" / r["band"]
        if not dst.exists():
            os.symlink(BASE / "bands" / r["band"], dst)

    new, circles = [], {}
    for d in sorted(p for p in RAW.iterdir() if (p / "capture.json").exists() and not p.name.startswith("test_")):
        year, mint, digits = parse_2026(d.name)
        all_bgr = cv2.imread(str(d / "all.jpg"))
        hh, ww = all_bgr.shape[:2]
        cx, cy, r = coin_circle(cv2.cvtColor(all_bgr, cv2.COLOR_BGR2GRAY))
        circles[d.name] = (cx, cy, r)
        x1, y1, x2, y2 = band_box(cx, cy, r, ww, hh)

        if mode == "photos":
            variants = [("all", all_bgr)] + [(n, cv2.imread(str(d / f"{n}.jpg"))) for n in "NESW"]
        else:
            f = load_capture(d)
            mx1, my1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
            mx2, my2 = min(ww, x2 + MARGIN), min(hh, y2 + MARGIN)
            sub = {n: a[my1:my2, mx1:mx2] for n, a in f.items()}
            variants = [("all", all_bgr)]
            for ang in FUSE_ANGLES:
                rgb = np.asarray(fuse(sub, FUSE_K, ang))
                full = np.zeros((hh, ww, 3), np.uint8)
                full[my1:my2, mx1:mx2] = rgb[..., ::-1]           # PIL RGB -> cv2 BGR
                variants.append((f"fused{ang}", full))

        for i, (tag, img) in enumerate(variants):
            rec = make_record("2026data", d.name, i, year, mint, digits, d / "all.jpg",
                              (cx, cy, r), (x1, y1, x2, y2), f"raking-{mode}-{tag}")
            cv2.imwrite(str(out / "bands" / rec["band"]), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            new.append(rec)
        print(f"{d.name:16s} circle ({cx:.0f}, {cy:.0f}, r {r:.0f})  {len(variants)} bands")

    (out / "manifest.json").write_text(json.dumps(keep + new, indent=1))
    print(f"\n{len(keep)} kept bands + {len(new)} rig bands ({len(circles)} coins) -> {out}")


if __name__ == "__main__":
    main()
