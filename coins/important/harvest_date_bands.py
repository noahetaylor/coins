"""
harvest_date_bands.py
=====================
Collects date-band crops from every available coin source into ONE folder
with a single manifest, so downstream digit detection/classification has a
unified dataset to train on.

READ-ONLY ON SOURCES. Every source image is opened for reading and never
written, moved, or deleted. Existing YOLO label files are COPIED. The
original datasets (2026Data, Supers, PCGS, Bounds, digitsannotations)
remain byte-for-byte intact.

WHY A DATE BAND
---------------
The date sits at a fixed geometric position relative to the coin circle on
the Washington quarter obverse. Given (cx, cy, r) the band is:

    x: cx - HALF_W*r  ..  cx + HALF_W*r
    y: cy + TOP*r     ..  cy + BOT*r

No model is involved. Each source only needs a way to find (cx, cy, r).

SOURCES
-------
  2026data   53 studio coins, circle from labels/crop_labels.json.
             70 lighting variants per coin -> sampled, not all taken.
             Year+mint known from folder name.
  supers     14 studio coins (1980-1986 x2), circle via Hough.
             Year known from folder name; mint unknown.
  pcgs       504 slab scans, obverse is the LEFT half, white background,
             circle via contour. Year known from folder name.
             (Bounds/OriginalsUncropped is a byte-identical duplicate of
             PCGS/OriginalsUncropped and is skipped by default.)
  crops      384 pre-extracted date bands from digitsannotations.
             Year UNKNOWN. 75 of them carry YOLO digit labels, which are
             deduplicated across the overlapping base*/out* folders and
             copied alongside.

KNOWN-YEAR SOURCES ARE THE POINT
--------------------------------
For 2026data / supers / pcgs the year is known, so the four digit CLASSES
are known without any labeling. Only box POSITIONS need to be produced.
The manifest records `digits` for these, and null for `crops`.

USAGE
-----
    python harvest_date_bands.py --dry-run       # counts only, writes nothing
    python harvest_date_bands.py                 # harvest everything
    python harvest_date_bands.py --sources pcgs supers
    python harvest_date_bands.py --per-coin 10   # more lighting variants

OUTPUT
------
    DateBands/
        bands/<source>__<coin_id>__<n>.jpg
        labels/<same stem>.txt      (only where a source label existed)
        manifest.json
        manifest.csv
        report.txt
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMPORTANT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = IMPORTANT_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "coins"
CROP_LABELS = IMPORTANT_DIR / "labels" / "crop_labels.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Date band geometry, in units of coin radius r, relative to coin centre.
#
# The date sits on an ARC, so its outer digits ride higher than its middle
# ones. The original window (top 0.68, half-width 0.46) was tight enough
# that on coins whose centre estimate drifted, the first or last digit was
# clipped off the TOP edge - not the side. That silently amputated a digit
# on several studio coins and the reader confidently mis-read them
# (1968 -> 1976, 1997 -> 1998) because the evidence genuinely was not in
# the image.
#
# Widening alone does not fix it; the top edge has to come up. Going much
# past top=0.55 starts pulling in "IN GOD WE TRUST" and the bust, which is
# distractor content, so this is about as generous as is useful.
BAND_TOP = 0.58
BAND_BOT = 1.02
BAND_HALF_W = 0.58

# Bicentennial quarters carry the dual date "1776-1976".
DUAL_DATE = "1776-1976"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def band_box(cx, cy, r, w, h, top=BAND_TOP, bot=BAND_BOT, half_w=BAND_HALF_W):
    """Date-band pixel box, clamped to image bounds. None if degenerate."""
    x1 = int(round(cx - half_w * r))
    x2 = int(round(cx + half_w * r))
    y1 = int(round(cy + top * r))
    y2 = int(round(cy + bot * r))
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 32 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def circle_from_white_bg(img, left_half=False):
    """Largest non-white blob -> min enclosing circle. For studio/slab scans."""
    if left_half:
        img = img[:, : img.shape[1] // 2]
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cv2.convexHull(cnt))
    if r < min(h, w) * 0.15:
        return None
    return float(cx), float(cy), float(r)


def circle_from_hough(img):
    """Hough circle. For the Supers rig where the coin sits on cloth."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1, minDist=max(h, w),
        param1=100, param2=30,
        minRadius=int(min(h, w) * 0.25), maxRadius=int(min(h, w) * 0.55),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r)


# ---------------------------------------------------------------------------
# Year / mint parsing
# ---------------------------------------------------------------------------

def parse_2026(folder):
    """'1965_1' -> (1965, None); '1984_P' -> (1984, 'P'); 'Bicentennial_1'."""
    if folder.lower().startswith("bicentennial"):
        return 1976, None, DUAL_DATE
    m = re.match(r"^(\d{4})(?:_([PD]))?", folder)
    if not m:
        return None, None, None
    year = int(m.group(1))
    return year, m.group(2), str(year)


def parse_supers(folder):
    """'1982b' -> (1982, None). Trailing 'b' marks the second physical coin."""
    m = re.match(r"^(\d{4})b?$", folder)
    if not m:
        return None, None, None
    year = int(m.group(1))
    return year, None, str(year)


def parse_pcgs(folder):
    """'1972' -> (1972, None); 'BicentennialsD' -> (1976, 'D', dual date)."""
    if folder.lower().startswith("bicentennial"):
        mint = "D" if folder.rstrip().endswith("D") else ("P" if folder.rstrip().endswith("P") else None)
        return 1976, mint, DUAL_DATE
    m = re.match(r"^(\d{4})$", folder)
    if not m:
        return None, None, None
    year = int(m.group(1))
    return year, None, str(year)


def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sample_evenly(items, count):
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


# ---------------------------------------------------------------------------
# Existing YOLO labels (deduplicated)
# ---------------------------------------------------------------------------

def collect_existing_labels(ann_root):
    """
    digitsannotations/ has several overlapping folders (date_cropmatchedbase*,
    date_cropmatchedout*, dumptrain) holding copies of the same annotations.
    Key by image stem, keep the richest version, so nothing is double counted.
    """
    best = {}
    if not ann_root.is_dir():
        return best
    for sub in sorted(ann_root.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub.name.startswith("date_cropmatched") or sub.name == "dumptrain"):
            continue
        for txt in sub.glob("*.txt"):
            lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            stem = txt.stem
            if stem not in best or len(lines) > len(best[stem]):
                best[stem] = lines
    return best


# ---------------------------------------------------------------------------
# Harvest record
# ---------------------------------------------------------------------------

def make_record(source, coin_id, idx, year, mint, digits, src, circle, box, method):
    stem = f"{source}__{coin_id}__{idx}"
    return {
        "band": f"{stem}.jpg",
        "stem": stem,
        "source": source,
        "coin_id": f"{source}/{coin_id}",   # physical-coin identity -> split key
        "year": year,
        "mint": mint,
        "digits": digits,                   # known digit string, or None
        "src_image": str(src.relative_to(PROJECT_ROOT)),
        "circle": None if circle is None else
                  {"cx": round(circle[0], 2), "cy": round(circle[1], 2), "r": round(circle[2], 2)},
        "band_box": None if box is None else list(box),
        "method": method,
        "has_digit_labels": False,
    }


class Harvester:
    def __init__(self, out_dir, dry_run):
        self.out = out_dir
        self.dry = dry_run
        self.bands_dir = out_dir / "bands"
        self.labels_dir = out_dir / "labels"
        self.records = []
        self.failures = defaultdict(list)
        if not dry_run:
            self.bands_dir.mkdir(parents=True, exist_ok=True)
            self.labels_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, rec, band_img=None, src_copy=None, label_lines=None):
        if not self.dry:
            dst = self.bands_dir / rec["band"]
            if band_img is not None:
                cv2.imwrite(str(dst), band_img)
            elif src_copy is not None:
                shutil.copy2(src_copy, dst)     # COPY - source untouched
            if label_lines:
                (self.labels_dir / f"{rec['stem']}.txt").write_text(
                    "\n".join(label_lines) + "\n")
        rec["has_digit_labels"] = bool(label_lines)
        self.records.append(rec)

    def fail(self, source, path, why):
        self.failures[source].append((str(path), why))


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

def harvest_2026(h, per_coin, geom):
    root = DATA_ROOT / "2026Data"
    if not root.is_dir():
        return
    if not CROP_LABELS.is_file():
        h.fail("2026data", CROP_LABELS, "crop_labels.json missing")
        return
    labels = json.loads(CROP_LABELS.read_text())
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name
        if name not in labels:
            h.fail("2026data", folder, "no crop label for folder")
            continue
        year, mint, digits = parse_2026(name)
        L = labels[name]
        cx, cy, r = float(L["cx"]), float(L["cy"]), float(L["r"])
        for i, src in enumerate(sample_evenly(list_images(folder), per_coin)):
            img = cv2.imread(str(src))
            if img is None:
                h.fail("2026data", src, "unreadable")
                continue
            hh, ww = img.shape[:2]
            box = band_box(cx, cy, r, ww, hh, *geom)
            if box is None:
                h.fail("2026data", src, "degenerate band box")
                continue
            x1, y1, x2, y2 = box
            rec = make_record("2026data", name, i, year, mint, digits,
                              src, (cx, cy, r), box, "manual-crop-label")
            h.emit(rec, band_img=img[y1:y2, x1:x2])


def harvest_supers(h, per_coin, geom):
    root = DATA_ROOT / "Supers"
    if not root.is_dir():
        return
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        year, mint, digits = parse_supers(folder.name)
        if year is None:
            h.fail("supers", folder, "unparseable folder name")
            continue
        # Prefer the evenly lit "all" frames; they are the most legible.
        imgs = [p for p in list_images(folder) if "_all" in p.name] or list_images(folder)
        circle = None
        for i, src in enumerate(sample_evenly(imgs, per_coin)):
            img = cv2.imread(str(src))
            if img is None:
                h.fail("supers", src, "unreadable")
                continue
            # Rig is fixed within a folder: detect once, reuse.
            if circle is None:
                circle = circle_from_hough(img)
                if circle is None:
                    h.fail("supers", src, "circle detection failed")
                    break
            cx, cy, r = circle
            hh, ww = img.shape[:2]
            box = band_box(cx, cy, r, ww, hh, *geom)
            if box is None:
                h.fail("supers", src, "degenerate band box")
                continue
            x1, y1, x2, y2 = box
            rec = make_record("supers", folder.name, i, year, mint, digits,
                              src, circle, box, "hough")
            h.emit(rec, band_img=img[y1:y2, x1:x2])


def harvest_pcgs(h, geom):
    root = DATA_ROOT / "PCGS" / "OriginalsUncropped"
    if not root.is_dir():
        return
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        year, mint, digits = parse_pcgs(folder.name)
        if year is None:
            h.fail("pcgs", folder, "unparseable folder name")
            continue
        for src in list_images(folder):
            img = cv2.imread(str(src))
            if img is None:
                h.fail("pcgs", src, "unreadable")
                continue
            circle = circle_from_white_bg(img, left_half=True)
            if circle is None:
                h.fail("pcgs", src, "circle detection failed")
                continue
            cx, cy, r = circle
            hh, ww = img.shape[:2]
            box = band_box(cx, cy, r, ww, hh, *geom)
            if box is None:
                h.fail("pcgs", src, "degenerate band box")
                continue
            x1, y1, x2, y2 = box
            # coin_id is the slab cert number - one physical coin each.
            rec = make_record("pcgs", f"{folder.name}_{src.stem}", 0,
                              year, mint, digits, src, circle, box,
                              "contour-white-bg/left-half")
            h.emit(rec, band_img=img[y1:y2, x1:x2])


def harvest_crops(h):
    """Pre-extracted bands from digitsannotations. Year unknown; copy as-is."""
    ann = DATA_ROOT / "digitsannotations"
    if not ann.is_dir():
        return
    existing = collect_existing_labels(ann)
    seen = set()
    for sub in sorted(ann.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("date_crops"):
            continue
        for src in list_images(sub):
            if src.stem in seen:      # same band duplicated across crops folders
                continue
            seen.add(src.stem)
            rec = make_record("crops", src.stem, 0, None, None, None,
                              src, None, None, "pre-extracted")
            h.emit(rec, src_copy=src, label_lines=existing.get(src.stem))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(records, failures):
    lines = []
    by_source = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    lines.append("=" * 66)
    lines.append("DATE BAND HARVEST")
    lines.append("=" * 66)
    lines.append(f"{'source':<12}{'bands':>8}{'coins':>8}{'known yr':>10}{'labeled':>9}")
    lines.append("-" * 66)
    for s in sorted(by_source):
        rs = by_source[s]
        lines.append(f"{s:<12}{len(rs):>8}{len({r['coin_id'] for r in rs}):>8}"
                     f"{sum(1 for r in rs if r['digits']):>10}"
                     f"{sum(1 for r in rs if r['has_digit_labels']):>9}")
    lines.append("-" * 66)
    lines.append(f"{'TOTAL':<12}{len(records):>8}"
                 f"{len({r['coin_id'] for r in records}):>8}"
                 f"{sum(1 for r in records if r['digits']):>10}"
                 f"{sum(1 for r in records if r['has_digit_labels']):>9}")

    # Digit-class yield. Counts each PHYSICAL COIN once, not each lighting
    # variant, so the numbers reflect real independent examples.
    per_coin_digits = {}
    for r in records:
        if r["digits"] and r["digits"] != DUAL_DATE:
            per_coin_digits[r["coin_id"]] = r["digits"]
    cls = Counter()
    for d in per_coin_digits.values():
        for ch in d:
            cls[ch] += 1
    lines.append("")
    lines.append("Digit instances derivable for free from known years")
    lines.append("(counting each physical coin once):")
    for k in map(str, range(10)):
        lines.append(f"   '{k}': {cls.get(k, 0)}")
    lines.append(f"   TOTAL: {sum(cls.values())} across {len(per_coin_digits)} coins")

    dual = sum(1 for r in records if r["digits"] == DUAL_DATE)
    if dual:
        lines.append(f"   ({dual} bicentennial bands excluded - dual date {DUAL_DATE})")

    if failures:
        lines.append("")
        lines.append("Failures:")
        for s, fs in sorted(failures.items()):
            lines.append(f"   {s}: {len(fs)}")
            for path, why in fs[:5]:
                lines.append(f"      {why}: {path}")
            if len(fs) > 5:
                lines.append(f"      ... and {len(fs) - 5} more")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "DateBands")
    ap.add_argument("--sources", nargs="+",
                    choices=["2026data", "supers", "pcgs", "crops"],
                    default=["2026data", "supers", "pcgs", "crops"])
    ap.add_argument("--per-coin", type=int, default=6,
                    help="lighting variants to sample per studio coin (2026data/supers)")
    ap.add_argument("--band-top", type=float, default=BAND_TOP)
    ap.add_argument("--band-bot", type=float, default=BAND_BOT)
    ap.add_argument("--band-half-width", type=float, default=BAND_HALF_W)
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts without writing anything")
    args = ap.parse_args()

    # Refuse to write into a source tree.
    protected = {"2026Data", "Supers", "PCGS", "Bounds", "digitsannotations",
                 "2026RotationTraining"}
    if args.out.resolve().name in protected or \
       any(p.name in protected for p in args.out.resolve().parents
           if DATA_ROOT in p.parents or p == DATA_ROOT):
        sys.exit(f"Refusing to write into a source dataset: {args.out}")

    geom = (args.band_top, args.band_bot, args.band_half_width)
    h = Harvester(args.out, args.dry_run)

    if "2026data" in args.sources:
        harvest_2026(h, args.per_coin, geom)
    if "supers" in args.sources:
        harvest_supers(h, args.per_coin, geom)
    if "pcgs" in args.sources:
        harvest_pcgs(h, geom)
    if "crops" in args.sources:
        harvest_crops(h)

    report = build_report(h.records, h.failures)
    print(report)

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(h.records, indent=2))
    fields = ["band", "source", "coin_id", "year", "mint", "digits",
              "src_image", "method", "has_digit_labels"]
    with open(args.out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(h.records)
    (args.out / "report.txt").write_text(report + "\n")
    print(f"\nwrote {len(h.records)} bands -> {args.out}")
    print("sources untouched (copy-only)")


if __name__ == "__main__":
    main()
