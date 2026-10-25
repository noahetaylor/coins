"""
read_coin.py
============
End-to-end inference: raw coin photo(s) -> predicted year, with confidence.

This is the piece the training/eval scripts never needed - everything in
train_date_reader_v1.py operates on bands already cropped by
harvest_date_bands.py from known sources. A live capture station has no
manifest and no hand-labeled circle geometry, so this module owns the
raw-image -> band -> digits path.

Reuses the model, checkpoint format, and EVAL_TF preprocessing from
train_date_reader_v1.py directly (not reimplemented) so inference can not
silently drift from what cross-validation actually measured.

USAGE
-----
    # one frame
    python read_coin.py path/to/frame.jpg

    # multiple lighting frames of the SAME physical coin -> majority vote
    python read_coin.py coin_dir/*.jpg

    # a directory of independent coins, one prediction per subfolder
    python read_coin.py --batch coins_root/
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest_date_bands import (IMAGE_SUFFIXES, band_box, circle_from_hough,
                                 circle_from_white_bg)
from train_date_reader_v1 import DEVICE, EVAL_TF, IN_H, IN_W, DateReader, load_checkpoint

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "date_reader_v1.pth"

# Circle-detection strategies tried in order; first one that returns a
# plausible circle wins.
CIRCLE_FINDERS = [circle_from_hough, circle_from_white_bg]

# circle_from_hough/circle_from_white_bg were tuned against the Supers rig
# (720x1280). Studio/slab captures run 3500-5800px on a side - at that size
# Hough's accumulator search becomes pathologically slow (minutes, not
# seconds) and contour detection gets noisier. Detect on a downscaled copy
# and scale the result back up; it's what the size these finders were
# actually tuned for anyway.
DETECT_MAX_SIDE = 1280


def find_circle(img):
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
    for finder in CIRCLE_FINDERS:
        circle = finder(small)
        if circle is not None:
            cx, cy, r = circle
            return cx / scale, cy / scale, r / scale
    return None


def crop_band(img, circle):
    h, w = img.shape[:2]
    box = band_box(*circle, w, h)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def load_model(checkpoint=DEFAULT_CHECKPOINT):
    model = DateReader(pretrained=False).to(DEVICE)
    ck_size = load_checkpoint(model, checkpoint)
    if ck_size and tuple(ck_size) != (IN_W, IN_H):
        print(f"warning: checkpoint trained at {ck_size[0]}x{ck_size[1]}, "
              f"module default is {IN_W}x{IN_H}", file=sys.stderr)
    model.eval()
    return model


def read_band(model, band_bgr):
    """One band crop (BGR ndarray) -> (digit_string, per_digit_confidence)."""
    rgb = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = EVAL_TF(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]                      # (N_POS, N_CLS)
        probs = F.softmax(logits, dim=-1)
        conf, digits = probs.max(dim=-1)
    digit_str = "".join(str(int(d)) for d in digits.tolist())
    return digit_str, conf.tolist()


def read_frame(model, img_bgr):
    """Raw photo -> (digit_string, per_digit_confidence) or None if no
    coin circle was found."""
    circle = find_circle(img_bgr)
    if circle is None:
        return None
    band = crop_band(img_bgr, circle)
    if band is None:
        return None
    return read_band(model, band)


def read_coin(model, image_paths):
    """Majority-vote a physical coin's year across N lighting frames.

    Returns a dict: year, per-frame votes, agreement fraction, and the
    mean confidence of the winning frames - the two signals a sorter
    should gate on before trusting a read.
    """
    votes, frame_results = [], []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            frame_results.append({"path": str(p), "error": "unreadable"})
            continue
        result = read_frame(model, img)
        if result is None:
            frame_results.append({"path": str(p), "error": "no circle found"})
            continue
        digits, conf = result
        votes.append(digits)
        frame_results.append({"path": str(p), "digits": digits,
                              "confidence": conf, "min_confidence": min(conf)})

    if not votes:
        return {"year": None, "agreement": 0.0, "mean_confidence": 0.0,
                "n_frames": len(image_paths), "n_read": 0, "frames": frame_results}

    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    winner = max(counts, key=counts.get)
    agreement = counts[winner] / len(votes)
    winning_confidences = [min(f["confidence"]) for f in frame_results
                           if f.get("digits") == winner]
    return {
        "year": winner,
        "agreement": agreement,
        "mean_confidence": float(np.mean(winning_confidences)),
        "n_frames": len(image_paths),
        "n_read": len(votes),
        "frames": frame_results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", type=Path,
                    help="one or more frames of the SAME physical coin")
    ap.add_argument("--batch", type=Path,
                    help="directory of subfolders, one coin per subfolder, "
                         "all images in a subfolder are frames of that coin")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if not args.images and not args.batch:
        ap.error("pass image path(s) or --batch a_directory")

    model = load_model(args.checkpoint)

    def emit(label, result):
        if args.json:
            print(json.dumps({"coin": label, **result}))
        else:
            year = result["year"] or "UNREADABLE"
            print(f"{label:30s} year={year}  agreement={result['agreement']:.0%}  "
                  f"confidence={result['mean_confidence']:.2f}  "
                  f"({result['n_read']}/{result['n_frames']} frames read)")

    if args.batch:
        for sub in sorted(p for p in args.batch.iterdir() if p.is_dir()):
            imgs = sorted(p for p in sub.iterdir()
                          if p.suffix.lower() in IMAGE_SUFFIXES)
            if not imgs:
                continue
            emit(sub.name, read_coin(model, imgs))
    else:
        emit(args.images[0].parent.name or "coin", read_coin(model, args.images))


if __name__ == "__main__":
    main()
