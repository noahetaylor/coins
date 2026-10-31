"""
fuse_raking.py
==============
Fuse one rig capture (dark / N / E / S / W / all from coins/rig/capture_raking.py)
into a single relief-enhanced photo, so rig coins can be read by the same
model that trains on single scraped photos.

Method - "virtual raking light":
  1. Subtract the dark frame (ambient room light) from every frame, scaled
     to each frame's exposure.
  2. Per pixel, gx = (E-W)/(E+W) and gy = (N-S)/(N+S): how much that spot of
     relief faces east vs west and north vs south. Normalising by the sum
     cancels the coin's reflectance, leaving surface orientation only.
  3. Re-shade the flat-lit `all` frame as if lit from a low virtual light at
     --light-deg (default 135 = upper left):  out = all * (1 + k * s),
     where s is the slope component facing that light.

The output keeps the colour and overall look of a normal coin photo, with the
struck relief (digits) exaggerated by --k.

USAGE
-----
    python fuse_raking.py data/coins/RakingRaw/1996_P            # writes fused.jpg next to the frames
    python fuse_raking.py data/coins/RakingRaw --batch             # every coin folder
    python fuse_raking.py <coin_dir> --k 0 0.8 1.6 --sheet sheet.jpg   # compare strengths
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

DIRS = ("N", "E", "S", "W")


def box_blur(a, k):
    """Separable box blur, applied 3x (~gaussian). Works on float arrays."""
    if k < 1:
        return a
    for _ in range(3):
        for ax in (0, 1):
            pad = [(k, k) if i == ax else (0, 0) for i in range(a.ndim)]
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax)
            n = c.shape[ax]
            a = (np.take(c, range(2 * k, n), axis=ax) - np.take(c, range(0, n - 2 * k), axis=ax)) / (2 * k)
    return a


def load_capture(coin_dir):
    coin_dir = Path(coin_dir)
    meta = json.loads((coin_dir / "capture.json").read_text())
    exp = meta["exposure"]
    f = {n: np.asarray(Image.open(coin_dir / f"{n}.jpg"), dtype=np.float32)
         for n in ("dark", "all", *DIRS)}
    dark = f.pop("dark")
    # dark frame was shot at the arc exposure; rescale for the all-on frame
    for n in DIRS:
        f[n] = np.clip(f[n] - dark, 0, None)
    f["all"] = np.clip(f["all"] - dark * (exp["all"] / exp["dark"]), 0, None)
    return f


def slopes(f, smooth=1):
    lum = {n: box_blur(f[n].mean(axis=2), smooth) for n in DIRS}
    eps = 4.0                                    # keeps near-black pixels from blowing up the ratio
    gx = (lum["E"] - lum["W"]) / (lum["E"] + lum["W"] + eps)
    gy = (lum["N"] - lum["S"]) / (lum["N"] + lum["S"] + eps)
    return gx, gy


def fuse(f, k=1.6, light_deg=135.0, smooth=1):
    gx, gy = slopes(f, smooth)
    t = math.radians(light_deg)                  # 0 = from image right, 90 = from image top
    s = gx * math.cos(t) + gy * math.sin(t)
    s -= box_blur(s, 40)                         # keep relief-scale shading, drop the frame-wide tilt
    out = f["all"] * (1.0 + k * s)[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("path")
    p.add_argument("--batch", action="store_true", help="path holds one folder per coin")
    p.add_argument("--k", type=float, nargs="+", default=[1.6])
    p.add_argument("--light-deg", type=float, default=135.0)
    p.add_argument("--sheet", help="save a side-by-side of the --k values (single coin only)")
    args = p.parse_args()

    dirs = sorted(d for d in Path(args.path).iterdir() if (d / "capture.json").exists()) \
        if args.batch else [Path(args.path)]
    for d in dirs:
        f = load_capture(d)
        outs = [fuse(f, k, args.light_deg) for k in args.k]
        outs[0].save(d / "fused.jpg", quality=95)
        if args.sheet:
            w, h = outs[0].size
            sheet = Image.new("RGB", (w * len(outs), h))
            for i, im in enumerate(outs):
                sheet.paste(im, (i * w, 0))
            sheet.reduce(4).save(args.sheet, quality=90)
        print(f"{d.name}: fused.jpg (k={args.k[0]})")


if __name__ == "__main__":
    main()
