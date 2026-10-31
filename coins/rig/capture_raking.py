#!/usr/bin/python3
"""
capture_raking.py
=================
Directional (photometric-stereo style) capture of one coin on the rig.

Per coin it saves six frames plus the settings used:

    dark.jpg   LEDs off           -> ambient light, subtracted during fusion
    N/E/S/W    one 5-LED arc each -> light from one side of the coin
    all.jpg    whole ring on      -> flat-lit reference

Why these choices (measured on the rig, 2026-10-25):
  - Ring is 16 LEDs, not 24. LED 0 lights from the image top and the index
    runs clockwise ~22.5 deg per LED, so arcs are centred on 0/4/8/12.
  - Arcs must be pure (other LEDs OFF). The old capture_rotation.py kept
    every LED at 45% + 30% white fill, which flattens the shading; pure arcs
    give a N-vs-S relief signal ~8x the repeat-capture noise floor.
  - Exposure and white balance are locked. In auto mode the camera would
    re-level each direction and erase the differences we are capturing.
  - Under AUTO exposure the first frames of a fresh stream are near-black;
    with exposure locked, frame 0 is already correct. Frame 2 is kept as margin.
  - LED colour 255,138,50 reads neutral (within 1 level per channel) on the
    white stage at a locked 5000 K white balance (plain 255,255,255 comes
    out strongly blue).

USAGE
-----
    python3 capture_raking.py 1996_P          # place coin date-down, upright
    python3 capture_raking.py 1996_P --size 2320x1744   # quick test

A label that already exists gets a _1, _2 ... suffix, matching 2026Data.
LEDs are driven through `sudo rig-led` (passwordless for that one command).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------

CAMERA_DEVICE = "/dev/video0"
OUT_ROOT = "/home/noah/coin-sorter/data/raking"
SKIP_FRAMES = 2

NUM_LEDS = 16
LED_BRIGHTNESS = 0.5
LED_COLOR = (255, 138, 50)

ARCS = {
    "N": [14, 15, 0, 1, 2],
    "E": [2, 3, 4, 5, 6],
    "S": [6, 7, 8, 9, 10],
    "W": [10, 11, 12, 13, 14],
}

# exposure_time_absolute units (camera range 1-5000). Retuned 2026-10-28 after the
# ring measured ~8x dimmer than on 10-25 (same camera response); one arc is 5/16 of the
# ring, so it needs ~3x the all-on exposure.
EXPOSURE_ARC = 210
EXPOSURE_ALL = 84
EXPOSURE_DARK = EXPOSURE_ARC      # dark frame must match the arc frames it is subtracted from

CAMERA_LOCK = {
    "auto_exposure": 1,               # 1 = manual
    "white_balance_automatic": 0,
    "white_balance_temperature": 5000,
    "backlight_compensation": 0,
    "gain": 0,
}

# -----------------------------
# HARDWARE
# -----------------------------

def leds(*args):
    cmd = ["sudo", "-n", "/usr/local/bin/rig-led", "--n", str(NUM_LEDS),
           "--brightness", str(LED_BRIGHTNESS), *map(str, args)]
    subprocess.run(cmd, check=True)


def retry(fn, *args, tries=4, wait=2.0):
    """The camera sometimes drops off USB for a second and re-enumerates; wait it out."""
    for k in range(tries):
        try:
            return fn(*args)
        except (subprocess.CalledProcessError, RuntimeError):
            if k == tries - 1:
                raise
            print(f"  camera not responding, retrying in {wait:.0f} s ...")
            time.sleep(wait)


def set_camera(**ctrls):
    spec = ",".join(f"{k}={v}" for k, v in ctrls.items())
    retry(lambda: subprocess.run(["v4l2-ctl", "-d", CAMERA_DEVICE, "-c", spec], check=True))


def grab(path, size):
    subprocess.run([
        "ffmpeg", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "mjpeg", "-video_size", size,
        "-i", CAMERA_DEVICE,
        "-vf", f"select=eq(n\\,{SKIP_FRAMES})", "-vsync", "0", "-vframes", "1",
        path, "-y",
    ], check=True, stderr=subprocess.DEVNULL)
    if not os.path.exists(path):
        raise RuntimeError(f"camera returned no frame for {path}")


def frame_check(path):
    """Mean brightness and % clipped pixels, on a downsampled copy."""
    g = np.asarray(Image.open(path).convert("L").reduce(8), dtype=float)
    return round(float(g.mean()), 1), round(float((g >= 250).mean() * 100), 2)

# -----------------------------
# MAIN
# -----------------------------

def unique_dir(label):
    path, n = os.path.join(OUT_ROOT, label), 0
    while os.path.exists(path):
        n += 1
        path = os.path.join(OUT_ROOT, f"{label}_{n}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("label", help="coin label, e.g. 1996_P")
    p.add_argument("--size", default="4656x3496")
    args = p.parse_args()

    out = unique_dir(args.label)
    os.makedirs(out)
    print(f"Saving to {out}")

    set_camera(**CAMERA_LOCK)
    plan = [("dark", "off", (), EXPOSURE_DARK)]
    plan += [(name, "leds", (",".join(map(str, idx)), *LED_COLOR), EXPOSURE_ARC)
             for name, idx in ARCS.items()]
    plan += [("all", "all", LED_COLOR, EXPOSURE_ALL)]

    checks = {}
    try:
        for name, mode, led_args, exposure in plan:
            leds(mode, *led_args)
            set_camera(exposure_time_absolute=exposure)
            path = os.path.join(out, f"{name}.jpg")
            retry(grab, path, args.size)
            checks[name] = frame_check(path)
            mean, clipped = checks[name]
            print(f"  {name:4s} exposure {exposure:3d}  mean {mean:5.1f}  clipped {clipped:5.2f}%")
    finally:
        leds("off")

    warnings = [f"{n}: {c[1]}% clipped" for n, c in checks.items() if c[1] > 1.0]
    warnings += [f"{n}: too dark (mean {c[0]})" for n, c in checks.items() if n != "dark" and c[0] < 40]
    if checks["dark"][0] > 20:
        warnings.append(f"dark frame mean {checks['dark'][0]} - a lot of room light is leaking in")

    with open(os.path.join(out, "capture.json"), "w") as f:
        json.dump({
            "label": args.label,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "size": args.size, "skip_frames": SKIP_FRAMES,
            "num_leds": NUM_LEDS, "led_brightness": LED_BRIGHTNESS, "led_color": LED_COLOR,
            "arcs": ARCS,
            "exposure": {"arc": EXPOSURE_ARC, "all": EXPOSURE_ALL, "dark": EXPOSURE_DARK},
            "camera": CAMERA_LOCK,
            "frame_checks": checks, "warnings": warnings,
        }, f, indent=2)

    for w in warnings:
        print("  WARNING:", w)
    print("Done." if not warnings else "Done, with warnings - check before moving on.")
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
