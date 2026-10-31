#!/usr/bin/env python3
"""
demo_station.py - one-button scan demo for the rig.

Press Scan (or Space) in the page: the Pi takes a raking-light capture, the
frames come back to this PC, and the date reader votes a year across them.

    cd ~/projects/image-identifier/coins/rig
    ../../.venv/bin/python demo_station.py            # then open http://localhost:5050

Runs in WSL but drives the Pi through Windows' OpenSSH, which holds the key.
Set PI_HOST if the Pi's address changes:  PI_HOST=user@<pi-address> ...

Uses models/date_reader_raking.pth (trained on the raking-rig photos, 80% of
coins; the rest are listed in date_reader_raking.split.json as held out).
"""
import base64
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, jsonify, send_file

HERE = Path(__file__).resolve().parent
IMPORTANT = HERE.parent / "important"
sys.path.insert(0, str(IMPORTANT))
from build_raking_bands import coin_circle               # noqa: E402
from harvest_date_bands import band_box                  # noqa: E402
from read_coin import load_model, read_band              # noqa: E402

PI = os.environ.get("PI_HOST", "user@<pi-address>")
WIN_SSH = "/mnt/c/Windows/System32/OpenSSH/"
CHECKPOINT = IMPORTANT / "models" / "date_reader_raking.pth"
LOCAL = Path("/mnt/c/Users/NoahE/AppData/Local/Temp/coin_demo")   # Windows scp writes here
FRAMES = ("all", "N", "E", "S", "W")

app = Flask(__name__)
model = load_model(CHECKPOINT)
state = {"stage": "idle", "lit": None, "done": [], "result": None, "error": None, "busy": False}
lock = threading.Lock()


def set_state(**kw):
    with lock:
        state.update(kw)


def jpeg_b64(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def scan():
    label = time.strftime("test_demo_%H%M%S")
    try:
        set_state(stage="capturing", lit="dark", done=[], result=None, error=None)
        cmd = [WIN_SSH + "ssh.exe", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", PI,
               f"~/coin-sorter/scripts/camera/capture_raking.py {label}"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        order = ["dark", "N", "E", "S", "W", "all"]
        for line in p.stdout:
            m = re.match(r"\s+(dark|N|E|S|W|all)\s+exposure", line)
            if m:
                name = m.group(1)
                nxt = order[order.index(name) + 1] if name != "all" else None
                with lock:
                    state["done"].append(name)
                    state["lit"] = nxt
        if p.wait() not in (0, 1):                        # 1 = captured with a warning
            raise RuntimeError("capture failed - is the Pi on, and is PI_HOST right?")

        set_state(stage="transferring", lit=None)
        shutil.rmtree(LOCAL, ignore_errors=True)
        LOCAL.mkdir(parents=True)
        win_dest = subprocess.check_output(["wslpath", "-w", str(LOCAL)], text=True).strip()
        subprocess.run([WIN_SSH + "scp.exe", "-q", "-r", "-o", "BatchMode=yes",
                        f"{PI}:coin-sorter/data/raking/{label}", win_dest], check=True)
        subprocess.run([WIN_SSH + "ssh.exe", "-o", "BatchMode=yes", PI,
                        f"rm -rf ~/coin-sorter/data/raking/{label}"], check=False)

        set_state(stage="reading")
        d = LOCAL / label
        imgs = {n: cv2.imread(str(d / f"{n}.jpg")) for n in FRAMES}
        hh, ww = imgs["all"].shape[:2]
        cx, cy, r = coin_circle(cv2.cvtColor(imgs["all"], cv2.COLOR_BGR2GRAY))
        x1, y1, x2, y2 = band_box(cx, cy, r, ww, hh)
        votes = []
        for n in FRAMES:
            digits, conf = read_band(model, imgs[n][y1:y2, x1:x2])
            votes.append({"frame": n, "digits": digits, "conf": round(min(conf), 3)})
        year, n_agree = Counter(v["digits"] for v in votes).most_common(1)[0]
        win_conf = [v["conf"] for v in votes if v["digits"] == year]

        m = int(1.08 * r)
        coin = imgs["all"][max(0, int(cy) - m):int(cy) + m, max(0, int(cx) - m):int(cx) + m]
        set_state(stage="done", result={
            "year": year, "agree": n_agree, "of": len(votes),
            "confidence": round(sum(win_conf) / len(win_conf), 3), "votes": votes,
            "coin": jpeg_b64(coin, 820), "band": jpeg_b64(imgs["all"][y1:y2, x1:x2], 900),
        })
    except Exception as e:                                 # shown in the page, not a crash
        set_state(stage="error", error=str(e))
    finally:
        set_state(busy=False)


@app.post("/scan")
def start_scan():
    with lock:
        if state["busy"]:
            return jsonify(ok=False)
        state["busy"] = True
    threading.Thread(target=scan, daemon=True).start()
    return jsonify(ok=True)


@app.post("/reset")
def reset():
    with lock:
        if not state["busy"]:
            state.update(stage="idle", lit=None, done=[], result=None, error=None)
    return jsonify(ok=True)


@app.get("/status")
def status():
    with lock:
        return jsonify(state)


@app.get("/")
def page():
    return send_file(HERE / "demo_station.html")


if __name__ == "__main__":
    print(f"Pi: {PI}   model: {CHECKPOINT.name}\nopen http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, threaded=True)
