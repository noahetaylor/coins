#!/usr/bin/python3
"""
focus_assist.py
===============
Live focus view for the rig, served as a web page from the Pi.

    python3 focus_assist.py            # then open http://<pi-address>:8080
                                       # (or http://localhost:8080 on the Pi itself)

The page shows the whole frame, a 1:1 crop of the centre (full sensor
resolution - this is where focus is judged), and a sharpness score with its
recent history. Score = variance of the Laplacian on the centre crop:
higher is sharper. It wobbles ~10% from noise alone, so follow the trend.

Lens rings: N<->inf is FOCUS (the one to turn). Leave W<->T (zoom) and
O<->C (iris) alone - they change coin size and brightness, which the
dataset and the tuned exposure depend on.

Ctrl+C to stop (LEDs switch off).
"""
import io
import json
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw

CAMERA_DEVICE = "/dev/video0"
W, H = 4656, 3496
CROP_W, CROP_H = 1800, 1400          # scored region, centre of frame
VIEW_W, VIEW_H = 900, 700            # 1:1 crop shown on the page
LED_COLOR = (255, 138, 50)
EXPOSURE = 8
PORT = 8080

state = {"full": b"", "crop": b"", "score": 0.0, "best": 0.0, "history": []}
lock = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Rig focus</title>
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui,sans-serif}
.wrap{display:flex;flex-wrap:wrap;gap:12px;padding:12px}
img{max-width:100%;border:1px solid #444;background:#000}
.panel{flex:1 1 420px}
#score{font-size:44px;font-weight:700}
#best{color:#9c9}
canvas{width:100%;height:120px;background:#000;border:1px solid #444}
.hint{color:#aaa;font-size:14px;line-height:1.4}
</style></head><body><div class="wrap">
<div class="panel"><div>Whole frame (box = scored area)</div><img id="full"></div>
<div class="panel"><div>Centre, 1:1 sensor pixels</div><img id="crop">
<div id="score">-</div><div id="best"></div><canvas id="hist" width="600" height="120"></canvas>
<p class="hint">Turn <b>N&harr;&infin;</b> (focus) slowly. Go past the peak until the score falls,
then come back. Leave <b>W&harr;T</b> (zoom) and <b>O&harr;C</b> (iris) alone.</p></div>
</div><script>
const full=document.getElementById('full'),crop=document.getElementById('crop');
function tick(){const t=Date.now();full.src='/full.jpg?'+t;crop.src='/crop.jpg?'+t;
fetch('/score').then(r=>r.json()).then(s=>{
 document.getElementById('score').textContent=s.score.toFixed(0);
 document.getElementById('best').textContent='best '+s.best.toFixed(0)+(s.score>=s.best?'  ← BEST':'');
 const c=document.getElementById('hist'),g=c.getContext('2d'),h=s.history;g.clearRect(0,0,c.width,c.height);
 if(h.length<2)return;const mx=Math.max(...h)*1.05,mn=Math.min(...h)*0.95;
 g.strokeStyle='#6cf';g.lineWidth=2;g.beginPath();
 h.forEach((v,i)=>{const x=i*c.width/(h.length-1),y=c.height-(v-mn)/(mx-mn)*c.height;i?g.lineTo(x,y):g.moveTo(x,y);});
 g.stroke();}).catch(()=>{});}
setInterval(tick,400);tick();
</script></body></html>"""


def sharpness(g):
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def jpeg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def camera_loop(ff):
    frame_bytes = W * H
    x0, y0 = (W - CROP_W) // 2, (H - CROP_H) // 2
    vx, vy = (W - VIEW_W) // 2, (H - VIEW_H) // 2
    while True:
        buf = ff.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            return
        g = np.frombuffer(buf, np.uint8).reshape(H, W)
        s = sharpness(g[y0:y0 + CROP_H, x0:x0 + CROP_W].astype(np.float32))

        full = Image.fromarray(np.ascontiguousarray(g[::4, ::4]))
        ImageDraw.Draw(full).rectangle([x0 // 4, y0 // 4, (x0 + CROP_W) // 4, (y0 + CROP_H) // 4], outline=255, width=2)
        crop = Image.fromarray(np.ascontiguousarray(g[vy:vy + VIEW_H, vx:vx + VIEW_W]))
        full_j, crop_j = jpeg(full), jpeg(crop)
        with lock:
            state["full"], state["crop"], state["score"] = full_j, crop_j, s
            state["best"] = max(state["best"], s)
            state["history"] = (state["history"] + [s])[-90:]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        with lock:
            if path == "/":
                body, ctype = PAGE.encode(), "text/html"
            elif path == "/full.jpg":
                body, ctype = state["full"], "image/jpeg"
            elif path == "/crop.jpg":
                body, ctype = state["crop"], "image/jpeg"
            elif path == "/score":
                body = json.dumps({k: state[k] for k in ("score", "best", "history")}).encode()
                ctype = "application/json"
            else:
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def stop(*_):
    raise KeyboardInterrupt


def main():
    # closing the ssh window sends SIGHUP and `kill` sends SIGTERM - treat both
    # like Ctrl+C so the LEDs always get switched off
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    subprocess.run(["v4l2-ctl", "-d", CAMERA_DEVICE, "-c",
                    "auto_exposure=1,white_balance_automatic=0,white_balance_temperature=5000,"
                    f"backlight_compensation=0,gain=0,exposure_time_absolute={EXPOSURE}"], check=True)
    subprocess.run(["sudo", "-n", "/usr/local/bin/rig-led", "all", *map(str, LED_COLOR)], check=True)
    ff = subprocess.Popen([
        "ffmpeg", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{W}x{H}", "-i", CAMERA_DEVICE,
        "-vf", "fps=3,format=gray", "-f", "rawvideo", "-",
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    threading.Thread(target=camera_loop, args=(ff,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Focus view: http://<pi-address>:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ff.terminate()
        subprocess.run(["sudo", "-n", "/usr/local/bin/rig-led", "off"])
        print(f"\nBest score seen: {state['best']:.1f}")


if __name__ == "__main__":
    main()
