from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
from PIL import Image, ImageDraw
import base64
import io
import json
import math

RAW_ROOT = Path("/home/noah/projects/image-identifier/data/coins/2026Data")
IMPORTANT_DIR = Path(__file__).resolve().parents[1]
OUT_LABELS = IMPORTANT_DIR / "labels" / "crop_labels.json"

app = Flask(__name__)

# Load folders
folders = [f for f in sorted(RAW_ROOT.rglob("*")) if f.is_dir()]
folder_index = 0

# Load existing labels if any
if OUT_LABELS.exists():
    with open(OUT_LABELS, "r") as f:
        labels = json.load(f)
else:
    labels = {}

def encode_image(img_path, clicks=None):
    """
    Convert image to base64 and draw click markers if provided.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw clicks
    if clicks:
        for (x, y) in clicks:
            r = 8
            draw.ellipse((x-r, y-r, x+r, y+r), fill="red")

        # If two clicks, draw circle
        if len(clicks) == 2:
            (cx, cy), (rx, ry) = clicks
            radius = math.sqrt((rx - cx)**2 + (ry - cy)**2)
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline="red",
                width=3
            )

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_first_image(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = [p for ext in exts for p in folder.glob(ext)]
    images = sorted(images)
    return images[0] if images else None

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Coin Labeler</title>
<style>
  body { font-family: sans-serif; }
  #img { border: 2px solid black; cursor: crosshair; }
  .info { margin-bottom: 10px; }
</style>
</head>
<body>
<h2>Folder: {{folder_name}}</h2>
<p class="info">Click CENTER of coin, then click a point on the RIM.</p>

<img id="img" src="data:image/jpeg;base64,{{img_data}}">

<script>
let clicks = [];

document.getElementById("img").addEventListener("click", function(e) {
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    clicks.push({x: x, y: y});
    console.log(clicks);

    if (clicks.length === 2) {
        fetch("/submit", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(clicks)
        }).then(r => window.location.reload());
    } else {
        // Refresh image with click marker
        fetch("/preview", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(clicks)
        }).then(r => r.json()).then(data => {
            document.getElementById("img").src = "data:image/jpeg;base64," + data.img;
        });
    }
});
</script>

</body>
</html>
"""

@app.route("/")
def index():
    global folder_index

    if folder_index >= len(folders):
        return "<h1>All folders labeled!</h1>"

    folder = folders[folder_index]
    img_path = get_first_image(folder)

    if img_path is None:
        folder_index += 1
        return index()

    img_data = encode_image(img_path)

    return render_template_string(
        HTML,
        folder_name=folder.name,
        img_data=img_data
    )

@app.route("/preview", methods=["POST"])
def preview():
    global folder_index
    clicks = request.get_json()

    folder = folders[folder_index]
    img_path = get_first_image(folder)

    # Convert click dicts to tuples
    click_list = [(float(c["x"]), float(c["y"])) for c in clicks]

    img_data = encode_image(img_path, click_list)
    return jsonify({"img": img_data})

@app.route("/submit", methods=["POST"])
def submit():
    global folder_index

    clicks = request.get_json()

    # Extract dict values properly
    c = clicks[0]
    r = clicks[1]

    cx = float(c["x"])
    cy = float(c["y"])
    rx = float(r["x"])
    ry = float(r["y"])

    r_val = math.sqrt((rx - cx)**2 + (ry - cy)**2)

    folder = folders[folder_index]
    labels[folder.name] = {"cx": cx, "cy": cy, "r": r_val}

    with open(OUT_LABELS, "w") as f:
        json.dump(labels, f, indent=2)

    folder_index += 1
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
