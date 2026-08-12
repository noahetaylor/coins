"""
year_labeler_web.py
===================
Type the year for each date band that has `year: null` in the DateBands
manifest — currently the 384 bands harvested from digitsannotations, which
are legible but arrived without provenance.

Knowing the year gives the four digit CLASSES for free, so every band
labeled here becomes usable digit-classification training data without
drawing a single box.

    python pipeline/year_labeler_web.py          # http://localhost:5001
    python pipeline/year_labeler_web.py --merge  # write results into manifest

NON-DESTRUCTIVE
---------------
Typing saves to labels/crops_years.json only. The manifest is untouched
until you explicitly run --merge, and --merge writes a .bak first.

KEYS
----
    0-9        type the year, auto-advances at 4 digits
    Enter      confirm a partial entry
    B          bicentennial (1776-1976)
    U          unreadable / skip
    <- ->      previous / next band
    Backspace  clear current entry
"""

import argparse
import base64
import csv
import json
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

IMPORTANT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = IMPORTANT_DIR.parents[1]
BANDS_DIR = PROJECT_ROOT / "data" / "coins" / "DateBands"
MANIFEST = BANDS_DIR / "manifest.json"
OUT_LABELS = IMPORTANT_DIR / "labels" / "crops_years.json"

DUAL_DATE = "1776-1976"
UNREADABLE = "unreadable"

app = Flask(__name__)
state = {"records": [], "index": 0, "labels": {}}


def load_labels():
    if OUT_LABELS.exists():
        return json.loads(OUT_LABELS.read_text())
    return {}


def save_labels(labels):
    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.write_text(json.dumps(labels, indent=2, sort_keys=True))


def load_unlabeled():
    """Bands still missing a year. Ordered so unlabeled ones come first."""
    manifest = json.loads(MANIFEST.read_text())
    return [r for r in manifest if r.get("year") is None]


def encode_band(band_name):
    return base64.b64encode((BANDS_DIR / "bands" / band_name).read_bytes()).decode()


HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Year Labeler</title>
<style>
  body { font-family: system-ui, sans-serif; background:#1e1e1e; color:#eee;
         margin:0; padding:20px; }
  #band { width:100%; max-width:1100px; border:2px solid #555; border-radius:6px;
          image-rendering:-webkit-optimize-contrast; background:#000; }
  .row { max-width:1100px; margin:0 auto; }
  #entry { font-size:56px; letter-spacing:14px; font-family:ui-monospace,monospace;
           color:#4ec9b0; min-height:70px; }
  #meta { color:#888; font-size:13px; margin-bottom:6px; word-break:break-all; }
  #progress { font-size:15px; margin-bottom:10px; }
  .bar { height:6px; background:#333; border-radius:3px; overflow:hidden; margin-bottom:14px; }
  .bar > div { height:100%; background:#4ec9b0; }
  .keys { color:#777; font-size:13px; margin-top:14px; line-height:1.8; }
  kbd { background:#333; border:1px solid #555; border-radius:3px; padding:1px 6px; color:#ccc; }
  .done { color:#6a9955; } .skip { color:#c586c0; }
</style>
</head>
<body>
<div class="row">
  <div id="progress"></div>
  <div class="bar"><div id="barfill" style="width:0%"></div></div>
  <div id="meta"></div>
  <img id="band" src="">
  <div id="entry"></div>
  <div class="keys">
    <kbd>0-9</kbd> type year (auto-advance) &nbsp;
    <kbd>B</kbd> bicentennial &nbsp;
    <kbd>U</kbd> unreadable &nbsp;
    <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &nbsp;
    <kbd>Backspace</kbd> clear
    <br><span id="saved"></span>
  </div>
</div>
<script>
let entry = "";
let cur = null;

async function load(delta) {
  const r = await fetch("/band" + (delta !== undefined ? "?delta=" + delta : ""));
  cur = await r.json();
  document.getElementById("band").src = "data:image/jpeg;base64," + cur.image;
  document.getElementById("meta").textContent = cur.band + "   |   " + cur.src_image;
  document.getElementById("progress").innerHTML =
    "Band " + (cur.index + 1) + " / " + cur.total +
    "  —  <span class='done'>" + cur.labeled + " labeled</span>" +
    (cur.skipped ? "  —  <span class='skip'>" + cur.skipped + " unreadable</span>" : "");
  document.getElementById("barfill").style.width =
    (100 * (cur.labeled + cur.skipped) / cur.total) + "%";
  entry = "";
  render(cur.existing || "");
}

function render(text) {
  const e = document.getElementById("entry");
  e.textContent = text || entry || "____".slice(entry.length ? entry.length : 0);
  if (!entry && cur && cur.existing) { e.textContent = cur.existing; }
  else { e.textContent = entry.padEnd(4, "_"); }
}

async function submit(value) {
  await fetch("/submit", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({band: cur.band, value: value})
  });
  document.getElementById("saved").textContent = "saved " + cur.band + " = " + value;
  load(1);
}

document.addEventListener("keydown", (ev) => {
  if (ev.key >= "0" && ev.key <= "9") {
    entry += ev.key; render();
    if (entry.length === 4) submit(entry);
  } else if (ev.key === "Enter" && entry.length) { submit(entry); }
  else if (ev.key === "Backspace") { ev.preventDefault(); entry = ""; render(); }
  else if (ev.key === "b" || ev.key === "B") { submit("BICENTENNIAL"); }
  else if (ev.key === "u" || ev.key === "U") { submit("UNREADABLE"); }
  else if (ev.key === "ArrowRight") { load(1); }
  else if (ev.key === "ArrowLeft") { load(-1); }
});
load();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/band")
def band():
    recs = state["records"]
    delta = request.args.get("delta")
    if delta is not None:
        state["index"] = (state["index"] + int(delta)) % len(recs)
    else:
        # resume at the first unlabeled band
        for i, r in enumerate(recs):
            if r["band"] not in state["labels"]:
                state["index"] = i
                break
    rec = recs[state["index"]]
    labels = state["labels"]
    return jsonify({
        "band": rec["band"],
        "src_image": rec["src_image"],
        "image": encode_band(rec["band"]),
        "index": state["index"],
        "total": len(recs),
        "existing": labels.get(rec["band"], ""),
        "labeled": sum(1 for v in labels.values() if v != UNREADABLE),
        "skipped": sum(1 for v in labels.values() if v == UNREADABLE),
    })


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    value = data["value"]
    if value == "BICENTENNIAL":
        value = DUAL_DATE
    elif value == "UNREADABLE":
        value = UNREADABLE
    state["labels"][data["band"]] = value
    save_labels(state["labels"])
    return jsonify({"ok": True})


def merge_into_manifest():
    labels = load_labels()
    if not labels:
        print("no labels in", OUT_LABELS)
        return
    manifest = json.loads(MANIFEST.read_text())
    shutil.copy2(MANIFEST, MANIFEST.with_suffix(".json.bak"))

    by_band = {r["band"]: r for r in manifest}
    applied = skipped = 0
    for band, value in labels.items():
        rec = by_band.get(band)
        if rec is None:
            continue
        if value == UNREADABLE:
            rec["year_unreadable"] = True
            skipped += 1
            continue
        if value == DUAL_DATE:
            rec["year"], rec["digits"] = 1976, DUAL_DATE
        else:
            rec["year"], rec["digits"] = int(value), value
        rec["year_source"] = "hand-read"
        applied += 1

    MANIFEST.write_text(json.dumps(manifest, indent=2))

    # Keep the CSV in step with the JSON, or downstream code reading the CSV
    # silently sees the pre-merge state.
    csv_path = MANIFEST.with_name("manifest.csv")
    fields = ["band", "source", "coin_id", "year", "mint", "digits",
              "src_image", "method", "has_digit_labels", "year_source",
              "year_unreadable"]
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(manifest)

    known = sum(1 for r in manifest if r.get("digits"))
    print(f"applied {applied} years, {skipped} marked unreadable")
    print(f"backup at {MANIFEST.with_suffix('.json.bak')}")
    print(f"manifest now has {known} / {len(manifest)} bands with known digits")
    print(f"rewrote {csv_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true",
                    help="write collected years into manifest.json (makes a .bak)")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    if args.merge:
        merge_into_manifest()
        return

    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST} - run harvest_date_bands.py first")

    state["records"] = load_unlabeled()
    state["labels"] = load_labels()
    print(f"{len(state['records'])} bands need a year; "
          f"{len(state['labels'])} already done")
    print(f"open http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
