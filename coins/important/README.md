# Coin Processing Workspace

Reads the mint year off a photo of a quarter. The active pipeline finds the
coin, crops the date band, and reads the 4 digits with a column-aligned CNN.
As of 2026-08-30: 95.6% per-band accuracy, 99.1% per-coin accuracy (majority
vote across a coin's bands) on 5-fold cross-validation over 751 coins. Every
remaining failure is a coin from the `2026data` studio rig, not the scraped
grading-site sources, which sit near 100%.

## Active Pipeline

1. `harvest_date_bands.py` - locates each coin's circle geometry and crops
   the date band (the strip containing the 4-digit year) from raw source
   images across all sources (`2026data`, `pcgs`, `crops`, `supers`). Writes
   `data/coins/DateBands/manifest.json` plus the cropped band images.
2. `train_date_reader_v1.py` - trains a ResNet-18 trunk with 4 column-aligned
   heads, one per digit position, on the harvested bands. Column alignment
   (not global average pooling) is required - see the docstring on
   `DateReader` for why.

Run 5-fold coin-level cross-validation (the right way to evaluate changes -
splits by `coin_id` so the same physical coin never appears in both train and
val under different lighting):

```bash
python coins/important/train_date_reader_v1.py --folds 5 --epochs 30 --wear-aug
```

Train a single deployable checkpoint:

```bash
python coins/important/train_date_reader_v1.py --epochs 30 --wear-aug \
    --out coins/important/models/date_reader_v1.pth
```

Key flags:

- `--wear-aug` - degrades pristine scraped coins toward studio-rig conditions
  (blur, low contrast, colour cast) instead of training on clean images only.
  The failures concentrate entirely on the rig domain, so this augmentation
  targets the regime that actually matters.
- `--in-size WxH` - model input resolution, default `448x168`. A 6-point
  sweep on 2026-08-29 (`models/resolution_sweep/`) found this is the peak:
  320x120 undersamples the band, but pushing past 448x168 makes things
  *worse* (1280x480 is -3.1% pooled band accuracy, with 3x the fold
  variance) - more pixels gives the augmentation pipeline more room to
  distort, not more signal. Resolution is a closed question; don't revisit
  it without new evidence.

Runs on GPU automatically if available (`DEVICE` picks CUDA when present),
falls back to CPU otherwise.

## Raking-Light Capture

The old rig capture (`rig/capture_rotation.py`) swept a bright spot around
the LED ring in 7 colours, but kept every other LED at 45% plus a 30% white
fill. That much fill light washed out the shading, so the struck relief of a
worn date barely showed, and the sweep mostly varied colour, which carries no
information about relief.

`rig/capture_raking.py` replaces it with a photometric-stereo style capture:
a dark frame, four pure 5-LED arcs (N/E/S/W, all other LEDs off), and one
flat-lit frame, with exposure and white balance locked so the directional
differences survive. `fuse_raking.py` turns the four directions into a single
relief-enhanced image (per-pixel slope from the E-W and N-S ratios, re-shaded
as if lit from a low angle). `build_raking_bands.py` builds a DateBands
variant where only the rig coins are swapped for the new captures, and
`train_date_reader_v1.py --bands-root` trains on it (`start_raking_cv.sh`).

Result, same 50 rig coins, same 5-fold cross-validation
(`models/raking_cv/`):

| Rig capture | Rig coins read correctly |
|---|---|
| Old colour sweep | 42 / 50 (84%) |
| Raking light, directional frames | 44 / 50 (88%) |
| Raking light, fused | 44 / 50 (88%) |

The gain comes from the lighting itself; fusing the frames ties with using
the directional frames directly. It is a modest gain from one training seed.

`rig/demo_station.py` is a one-button scan page: it triggers a raking capture
on the Pi, pulls the frames back and votes a year across them.

## Legacy / Not Currently Active

These directories represent an earlier approach (crop the coin -> correct
its rotation -> classify or detect the date region) that the project moved
away from in favour of the geometric band-extraction approach above. Kept on
disk for reference; do not resume without new evidence they're needed.

- `pipeline/`, `training/`, `preprocessing.py`, `build_crop_dataset.py`,
  `train_crop_model_v2.py`, `eval_crop_model_v2.py` - the crop pipeline.
  `crop_model_v2.pth` predicts a near-constant: cx varies ~40px on a 4656px
  image because the rig is fixed, so a fixed circle or Hough transform
  solves this without a model. `harvest_date_bands.py` already does the
  simpler thing.
- `build_rotation_dataset_v2.py`, `train_rotation_model_v3_cpu.py`,
  `eval_rotation_model.py`, `legacy/` - the rotation branch (v1-v4). Solves a
  non-problem: coins in `2026Data` are already upright within ~±15°, and the
  `rot_train_*` folders were made by hand-rotating already-upright coins.
- `date_detection/` - a YOLO-based date-region + digit detector. Nothing in
  the active pipeline imports from this folder; superseded by the simpler
  manual-crop-label + circle-geometry approach in `harvest_date_bands.py`.

## Data

- `data/coins/DateBands/manifest.json` - one record per harvested band:
  `coin_id`, `source`, `year`/`digits`, `band_box`, circle geometry.
  `2026data` is 53 physical coins (751 coins total across all sources); split
  cross-validation by `coin_id`, never by image, or the same coin ends up on
  both sides of the split under different lighting.
- CPU-only training takes ~10 min for 320x120 at 30 epochs; GPU cuts this to
  minutes. See `models/resolution_sweep/sweep_summary.json` for the full
  per-resolution timing and accuracy breakdown.
