# Coin Processing Workspace

This folder contains the active coin crop/rotation pipeline, training scripts, model weights, labels, older rotation experiments, and date/digit detection utilities.

## Layout

- `pipeline/` - scripts that process raw coin images into cropped or rotated outputs.
- `training/` - scripts that build datasets or train crop and rotation models.
- `models/` - local model weights used by the pipeline.
- `labels/` - manual crop labels.
- `date_detection/` - YOLO date-region and digit-recognition workflow.
- `legacy/` - older rotation experiments kept for reference.

## Active Crop/Rotation Pipeline

- `pipeline/preprocess_dataset.py` - rebuilt manual-label preprocessing path. Uses `labels/crop_labels.json`, crops flush to the coin, predicts folder-level rotation with `models/fine_rotation_full.pth`, writes centered upright 512x512 images to `data/coins/RotatedDatasetManualV2`, and saves debug artifacts.
- `pipeline/process_with_labels.py` - compatibility entrypoint that runs `pipeline/preprocess_dataset.py`.
- `pipeline/click_labeler_web.py` - browser-based manual labeling tool for coin center/rim labels.
- `pipeline/rotate_opencv.py` - ML-crop version of the pipeline. Kept active but experimental until the crop model is improved.
- `pipeline/test_crop_rotate_debug.py` - debug runner that saves crop/rotation visualization images for model predictions.
- `preprocessing.py` - shared crop, rotation, path, model-loading, and visualization helpers used by the pipeline scripts.

Small top-level launchers are kept for `process_with_labels.py`, `rotate_opencv.py`, and `test_crop_rotate_debug.py` so older editor tabs and commands still work.

Run the rebuilt manual-label pipeline:

```bash
python coins/important/pipeline/process_with_labels.py
```

Output goes to `data/coins/RotatedDatasetManualV2`.

Debug images go to `coins/important/debug/preprocess_v2`.

Run the manual labeler:

```bash
python coins/important/pipeline/click_labeler_web.py
```

Run the experimental ML-crop pipeline:

```bash
python coins/important/pipeline/rotate_opencv.py
```

Create crop-model debug overlays:

```bash
python coins/important/pipeline/test_crop_rotate_debug.py --limit 20 --mask
```

## Training

- `training/train_crop_model.py` - trains `models/crop_model.pth` from `labels/crop_labels.json`.
- `training/1800_rotater.py` - current fine rotation regressor trainer; writes `models/fine_rotation_full.pth`.
- `training/train_fine_rotation_regressor.py` - older fine rotation trainer; writes `models/fine_rotation_model.pth`.
- `training/build_fine_rotation_dataset.py` - builds `data/coins/FineRotationDataset`.
- `training/split_rotation_dataset.py` - splits coarse rotation training data.

Train the crop model:

```bash
python coins/important/training/train_crop_model.py --epochs 60 --samples-per-folder 0
```

The crop trainer validates by folder, uses bounded outputs, and prints normalized loss plus approximate pixel error.

## Model And Label Files

- `models/crop_model.pth` - experimental crop model.
- `models/fine_rotation_full.pth` - current rotation model used by the trusted pipeline.
- `models/fine_rotation_model.pth` - older fine rotation model.
- `models/rotation_model.pth` - older 0/90/180/270 classifier.
- `labels/crop_labels.json` - manual per-folder crop circle labels.

## Date Detection

- `date_detection/Bounds.py` - main YOLO date-region and digit recognition pipeline.
- `date_detection/yolofilenamereorganizer.py` - converts year-labeled image folders into YOLO train/validation directories.
- `date_detection/digitreorganizer.py` - builds a YOLO-formatted digit training dataset from labeled digit crops.
- `date_detection/cropsaver.py` - extracts cropped date-region images from YOLO label files.
- `date_detection/traindigitsyolo.py` - configurable digit model training script.
- `date_detection/Cropper.py` - raw image cropping helper for splitting original coin scans.

Train the date detector:

```bash
python coins/important/date_detection/Bounds.py --mode train-date --base-dir data/coins/Bounds/Heads --date-weights yolov8n.pt --epochs-date 30
```

Train the digit detector:

```bash
python coins/important/date_detection/Bounds.py --mode train-digit --base-dir data/coins/Bounds/Heads --digit-weights yolov8n.pt --epochs-digit 40
```

Evaluate the date/digit pipeline:

```bash
python coins/important/date_detection/Bounds.py --mode eval --base-dir data/coins/Bounds/Heads --failure-csv coins/important/failures.csv
```

## Notes

The scripts that use labels or weights resolve those paths relative to `coins/important`, so they can be run from the project root without depending on the current working directory.
