#!/usr/bin/env bash
# Cross-validate the date reader on the raking-light rig captures (photos, then
# fused), with the same settings as the 448x168 resolution-sweep point so the
# only change from that baseline is the rig images. Detached like start_sweep.sh.
set -euo pipefail
cd "$(dirname "$0")"
OUT=models/raking_cv
mkdir -p "$OUT"
setsid nohup bash -c '
for mode in photos fused; do
    mkdir -p '"$OUT"'/$mode
    ../../.venv/bin/python -u train_date_reader_v1.py \
        --bands-root ../../data/coins/DateBands_raking_$mode \
        --folds 5 --seed 0 --epochs 30 --wear-aug --workers 4 --in-size 448x168 \
        --out '"$OUT"'/$mode/date_reader.pth > '"$OUT"'/$mode/train.log 2>&1
done' > "$OUT/driver.log" 2>&1 < /dev/null &
disown || true
echo "started; follow with: tail -f $(pwd)/$OUT/photos/train.log"
