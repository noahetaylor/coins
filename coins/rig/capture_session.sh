#!/bin/bash
# capture_session.sh - walk through every coin in rig_labels.txt with capture_raking.py.
#
#   ./capture_session.sh               # resumes: coins already in data/raking/ are skipped
#
# For each coin: place it date-DOWN (date at the bottom of the screen, upright
# within ~15 deg), press Enter. 's' skips a coin, 'q' quits. If a capture
# warns (clipping, too dark, room light), you can retake it straight away.
cd "$(dirname "$0")"
OUT=/home/noah/coin-sorter/data/raking
mapfile -t LABELS < rig_labels.txt
total=${#LABELS[@]}

for i in "${!LABELS[@]}"; do
    L=${LABELS[$i]}
    [ -z "$L" ] && continue
    if [ -e "$OUT/$L/capture.json" ]; then
        echo "[$((i+1))/$total] $L already captured, skipping"
        continue
    fi
    while true; do
        read -rp "[$((i+1))/$total] Place $L date-down, Enter to capture (s=skip, q=quit): " a
        case "$a" in
            q) exit 0 ;;
            s) break ;;
        esac
        ./capture_raking.py "$L" && break
        read -rp "  Capture warned. r=retake, k=keep it anyway: " b
        if [ "$b" = "k" ]; then break; fi
        rm -rf "${OUT:?}/$L"
    done
done
echo "Session complete: $(ls "$OUT" | grep -vc '^test_') coins in $OUT"
