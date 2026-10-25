#!/usr/bin/env bash
# Launch the resolution sweep fully detached from this shell.
#
# VS Code's WSL connection has been dropping mid-run. A normal background job
# dies with its parent shell; setsid + nohup reparent the sweep to init so it
# survives the terminal, the VS Code server, and the reconnect. Reattach by
# tailing the log - the driver skips resolutions that already finished.
set -euo pipefail
cd "$(dirname "$0")"
LOG=models/resolution_sweep_driver.log

if pgrep -f run_resolution_sweep.py >/dev/null; then
    echo "sweep already running (pid $(pgrep -f run_resolution_sweep.py | head -1))"
    echo "follow it with:  tail -f $(pwd)/$LOG"
    exit 0
fi

mkdir -p models/resolution_sweep
setsid nohup ../../.venv/bin/python -u run_resolution_sweep.py \
    >"$LOG" 2>&1 < /dev/null &
disown || true
sleep 2
echo "sweep detached, pid $(pgrep -f run_resolution_sweep.py | head -1)"
echo "follow it with:  tail -f $(pwd)/$LOG"
