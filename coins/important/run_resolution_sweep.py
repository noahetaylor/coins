#!/usr/bin/env python3
"""
Resolution sweep for the date reader.

The harvested bands are 1346-1674px wide but the reader resizes them to
320x120, which leaves a ResNet-18 feature map of only 4x10. The column-aligned
heads then split those 10 columns across 4 digit positions - 2.5 columns per
digit. This sweep asks whether that is the accuracy bottleneck.

    320x120 -> featmap 4x10  (2.5 cols/digit)
    448x168 -> featmap 6x14  (3.5 cols/digit)
    640x240 -> featmap 8x20  (5.0 cols/digit)

Each point gets a full 5-fold coin-level cross-validation, so every one of the
751 coins gets a held-out prediction. That matters: per-coin accuracy is at
99.07% (7 failures), so only pooled per-band accuracy has the power to resolve
a 1-2% difference.

Runs are sequential - the box has 8 torch threads and ~4GB of RAM, so parallel
points would thrash. Already-finished points are skipped, so the sweep can be
resumed after a crash.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "models" / "resolution_sweep"
TRAIN = HERE / "train_date_reader_v1.py"
PYTHON = HERE.parents[1] / ".venv" / "bin" / "python"

RESOLUTIONS = ["320x120", "448x168", "640x240",
               "800x304", "960x360", "1280x480"]
COMMON = ["--folds", "5", "--seed", "0", "--epochs", "30", "--wear-aug",
          # Workers hold decoded full-size bands in host RAM, which is what
          # actually blew up at high resolution - the GPU is the bottleneck
          # now, not data loading, so fewer workers costs nothing.
          "--workers", "4"]


def _make_oom_victim():
    """Mark the training child as the preferred OOM-kill target.

    The VS Code server processes score 686-710, so an out-of-memory event was
    reaping the IDE and dropping the WSL connection mid-run. Scoring the
    trainer at 1000 means a memory spike kills the job we can restart, not the
    session we cannot.
    """
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass


def run_point(res):
    out_dir = SWEEP_DIR / res
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cross_val_report.json"
    log = out_dir / "train.log"

    if report.exists():
        print(f"[{res}] already done, skipping", flush=True)
        return json.loads(report.read_text())

    cmd = [str(PYTHON), str(TRAIN), *COMMON,
           "--in-size", res, "--out", str(out_dir / "date_reader.pth")]
    print(f"[{res}] starting: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE,
                             preexec_fn=_make_oom_victim)
    dt = time.time() - t0

    if rc != 0:
        # Exit 137 is the OOM killer; anything else is a real failure. Either
        # way, do not write a report - that keeps this point resumable.
        print(f"[{res}] FAILED rc={rc} after {dt/60:.1f}min - see {log}",
              flush=True)
        return None

    data = json.loads(report.read_text())
    data["wall_seconds"] = dt
    report.write_text(json.dumps(data, indent=2))
    print(f"[{res}] done in {dt/60:.1f}min  "
          f"band {data['pooled_band_acc']:.2%}  "
          f"coin {data['pooled_coin_acc']:.2%}", flush=True)
    return data


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for res in RESOLUTIONS:
        data = run_point(res)
        if data:
            results[res] = data

    if not results:
        print("\nno points completed")
        return 1

    print("\n" + "=" * 78)
    print("RESOLUTION SWEEP  (5-fold coin-level CV, wear-aug, 30 epochs, seed 0)")
    print("=" * 78)
    print(f"{'res':>9} {'cols/digit':>11} {'band mean':>11} {'band std':>9} "
          f"{'pooled band':>12} {'pooled coin':>12} {'hours':>7}")
    for res, d in results.items():
        cols = (d["in_w"] / 32) / 4
        print(f"{res:>9} {cols:>11.1f} {d['per_band_mean']:>10.2%} "
              f"{d['per_band_std']:>8.2%} {d['pooled_band_acc']:>11.2%} "
              f"{d['pooled_coin_acc']:>11.2%} "
              f"{d.get('wall_seconds', 0)/3600:>6.1f}h")

    base = results.get("320x120")
    if base:
        print(f"\ndelta vs 320x120 baseline (pooled per-band):")
        for res, d in results.items():
            if res == "320x120":
                continue
            delta = d["pooled_band_acc"] - base["pooled_band_acc"]
            # One fold-std is the noise floor; call anything under it a wash.
            note = "" if abs(delta) > base["per_band_std"] else "  (within fold noise)"
            print(f"  {res:>9} {delta:+.2%}{note}")

    print(f"\nper-point failure lists:")
    for res, d in results.items():
        ids = ", ".join(f["coin_id"].split("/")[-1] for f in d["failures"])
        print(f"  {res:>9} {len(d['failures']):>2} coins: {ids}")

    summary = SWEEP_DIR / "sweep_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
