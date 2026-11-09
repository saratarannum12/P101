#!/usr/bin/env python3
"""
s3_batch.py  —  S3 Batch Mode

Runs the Max-Div alignment pipeline on every video clip in a folder,
collects per-clip accuracy statistics, and writes a CSV summary table.

USAGE
-----
  # Run all clips with the same known true azimuth:
  python s3_batch.py --batch C:/Users/sarat/Downloads/SaraCode --true-azimuth 0

  # Run without a known azimuth (direction tracking only, no error column):
  python s3_batch.py --batch C:/Users/sarat/Downloads/SaraCode

  # Specify output CSV path:
  python s3_batch.py --batch ./clips --true-azimuth 0 --out results.csv

OUTPUT CSV COLUMNS
------------------
  clip               — filename
  total_frames       — total frames in the video
  valid_frames       — frames where estimator was confident enough to record
  coverage_pct       — valid_frames / total_frames * 100
  mean_est_az_deg    — mean estimated azimuth over valid frames
  true_azimuth_deg   — ground truth azimuth (blank if not supplied)
  mean_error_deg     — mean absolute angular error (blank if no true azimuth)
  median_error_deg   — median absolute angular error
  max_error_deg      — worst-case absolute angular error
  mean_tau_s         — mean TTC estimate over valid frames (seconds)
  status             — OK / NO_VALID_FRAMES / ERROR
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

# ── optional matplotlib for per-clip plots ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

from maxdiv_perception import MaxDivProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Constants (match m3m4.py)
# ─────────────────────────────────────────────────────────────────────────────
MIN_VECTORS       = 20
MIN_AXIS_STRENGTH = 0.005

VIDEO_EXTENSIONS  = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MOV", ".MP4"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def angular_diff(est, true):
    """Signed circular difference est - true, in [-180, 180)."""
    return (est - true + 180.0) % 360.0 - 180.0


def process_clip(video_path, true_azimuth=None, save_plot=False):
    """
    Run the full pipeline on one video file (headless — no OpenCV windows).

    Returns a dict of per-clip stats, or None on hard failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "ERROR", "detail": "could not open file"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt           = 1.0 / video_fps

    proc = MaxDivProcessor()
    proc.reset()

    # prime on the first frame
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return {"status": "ERROR", "detail": "could not read first frame"}
    proc.process(frame, dt=0.0, want_render=False)

    frame_nums  = []
    est_angles  = []
    tau_values  = []

    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = proc.process(frame, dt=dt, want_render=False)

        if (res.valid
                and res.n_vectors >= MIN_VECTORS
                and res.axis_strength >= MIN_AXIS_STRENGTH):
            avx, avy = res.arc_vec
            angle_est = math.degrees(math.atan2(avy, avx))
            frame_nums.append(frame_idx)
            est_angles.append(angle_est)
            if res.tau_s is not None and math.isfinite(res.tau_s) and res.tau_s > 0:
                tau_values.append(res.tau_s)

        frame_idx += 1

    cap.release()

    n = len(est_angles)
    clip_name = os.path.basename(video_path)

    if n == 0:
        return {
            "clip":             clip_name,
            "total_frames":     total_frames,
            "valid_frames":     0,
            "coverage_pct":     0.0,
            "mean_est_az_deg":  "",
            "true_azimuth_deg": true_azimuth if true_azimuth is not None else "",
            "mean_error_deg":   "",
            "median_error_deg": "",
            "max_error_deg":    "",
            "mean_tau_s":       "",
            "status":           "NO_VALID_FRAMES",
        }

    mean_est  = sum(est_angles) / n
    mean_tau  = sum(tau_values) / len(tau_values) if tau_values else ""

    if true_azimuth is not None:
        errors       = [abs(angular_diff(e, true_azimuth)) for e in est_angles]
        mean_err     = sum(errors) / n
        median_err   = sorted(errors)[n // 2]
        max_err      = max(errors)

        if save_plot and _MPL:
            _save_plot(video_path, true_azimuth, frame_nums, est_angles, errors)
    else:
        errors = mean_err = median_err = max_err = None

    return {
        "clip":             clip_name,
        "total_frames":     total_frames,
        "valid_frames":     n,
        "coverage_pct":     round(n / total_frames * 100, 1) if total_frames > 0 else 0.0,
        "mean_est_az_deg":  round(mean_est, 1),
        "true_azimuth_deg": true_azimuth if true_azimuth is not None else "",
        "mean_error_deg":   round(mean_err, 1)   if mean_err   is not None else "",
        "median_error_deg": round(median_err, 1) if median_err is not None else "",
        "max_error_deg":    round(max_err, 1)    if max_err    is not None else "",
        "mean_tau_s":       round(mean_tau, 3)   if mean_tau != "" else "",
        "status":           "OK",
    }


def _save_plot(video_path, true_az, frame_nums, est_angles, errors):
    clip_stem = os.path.splitext(os.path.basename(video_path))[0]
    plot_path = f"summary_{clip_stem}.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"S3 Batch Summary — {clip_stem}\n"
        f"True azimuth: {true_az:.1f}°   Mean |error|: {sum(errors)/len(errors):.1f}°",
        fontsize=12
    )
    ax1.plot(frame_nums, est_angles, color="tab:blue", linewidth=1.0,
             label="Estimated azimuth")
    ax1.axhline(true_az, color="tab:orange", linewidth=1.5, linestyle="--",
                label=f"True ({true_az:.1f}°)")
    ax1.set_ylabel("Azimuth (deg)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-190, 190)

    ax2.fill_between(frame_nums, errors, color="tab:red", alpha=0.5, label="|error|")
    ax2.axhline(sum(errors) / len(errors), color="darkred", linewidth=1.2,
                linestyle="--",
                label=f"Mean ({sum(errors)/len(errors):.1f}°)")
    ax2.set_ylabel("|Angular error| (deg)")
    ax2.set_xlabel("Frame number")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"    plot → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "clip", "total_frames", "valid_frames", "coverage_pct",
    "mean_est_az_deg", "true_azimuth_deg",
    "mean_error_deg", "median_error_deg", "max_error_deg",
    "mean_tau_s", "status",
]


def write_csv(rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[S3] CSV saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="S3 Batch Mode — run Max-Div pipeline on all clips in a folder"
    )
    ap.add_argument("--batch", required=True, metavar="FOLDER",
                    help="Folder containing video clips to process")
    ap.add_argument("--true-azimuth", type=float, default=None, metavar="DEGREES",
                    help="Known ground-truth tilt azimuth for ALL clips (optional)")
    ap.add_argument("--out", type=str, default=None, metavar="FILE",
                    help="Output CSV path (default: batch_results.csv in the batch folder)")
    ap.add_argument("--save-plots", action="store_true",
                    help="Save a per-clip azimuth plot PNG alongside the CSV")
    args = ap.parse_args()

    folder = args.batch
    if not os.path.isdir(folder):
        print(f"[ERROR] Not a folder: {folder}")
        sys.exit(1)

    # find all video files
    clips = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1] in VIDEO_EXTENSIONS
    ])

    if not clips:
        print(f"[ERROR] No video files found in: {folder}")
        print(f"        Looking for: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)

    out_path = args.out or os.path.join(folder, "batch_results.csv")

    print("=" * 60)
    print("S3  BATCH MODE")
    print("=" * 60)
    print(f"  folder        : {folder}")
    print(f"  clips found   : {len(clips)}")
    if args.true_azimuth is not None:
        print(f"  true azimuth  : {args.true_azimuth:.1f} deg")
    else:
        print(f"  true azimuth  : (not supplied — no error column)")
    print(f"  output CSV    : {out_path}")
    print()

    rows = []
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        print(f"[{i}/{len(clips)}] {clip_name} ...", end="", flush=True)
        row = process_clip(clip_path, args.true_azimuth, save_plot=args.save_plots)
        if row is None:
            row = {"clip": clip_name, "status": "ERROR"}
            # fill missing fields with empty string
            for f in CSV_FIELDS:
                row.setdefault(f, "")

        status = row.get("status", "?")
        if status == "OK":
            valid = row["valid_frames"]
            total = row["total_frames"]
            cov   = row["coverage_pct"]
            err   = row.get("mean_error_deg", "")
            err_s = f"  mean_err={err}°" if err != "" else ""
            print(f"  {valid}/{total} valid frames ({cov}%){err_s}")
        else:
            print(f"  {status}")

        rows.append(row)

    write_csv(rows, out_path)

    # ── summary to console ────────────────────────────────────────────────────
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    print()
    print("=" * 60)
    print(f"  Processed : {len(clips)} clips")
    print(f"  OK        : {len(ok_rows)}")
    print(f"  Failed    : {len(clips) - len(ok_rows)}")
    if ok_rows and args.true_azimuth is not None:
        all_errors = [r["mean_error_deg"] for r in ok_rows if r.get("mean_error_deg") != ""]
        if all_errors:
            overall = sum(all_errors) / len(all_errors)
            print(f"  Overall mean |error| across all clips: {overall:.1f} deg")
    print("=" * 60)


if __name__ == "__main__":
    main()