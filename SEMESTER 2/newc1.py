#!/usr/bin/env python3
"""
m3m4.py  —  Milestones 1 + 2 + 3 + 4 + C1 + Webcam

Builds on m2.py by adding:

  M3 — Proximity readout
       A clear visual bar and text showing time-to-contact (tau_s) and
       how close the camera is to the surface. The bar grows as the
       camera approaches.

  M4 — Direction-to-frontal indicator + confidence flag
       A large plain-language panel showing:
         "NEAR FRONTAL"   — camera is already facing the surface square-on
         "TILT THIS WAY"  — with an arrow showing which direction to tilt
         "LOW CONFIDENCE" — when the estimate cannot be trusted (too few
                            vectors or the shear axis is too weak)

  C1 — Live RealSense camera input
       Pass --live instead of --video to read from a connected D415 camera.
       All overlays (M3 proximity bar, M4 direction panel) work identically.

  --webcam — Built-in / USB webcam input
       Pass --webcam to read from your laptop camera (or any webcam).
       Uses cv2.VideoCapture(0). No RealSense required.
       All overlays work identically to video and live modes.

CONTROLS
--------
  Space       — pause / resume  (video mode only)
  D or →      — step forward one frame (when paused, video mode only)
  A or ←      — step backward one frame (when paused, video mode only)
  R           — toggle recording
  Q or Escape — quit

USAGE
-----
  # Video file:
  python m3m4.py --video path/to/clip.mp4 --no-rotate
  python m3m4.py --video clip.mp4

  # Laptop / USB webcam:
  python m3m4.py --webcam
  python m3m4.py --webcam --no-rotate

  # Live RealSense camera (C1):
  python m3m4.py --live
  python m3m4.py --live --no-rotate
"""

import cv2
import argparse
import os
import sys
import math
import numpy as np

from maxdiv_perception import MaxDivProcessor
from maxdiv_viz import MaxDivVisualizer

# ── Optional RealSense import (C1) ───────────────────────────────────────────
try:
    from realsense_cue import RealSenseCue
    _REALSENSE_OK = True
except ImportError:
    _REALSENSE_OK = False

# ─────────────────────────────────────────────
# Thresholds (tune these based on Chris's feedback)
# ─────────────────────────────────────────────
MIN_VECTORS       = 20      # below this n_vectors -> LOW CONFIDENCE
MIN_AXIS_STRENGTH = 0.005   # below this axis_strength -> NEAR FRONTAL
TAU_MAX_DISPLAY   = 10.0    # seconds — bar is empty beyond this distance
TAU_WARN          = 2.0     # seconds — bar turns red below this

# ─────────────────────────────────────────────
# Command Line Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Max-Div Validator — M3 Proximity + M4 Direction-to-Frontal + C1 Live"
)

# Input source — must supply exactly one of --video, --webcam, or --live
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument("--video", type=str, metavar="FILE",
                          help="Path to the input video file")
source_group.add_argument("--webcam", action="store_true",
                          help="Read from built-in or USB webcam (cv2.VideoCapture(0))")
source_group.add_argument("--live", action="store_true",
                          help="Read from a live RealSense D415 camera (C1)")

parser.add_argument("--fps", type=float, default=None,
                    help="Override playback FPS (video mode only)")
parser.add_argument("--no-rotate", action="store_true",
                    help="Do not rotate 180 deg (use for normal hand-held footage)")

# ── S2 end-of-clip summary ────────────────────────────────────────────────────
parser.add_argument("--true-azimuth", type=float, default=None, metavar="DEGREES",
                    help="Known ground-truth tilt azimuth (deg). "
                         "When supplied, prints per-clip accuracy summary on exit.")

args = parser.parse_args()

# ─────────────────────────────────────────────
# Validate inputs
# ─────────────────────────────────────────────
if args.live and not _REALSENSE_OK:
    print("[ERROR] --live requires realsense_cue.py and pyrealsense2.")
    print("        Run: pip install pyrealsense2")
    print("        and make sure realsense_cue.py is in the same folder.")
    sys.exit(1)

rotate_180 = not args.no_rotate

# ─────────────────────────────────────────────
# Input source setup
# ─────────────────────────────────────────────
if args.live:
    # ── C1: Live RealSense ────────────────────────────────────────────────
    print("[INFO] C1 — Live RealSense mode")
    cue = RealSenseCue()
    cue.start()

    # For live mode we use playback_fps / frame_delay for display pacing only
    video_fps    = 60.0
    playback_fps = 60.0
    frame_delay  = 16          # ~60 fps display loop; actual timing driven by camera
    total_frames = None        # unknown for live feed
    cap          = None

    print("[INFO] RealSense started. Press Q or Esc to stop.")
    print()
    print("[INFO] Controls: R=Rec  Q=Quit")

elif args.webcam:
    # ── Webcam: built-in or USB camera ───────────────────────────────────
    print("[INFO] Webcam mode — opening cv2.VideoCapture(0)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure a camera is connected.")
        sys.exit(1)

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    playback_fps = args.fps if args.fps else video_fps
    frame_delay  = 16          # non-blocking; webcam delivers frames at its own pace
    total_frames = None        # unknown / live
    cue          = None

    print(f"[INFO] Webcam opened  |  {video_fps:.0f} fps requested")
    print(f"[INFO] Rotate 180: {rotate_180}")
    print()
    print("[INFO] Controls: R=Rec  Q=Quit")

else:
    # ── Video file ────────────────────────────────────────────────────────
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    playback_fps = args.fps if args.fps else video_fps
    frame_delay  = max(1, int(1000 / playback_fps))
    cue          = None

    print(f"[INFO] Loaded: {args.video}  |  {total_frames} frames  |  {video_fps:.1f} fps")
    print(f"[INFO] Rotate 180: {rotate_180}")
    print()
    print("[INFO] Controls: Space=Pause  D/→=Step Fwd  A/←=Step Back  R=Rec  Q=Quit")

dt = 1.0 / video_fps   # used only in video mode; live mode gets dt from the camera

# ─────────────────────────────────────────────
# Perception + Visualiser
# ─────────────────────────────────────────────
proc = MaxDivProcessor()
viz  = MaxDivVisualizer(
    out_dir="recordings",
    fps=playback_fps,
    window="Max-Div Validator  (flow | divergence)",
    rotate_180=rotate_180,
)

# ─────────────────────────────────────────────
# S2 — End-of-clip summary state
# ─────────────────────────────────────────────
validate_mode = args.true_azimuth is not None
true_azimuth  = args.true_azimuth
s2_frame_nums = []
s2_est_angles = []


def _angular_diff(est, true):
    """Signed circular difference est - true, result in [-180, 180)."""
    return (est - true + 180.0) % 360.0 - 180.0


def s2_record_frame(res, frame_num, rot):
    """Record azimuth for this frame if the estimate is confident enough."""
    if not res.valid:
        return
    if res.n_vectors < MIN_VECTORS:
        return
    if res.axis_strength < MIN_AXIS_STRENGTH:
        return
    avx, avy = res.arc_vec
    if rot:
        avx, avy = -avx, -avy
    angle_est = math.degrees(math.atan2(avy, avx))
    s2_frame_nums.append(frame_num)
    s2_est_angles.append(angle_est)


def s2_print_summary(clip_label, true_az, frame_nums, est_angles):
    """Print end-of-clip accuracy summary to the console."""
    n = len(est_angles)
    print()
    print("=" * 56)
    print(f"  S2 SUMMARY — {clip_label}")
    print("=" * 56)
    if n == 0:
        print("  No valid frames recorded — cannot compute error.")
        print("=" * 56)
        return
    errors = [abs(_angular_diff(e, true_az)) for e in est_angles]
    mean_err   = sum(errors) / n
    median_err = sorted(errors)[n // 2]
    max_err    = max(errors)
    mean_est   = sum(est_angles) / n
    print(f"  Valid frames recorded : {n}")
    print(f"  True azimuth          : {true_az:.1f} deg")
    print(f"  Mean estimated az     : {mean_est:.1f} deg")
    print(f"  Mean   |error|        : {mean_err:.1f} deg")
    print(f"  Median |error|        : {median_err:.1f} deg")
    print(f"  Max    |error|        : {max_err:.1f} deg")
    print("=" * 56)

    # ── optional matplotlib plot ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clip_stem = clip_label.replace(" ", "_")
        plot_path = f"summary_{clip_stem}.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(
            f"S2 Summary — {clip_label}\n"
            f"True azimuth: {true_az:.1f}°   Mean |error|: {mean_err:.1f}°",
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

        ax2.fill_between(frame_nums, errors, color="tab:red", alpha=0.5,
                         label="|error|")
        ax2.axhline(mean_err, color="darkred", linewidth=1.2, linestyle="--",
                    label=f"Mean ({mean_err:.1f}°)")
        ax2.set_ylabel("|Angular error| (deg)")
        ax2.set_xlabel("Frame number")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"  Plot saved → {plot_path}")
    except ImportError:
        pass   # matplotlib not installed — skip plot


# ─────────────────────────────────────────────
# M3 — Proximity Readout Panel
# ─────────────────────────────────────────────
def draw_proximity_panel(canvas, res, panel_x, panel_y, panel_w, panel_h):
    """
    Draws a proximity bar and TTC readout on the canvas.

    The bar fills from left to right as the camera approaches.
    Green -> Yellow -> Red as TTC decreases.
    """
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (70, 70, 70), 1)

    cv2.putText(canvas, "PROXIMITY",
                (panel_x + 8, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    tau = res.tau_s if res else None
    bar_x      = panel_x + 8
    bar_y      = panel_y + 30
    bar_w      = panel_w - 16
    bar_h_size = 22

    cv2.rectangle(canvas,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h_size),
                  (50, 50, 50), -1)
    cv2.rectangle(canvas,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h_size),
                  (100, 100, 100), 1)

    if tau is not None and tau > 0:
        proportion = max(0.0, min(1.0, 1.0 - (tau / TAU_MAX_DISPLAY)))
        fill_w = int(bar_w * proportion)

        if tau > TAU_WARN * 2:
            bar_col = (0, 200, 80)
        elif tau > TAU_WARN:
            bar_col = (0, 200, 255)
        else:
            bar_col = (0, 60, 255)

        if fill_w > 0:
            cv2.rectangle(canvas,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h_size),
                          bar_col, -1)

        tau_str = f"TTC: {tau:.2f} s"
        cv2.putText(canvas, tau_str,
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        d_str = f"D_raw: {res.divmax_raw:.5f}" if res and math.isfinite(res.divmax_raw) else "D_raw: --"
        cv2.putText(canvas, d_str,
                    (panel_x + 8, panel_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        cv2.putText(canvas, f"vectors: {res.n_vectors if res else 0}",
                    (panel_x + 8, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    else:
        cv2.putText(canvas, "TTC: --",
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)


# ─────────────────────────────────────────────
# M4 — Direction-to-Frontal Panel  (M5 confidence flag built in)
# ─────────────────────────────────────────────
def draw_direction_panel(canvas, res, panel_x, panel_y, panel_w, panel_h):
    """
    Draws a plain-language direction-to-frontal indicator.

    Three possible states:
      LOW CONFIDENCE  — not enough vectors or axis too weak to trust  (M5)
      NEAR FRONTAL    — axis is weak because we are already near frontal
      TILT THIS WAY   — arrow showing which direction to tilt
    """
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (70, 70, 70), 1)

    cv2.putText(canvas, "DIRECTION TO FRONTAL",
                (panel_x + 8, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cx = panel_x + panel_w // 2
    cy = panel_y + panel_h // 2 + 15

    if res is None or not res.valid:
        cv2.putText(canvas, "PRIMING...",
                    (panel_x + 8, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        return

    low_vectors  = res.n_vectors < MIN_VECTORS
    near_frontal = res.axis_strength < MIN_AXIS_STRENGTH

    if low_vectors:
        # ── LOW CONFIDENCE  (M5) ─────────────────────────────────────────
        cv2.putText(canvas, "LOW",
                    (panel_x + 8, panel_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(canvas, "CONFIDENCE",
                    (panel_x + 8, panel_y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        cv2.putText(canvas, f"only {res.n_vectors} vectors",
                    (panel_x + 8, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
        cv2.circle(canvas, (cx, cy + 20), 28, (0, 130, 200), 2)
        cv2.putText(canvas, "?",
                    (cx - 9, cy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 130, 200), 2)

    elif near_frontal:
        # ── NEAR FRONTAL ─────────────────────────────────────────────────
        cv2.putText(canvas, "NEAR",
                    (panel_x + 8, panel_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 120), 2)
        cv2.putText(canvas, "FRONTAL",
                    (panel_x + 8, panel_y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 120), 2)
        cv2.putText(canvas, f"|s| = {res.axis_strength:.4f}",
                    (panel_x + 8, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
        cv2.circle(canvas, (cx, cy + 20), 28, (0, 200, 80), 2)
        cv2.putText(canvas, "OK",
                    (cx - 14, cy + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)

    else:
        # ── TILT THIS WAY ────────────────────────────────────────────────
        cv2.putText(canvas, "TILT",
                    (panel_x + 8, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(canvas, "THIS WAY",
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(canvas, f"|s| = {res.axis_strength:.4f}",
                    (panel_x + 8, panel_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        avx, avy = res.arc_vec
        if rotate_180:
            avx, avy = -avx, -avy

        arrow_len = 38
        tip  = (int(cx + avx * arrow_len), int(cy + 20 + avy * arrow_len))
        tail = (int(cx - avx * arrow_len), int(cy + 20 - avy * arrow_len))
        cv2.arrowedLine(canvas, tail, tip, (0, 200, 255), 3, tipLength=0.3)

        angle_deg = math.degrees(math.atan2(avy, avx))
        cv2.putText(canvas, f"{angle_deg:.0f} deg",
                    (panel_x + 8, panel_y + panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


# ─────────────────────────────────────────────
# Playback state (video mode)
# ─────────────────────────────────────────────
paused    = False
frame_idx = 0

if args.live:
    # RealSenseCue warms up 20 frames in start(); fetch one to prime
    cue.get_tau(want_render=True)
    current_frame = None
    print("[INFO] Live feed running...")
else:
    # Video file or webcam — read first frame and prime the processor
    ret, current_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read the first frame.")
        cap.release()
        sys.exit(1)
    proc.process(current_frame, dt=0.0, want_render=False)
    print("[INFO] Primed. Starting...")


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:

    # ── Get the next frame + result ───────────────────────────────────────
    if args.live:
        # C1 — pull from RealSense; dt is measured inside RealSenseCue
        tau = cue.get_tau(want_render=True)
        res = cue.get_last()
        if res is None:
            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord('q'), 27):
                break
            continue
        frame_idx += 1

    elif args.webcam:
        # Webcam — grab next frame from cv2 and process it
        ret, current_frame = cap.read()
        if not ret:
            print("[WARN] Webcam frame grab failed — retrying.")
            continue
        frame_idx += 1
        res = proc.process(current_frame, dt=dt, want_render=True)

    else:
        # Video mode — process the current frame
        res = proc.process(current_frame, dt=dt, want_render=True)

    # ── S2 — record azimuth this frame if in validate mode ───────────────
    if validate_mode and res is not None:
        s2_record_frame(res, frame_idx, rotate_180)

    # ── Base visualiser (flow + heatmap side by side) ─────────────────────
    keep_going = viz.show(res, tau=res.tau_s, depth=None, event="")
    if not keep_going:
        break

    # ── Build M3 + M4 panel ───────────────────────────────────────────────
    if res.render is not None:
        h_frame, w_frame = res.render.frame_bgr.shape[:2]
        composite_w = w_frame * 2
        panel_h     = 160
        panel_w     = composite_w // 2

        panel = np.zeros((panel_h, composite_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        draw_proximity_panel(panel, res,
                             panel_x=0, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h)
        draw_direction_panel(panel, res,
                             panel_x=panel_w, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h)

        # Mode label
        if args.live:
            cv2.putText(panel, "LIVE  D415", (8, panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
        elif args.webcam:
            cv2.putText(panel, "WEBCAM", (8, panel_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 80), 1)

        cv2.imshow("Max-Div Validator  (proximity | direction)", panel)

    # ── Key handling ──────────────────────────────────────────────────────
    key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF

    if key in (ord('q'), 27):
        break

    elif key == ord('r'):
        viz.toggle_recording()

    # Video-only controls (not applicable to webcam or live)
    elif not args.live and not args.webcam:

        if key == ord(' '):
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx + 1}")

        elif key in (ord('d'), 83):
            if frame_idx < total_frames - 1:
                frame_idx += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, current_frame = cap.read()
                if not ret:
                    frame_idx -= 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    cap.read()
                proc.reset()
                paused = True
            else:
                print("[INFO] Already at the last frame.")

        elif key in (ord('a'), 81):
            if frame_idx > 0:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, current_frame = cap.read()
                if not ret:
                    print("[WARN] Could not read frame.")
                proc.reset()
                paused = True
            else:
                print("[INFO] Already at the first frame.")

        elif not paused:
            ret, next_frame = cap.read()
            if ret:
                frame_idx += 1
                current_frame = next_frame
            else:
                # End of video — trigger S2 summary then loop
                if validate_mode:
                    label = os.path.splitext(os.path.basename(args.video))[0]
                    s2_print_summary(label, true_azimuth, s2_frame_nums, s2_est_angles)
                print("[INFO] End of video — looping.")
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, current_frame = cap.read()
                proc.reset()
                s2_frame_nums.clear()
                s2_est_angles.clear()
                if not ret:
                    print("[ERROR] Could not loop.")
                    break

# ─────────────────────────────────────────────
# S2 summary on quit
# ─────────────────────────────────────────────
if validate_mode and s2_est_angles:
    label = "live" if args.live else os.path.splitext(os.path.basename(args.video))[0]
    s2_print_summary(label, true_azimuth, s2_frame_nums, s2_est_angles)

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
if args.live:
    cue.stop()
    print("[INFO] RealSense stopped.")
else:
    cap.release()

viz.close()
cv2.destroyAllWindows()
print("[INFO] Done.")