#!/usr/bin/env python3
"""
m3m4.py  —  Milestones 1 + 2 + 3 + 4

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

CONTROLS
--------
  Space       — pause / resume
  D or →      — step forward one frame (when paused)
  A or ←      — step backward one frame (when paused)
  R           — toggle recording
  Q or Escape — quit

USAGE
-----
  python m3m4.py --video path/to/clip.mp4 --no-rotate
  python m3m4.py --video clip.mp4
"""

import cv2
import argparse
import os
import sys
import math
import numpy as np

from maxdiv_perception import MaxDivProcessor
from maxdiv_viz import MaxDivVisualizer

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
    description="Max-Div Validator — M3 Proximity + M4 Direction-to-Frontal"
)
parser.add_argument("--video", type=str, required=True,
                    help="Path to the input video file")
parser.add_argument("--fps", type=float, default=None,
                    help="Override playback FPS")
parser.add_argument("--no-rotate", action="store_true",
                    help="Do not rotate 180 deg (use for normal hand-held footage)")
args = parser.parse_args()

# ─────────────────────────────────────────────
# Load Video
# ─────────────────────────────────────────────
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
rotate_180   = not args.no_rotate
dt           = 1.0 / video_fps

print(f"[INFO] Loaded: {args.video}  |  {total_frames} frames  |  {video_fps:.1f} fps")
print(f"[INFO] Rotate 180: {rotate_180}")
print()
print("[INFO] Controls: Space=Pause  D/→=Step Fwd  A/←=Step Back  R=Rec  Q=Quit")

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
# M3 — Proximity Readout Panel
# ─────────────────────────────────────────────
def draw_proximity_panel(canvas, res, panel_x, panel_y, panel_w, panel_h):
    """
    Draws a proximity bar and TTC readout on the canvas.

    The bar fills from left to right as the camera approaches.
    Green -> Yellow -> Red as TTC decreases.

    panel_x, panel_y : top-left corner of the panel on the canvas
    panel_w, panel_h : size of the panel
    """
    # Background
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (70, 70, 70), 1)

    # Title
    cv2.putText(canvas, "PROXIMITY",
                (panel_x + 8, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    tau = res.tau_s if res else None
    bar_x      = panel_x + 8
    bar_y      = panel_y + 30
    bar_w      = panel_w - 16
    bar_h_size = 22

    # Empty bar background
    cv2.rectangle(canvas,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h_size),
                  (50, 50, 50), -1)
    cv2.rectangle(canvas,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h_size),
                  (100, 100, 100), 1)

    if tau is not None and tau > 0:
        # Fill proportion — clamp between 0 and 1
        # Bar is FULL when very close (tau small), EMPTY when far (tau large)
        proportion = max(0.0, min(1.0, 1.0 - (tau / TAU_MAX_DISPLAY)))
        fill_w = int(bar_w * proportion)

        # Colour: green when far, yellow when getting close, red when very close
        if tau > TAU_WARN * 2:
            bar_col = (0, 200, 80)    # green
        elif tau > TAU_WARN:
            bar_col = (0, 200, 255)   # yellow
        else:
            bar_col = (0, 60, 255)    # red

        if fill_w > 0:
            cv2.rectangle(canvas,
                          (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h_size),
                          bar_col, -1)

        # TTC text
        tau_str = f"TTC: {tau:.2f} s"
        cv2.putText(canvas, tau_str,
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # D_raw
        d_str = f"D_raw: {res.divmax_raw:.5f}" if res and math.isfinite(res.divmax_raw) else "D_raw: --"
        cv2.putText(canvas, d_str,
                    (panel_x + 8, panel_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # nvec
        cv2.putText(canvas, f"vectors: {res.n_vectors if res else 0}",
                    (panel_x + 8, panel_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    else:
        cv2.putText(canvas, "TTC: --",
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)


# ─────────────────────────────────────────────
# M4 — Direction-to-Frontal Panel
# ─────────────────────────────────────────────
def draw_direction_panel(canvas, res, panel_x, panel_y, panel_w, panel_h):
    """
    Draws a plain-language direction-to-frontal indicator.

    Three possible states:
      LOW CONFIDENCE  — not enough vectors or axis too weak to trust
      NEAR FRONTAL    — axis is weak because we are already near frontal
      TILT THIS WAY   — arrow showing which direction to tilt

    The arrow is drawn as a large visual indicator pointing in the
    direction derived from res.arc_vec.
    """
    # Background
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 30), -1)
    cv2.rectangle(canvas,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (70, 70, 70), 1)

    # Title
    cv2.putText(canvas, "DIRECTION TO FRONTAL",
                (panel_x + 8, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cx = panel_x + panel_w // 2
    cy = panel_y + panel_h // 2 + 15

    if res is None or not res.valid:
        # No valid result yet — priming
        cv2.putText(canvas, "PRIMING...",
                    (panel_x + 8, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)
        return

    low_vectors = res.n_vectors < MIN_VECTORS
    near_frontal = res.axis_strength < MIN_AXIS_STRENGTH

    if low_vectors:
        # ── LOW CONFIDENCE ──────────────────────────────────────────────
        cv2.putText(canvas, "LOW",
                    (panel_x + 8, panel_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(canvas, "CONFIDENCE",
                    (panel_x + 8, panel_y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        cv2.putText(canvas, f"only {res.n_vectors} vectors",
                    (panel_x + 8, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        # Draw a question mark circle
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

        # Draw a tick circle
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

        # Draw the direction arrow
        # arc_vec is the E_m direction in perception coordinates
        avx, avy = res.arc_vec
        if rotate_180:
            avx, avy = -avx, -avy

        arrow_len = 38
        tip  = (int(cx + avx * arrow_len), int(cy + 20 + avy * arrow_len))
        tail = (int(cx - avx * arrow_len), int(cy + 20 - avy * arrow_len))

        cv2.arrowedLine(canvas, tail, tip, (0, 200, 255), 3, tipLength=0.3)

        # Angle label
        angle_deg = math.degrees(math.atan2(avy, avx))
        cv2.putText(canvas, f"{angle_deg:.0f} deg",
                    (panel_x + 8, panel_y + panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)


# ─────────────────────────────────────────────
# Playback state
# ─────────────────────────────────────────────
paused    = False
frame_idx = 0

ret, current_frame = cap.read()
if not ret:
    print("[ERROR] Could not read the first frame.")
    cap.release()
    sys.exit(1)

# Prime the processor
proc.process(current_frame, dt=0.0, want_render=False)
print("[INFO] Primed. Starting playback...")

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:

    # ── Perception ────────────────────────────────────────────────────────
    res = proc.process(current_frame, dt=dt, want_render=True)

    # ── Base visualiser (flow + heatmap side by side) ─────────────────────
    keep_going = viz.show(res, tau=res.tau_s, depth=None, event="")
    if not keep_going:
        break

    # ── Build M3 + M4 panel below the composite ───────────────────────────
    # Get the composite size so we can size our panel to match
    # We create a separate window for M3/M4 so it does not interfere
    # with the viz window's recording feature.
    if res.render is not None:
        h_frame, w_frame = res.render.frame_bgr.shape[:2]
        composite_w = w_frame * 2   # flow | divergence side by side
        panel_h     = 160
        panel_w     = composite_w // 2

        # Create the M3+M4 panel
        panel = np.zeros((panel_h, composite_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        # M3 — left half
        draw_proximity_panel(panel, res,
                             panel_x=0, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h)

        # M4 — right half
        draw_direction_panel(panel, res,
                             panel_x=panel_w, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h)

        cv2.imshow("Max-Div Validator  (proximity | direction)", panel)

    # ── Key handling ──────────────────────────────────────────────────────
    key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF

    if key in (ord('q'), 27):
        break

    elif key == ord(' '):
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

    elif key == ord('r'):
        viz.toggle_recording()

    elif not paused:
        ret, next_frame = cap.read()
        if ret:
            frame_idx += 1
            current_frame = next_frame
        else:
            print("[INFO] End of video — looping.")
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, current_frame = cap.read()
            proc.reset()
            if not ret:
                print("[ERROR] Could not loop.")
                break

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
viz.close()
cv2.destroyAllWindows()
print("[INFO] Done.")