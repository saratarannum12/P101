#!/usr/bin/env python3


import cv2
import time
import argparse
import os
import sys

from maxdiv_perception import MaxDivProcessor
from maxdiv_viz import MaxDivVisualizer


parser = argparse.ArgumentParser(
    description="Max-Div Alignment Validator — Video Player with Perception Overlay"
)
parser.add_argument("--video", type=str, required=True,
                    help="Path to the input video file")
parser.add_argument("--fps", type=float, default=None,
                    help="Override playback FPS (default: use video file FPS)")
parser.add_argument("--no-rotate", action="store_true",
                    help="Do NOT rotate the display 180 deg (default: rotate, "
                         "because the rig camera is mounted upside-down). "
                         "Use --no-rotate for normal hand-held footage.")
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

print(f"[INFO] Loaded  : {args.video}")
print(f"[INFO] Frames  : {total_frames}")
print(f"[INFO] Video FPS: {video_fps:.1f}   Playback FPS: {playback_fps:.1f}")
print(f"[INFO] Rotate 180: {rotate_180}  (use --no-rotate for hand-held footage)")
print()
print("[INFO] Controls:")
print("         Space   — pause / resume")
print("         D / →   — step forward one frame")
print("         A / ←   — step backward one frame")
print("         R       — toggle recording")
print("         Q / Esc — quit")

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

# dt between frames from the video nominal frame rate
dt = 1.0 / video_fps

# ─────────────────────────────────────────────
# Playback state
# ─────────────────────────────────────────────
paused    = False
frame_idx = 0

# Read and prime the very first frame
ret, current_frame = cap.read()
if not ret:
    print("[ERROR] Could not read the first frame.")
    cap.release()
    sys.exit(1)

# Prime the processor — first call always returns invalid
proc.process(current_frame, dt=0.0, want_render=False)
print("[INFO] Perception primed. Starting playback...")

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:

    # ── Run perception on the current frame ──────────────────────────────
    res = proc.process(current_frame, dt=dt, want_render=True)

    # ── Draw and show via the visualiser ─────────────────────────────────
    # viz.show() returns False when user presses Q/Esc inside its waitKey(1)
    keep_going = viz.show(res, tau=res.tau_s, depth=None, event="")
    if not keep_going:
        break

    # ── Additional key handling with playback-speed delay ─────────────────
    key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF

    if key in (ord('q'), 27):
        break

    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx + 1}")

    elif key in (ord('d'), 83):   # D or right arrow
        if frame_idx < total_frames - 1:
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, current_frame = cap.read()
            if not ret:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                cap.read()
            proc.reset()   # re-prime from new position
            paused = True
        else:
            print("[INFO] Already at the last frame.")

    elif key in (ord('a'), 81):   # A or left arrow
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

    # ── Advance to next frame when playing ───────────────────────────────
    elif not paused:
        ret, next_frame = cap.read()
        if ret:
            frame_idx += 1
            current_frame = next_frame
        else:
            # End of video — loop back to start
            print("[INFO] End of video — looping.")
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, current_frame = cap.read()
            proc.reset()
            if not ret:
                print("[ERROR] Could not loop.")
                break

cap.release()
viz.close()
print("[INFO] Done.")