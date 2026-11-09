#!/usr/bin/env python3
"""
video_player.py

Milestone 1 — Basic Video Player for the Max-Div Alignment Validator.

This is the foundation script that all future sprint work will be built on top of.
It loads and plays a video file in an OpenCV window with full playback controls.

CONTROLS
--------
  Space       — pause / resume
  D or →      — step forward one frame (when paused)
  A or ←      — step backward one frame (when paused)
  Q or Escape — quit

HUD
---
  Top left:    current frame / total frames, playback status, FPS
  Bottom bar:  progress bar showing current position in the video

USAGE
-----
  python video_player.py --video path/to/your/video.mp4
  python video_player.py --video clip.mp4 --fps 30
"""

import cv2
import argparse
import os
import sys

# ─────────────────────────────────────────────
# Command Line Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Max-Div Alignment Validator — Milestone 1 Video Player")
parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
parser.add_argument('--fps', type=float, default=None, help='Override playback FPS (default: use video file FPS)')
args = parser.parse_args()

# ─────────────────────────────────────────────
# Load Video
# ─────────────────────────────────────────────
if not os.path.exists(args.video):
    print(f"[ERROR] Video file not found: {args.video}")
    sys.exit(1)

cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {args.video}")
    sys.exit(1)

# Video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
playback_fps = args.fps if args.fps else video_fps
frame_delay  = max(1, int(1000 / playback_fps))   # ms per frame for cv2.waitKey

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Loaded: {args.video}")
print(f"[INFO] Resolution: {width}x{height}")
print(f"[INFO] Total frames: {total_frames}")
print(f"[INFO] Video FPS: {video_fps:.1f}")
print(f"[INFO] Playback FPS: {playback_fps:.1f}")
print()
print("[INFO] Controls:")
print("         Space     — pause / resume")
print("         D or →    — step forward one frame")
print("         A or ←    — step backward one frame")
print("         Q or Esc  — quit")

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
paused      = False
frame_idx   = 0     # current frame index

# ─────────────────────────────────────────────
# Helper — Draw HUD
# ─────────────────────────────────────────────
def draw_hud(frame, frame_idx, total_frames, paused, playback_fps):
    """Draw frame info and playback status on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    status = "PAUSED" if paused else "PLAYING"
    status_col = (0, 200, 255) if paused else (0, 255, 100)

    cv2.putText(frame, f"Frame: {frame_idx + 1} / {total_frames}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, status,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_col, 2)
    cv2.putText(frame, f"FPS: {playback_fps:.1f}",
                (w - 120, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Controls hint
    hint = "Space=Pause  D/Arrow=Step  Q=Quit" if paused else "Space=Pause  Q=Quit"
    cv2.putText(frame, hint,
                (w - len(hint) * 7 - 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    return frame


def draw_progress_bar(frame, frame_idx, total_frames):
    """Draw a progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h   = 18
    bar_y   = h - bar_h

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Progress fill
    if total_frames > 1:
        progress_w = int(w * (frame_idx / (total_frames - 1)))
    else:
        progress_w = 0
    cv2.rectangle(frame, (0, bar_y + 4), (progress_w, h - 4), (41, 182, 246), -1)

    # Playhead
    cv2.rectangle(frame, (max(0, progress_w - 2), bar_y + 2),
                  (min(w, progress_w + 2), h - 2), (255, 255, 255), -1)

    # Time text
    if total_frames > 1 and frame_idx > 0:
        t_cur = frame_idx / (total_frames - 1)
    else:
        t_cur = 0.0

    def fmt_time(frac, total_f, fps):
        secs = int(total_f * frac / max(fps, 1))
        return f"{secs // 60:02d}:{secs % 60:02d}"

    t_str = fmt_time(t_cur, total_frames, playback_fps)
    t_tot = fmt_time(1.0, total_frames, playback_fps)
    cv2.putText(frame, f"{t_str} / {t_tot}",
                (w // 2 - 30, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return frame


# ─────────────────────────────────────────────
# Read and cache the first frame
# ─────────────────────────────────────────────
ret, current_frame = cap.read()
if not ret:
    print("[ERROR] Could not read the first frame.")
    cap.release()
    sys.exit(1)

window_name = "Max-Div Alignment Validator — Video Player"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, width, height)

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:
    display = current_frame.copy()
    display = draw_hud(display, frame_idx, total_frames, paused, playback_fps)
    display = draw_progress_bar(display, frame_idx, total_frames)
    cv2.imshow(window_name, display)

    # Wait for key — shorter delay when playing, longer when paused
    wait_ms = 30 if paused else frame_delay
    key = cv2.waitKey(wait_ms) & 0xFF

    # ── Q or Escape — quit
    if key in (ord('q'), 27):
        print("[INFO] Quit.")
        break

    # ── Space — toggle pause / resume
    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx + 1}")

    # ── D or right arrow — step forward one frame
    elif key in (ord('d'), 83):  # 83 = right arrow on most systems
        if frame_idx < total_frames - 1:
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, current_frame = cap.read()
            if not ret:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, current_frame = cap.read()
            paused = True  # stepping always pauses
        else:
            print("[INFO] Already at the last frame.")

    # ── A or left arrow — step backward one frame
    elif key in (ord('a'), 81):  # 81 = left arrow on most systems
        if frame_idx > 0:
            frame_idx -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, current_frame = cap.read()
            if not ret:
                print("[WARN] Could not read frame.")
            paused = True  # stepping always pauses
        else:
            print("[INFO] Already at the first frame.")

    # ── Advance to next frame when playing
    elif not paused:
        ret, next_frame = cap.read()
        if ret:
            frame_idx += 1
            current_frame = next_frame
        else:
            # End of video — loop back to start
            print("[INFO] End of video — looping back to start.")
            frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, current_frame = cap.read()
            if not ret:
                print("[ERROR] Could not loop video.")
                break

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")