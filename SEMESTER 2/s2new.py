
#!/usr/bin/env python3
"""

CONTROLS
--------
  Space       — pause / resume
  D or →      — step forward one frame (when paused)
  A or ←      — step backward one frame (when paused)
  F           — toggle flow arrows on/off
  H           — toggle divergence heatmap on/off
  X           — toggle axis/arc arrow on/off
  R           — toggle recording
  Q or Escape — quit

USAGE
-----
  python m3m4.py --video path/to/clip.mp4 --no-rotate
  python m3m4.py --video clip.mp4

S2 — END-OF-CLIP VALIDATION SUMMARY
-------------------------------------
  Pass --true-azimuth <degrees> to enable validation mode.
  At end of clip, prints mean angular error and saves a PNG plot of
  estimated vs true azimuth over the clip frames.

  Example:
    python m3m4.py --video clip.mp4 --no-rotate --true-azimuth 45
    python m3m4.py --video synthetic_0deg.mp4 --no-rotate --true-azimuth 0
"""

import cv2
import argparse
import os
import sys
import math
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — safe on Windows/Linux
    import matplotlib.pyplot as plt
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False
    print("[WARN] matplotlib not found — S2 plot will not be saved (pip install matplotlib)")

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from maxdiv_perception import MaxDivProcessor

# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────
MIN_VECTORS       = 20
MIN_AXIS_STRENGTH = 0.005
TAU_MAX_DISPLAY   = 10.0
TAU_WARN          = 2.0
ARROW_SCALE       = 8      # multiply flow vectors by this so small movements are visible

# colours (BGR)
FLOW_COL   = (0, 255, 0)
DOT_COL    = (0, 180, 0)
CENTER_COL = (255, 0, 0)
MAXDIV_COL = (255, 255, 255)
ARC_STRONG = (0, 200, 255)
ARC_WEAK   = (120, 120, 120)
HUD_COL    = (255, 255, 255)
HUD_DIM    = (170, 170, 170)

# ─────────────────────────────────────────────
# Command Line Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Max-Div Validator — M1-M4 with toggles and always-visible grid"
)
parser.add_argument("--video", type=str, required=True,
                    help="Path to the input video file")
parser.add_argument("--fps", type=float, default=None,
                    help="Override playback FPS")
parser.add_argument("--no-rotate", action="store_true",
                    help="Do not rotate 180 deg (use for normal hand-held footage)")
# S2: end-of-clip validation
parser.add_argument("--true-azimuth", type=float, default=None,
                    metavar="DEGREES",
                    help=(
                        "S2: known ground-truth tilt azimuth (degrees). "
                        "When supplied, tracks estimated azimuth per frame, "
                        "prints mean angular error at clip end, and saves a plot PNG."
                    ))
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

# ── S2 state ──────────────────────────────────
validate_mode  = args.true_azimuth is not None
true_azimuth   = args.true_azimuth if validate_mode else 0.0
s2_frame_nums  = []   # frame index of each recorded sample
s2_est_angles  = []   # estimated azimuth (deg) at that frame

if validate_mode:
    print(f"[S2] Validation mode ON — true azimuth: {true_azimuth:.1f} deg")
    print(f"[S2] Estimated vs true azimuth will be plotted at end of clip.")
# ──────────────────────────────────────────────

print()
print("[INFO] Controls:")
print("         Space   — pause / resume")
print("         D / →   — step forward one frame")
print("         A / ←   — step backward one frame")
print("         F       — toggle flow arrows")
print("         H       — toggle divergence heatmap")
print("         X       — toggle axis/arc arrow")
print("         R       — toggle recording")
print("         Q / Esc — quit")

# ─────────────────────────────────────────────
# Toggle state
# ─────────────────────────────────────────────
show_flow    = True
show_heatmap = True
show_arc     = True

# ─────────────────────────────────────────────
# Recording state
# ─────────────────────────────────────────────
recording  = False
rec_writer = None

def start_recording(w, h, fps):
    os.makedirs("recordings", exist_ok=True)
    path   = f"recordings/recording_{int(cv2.getTickCount())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    print(f"[INFO] Recording started: {path}")
    return writer

def stop_recording(writer):
    if writer:
        writer.release()
    print("[INFO] Recording stopped.")
    return None

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def rot180_pt(pt, w, h):
    return (w - 1 - int(pt[0]), h - 1 - int(pt[1]))

def rot180(img):
    return cv2.flip(img, -1)

# ─────────────────────────────────────────────
# Draw flow view
# ─────────────────────────────────────────────
def draw_flow_view(render, res, show_flow_arrows, show_arc_arrow, rot180_flag):
    img  = render.frame_bgr.copy()
    h, w = img.shape[:2]
    if rot180_flag:
        img = rot180(img)

    gp   = render.grid_points
    u    = render.u
    v    = render.v
    mask = render.valid_lk

    for i in range(len(gp)):
        x1 = int(gp[i][0][0])
        y1 = int(gp[i][0][1])
        if rot180_flag:
            x1, y1 = rot180_pt((x1, y1), w, h)

        # Always draw dot at every grid point
        cv2.circle(img, (x1, y1), 2, DOT_COL, -1)

        # Draw arrow for every valid tracked point — scaled up for visibility
        if show_flow_arrows and mask[i]:
            dx = u[i] * ARROW_SCALE
            dy = v[i] * ARROW_SCALE
            if rot180_flag:
                dx = -dx
                dy = -dy
            x2 = int(x1 + dx)
            y2 = int(y1 + dy)
            cv2.arrowedLine(img, (x1, y1), (x2, y2),
                            FLOW_COL, 2, tipLength=0.3)

    # Center cross
    center = (w // 2, h // 2)
    cv2.drawMarker(img, center, CENTER_COL,
                   markerType=cv2.MARKER_CROSS, markerSize=45, thickness=2)

    # Max-div cross
    if res.max_loc is not None:
        loc = rot180_pt(res.max_loc, w, h) if rot180_flag else tuple(int(v) for v in res.max_loc)
        cv2.drawMarker(img, loc, MAXDIV_COL,
                       markerType=cv2.MARKER_CROSS, markerSize=45, thickness=3)

        # Arc arrow
        if show_arc_arrow and res.axis_strength > 0:
            L    = 90
            avx, avy = res.arc_vec
            if rot180_flag:
                avx, avy = -avx, -avy
            tip  = (int(loc[0] + L * avx), int(loc[1] + L * avy))
            tail = (int(loc[0] - L * avx), int(loc[1] - L * avy))
            weak = res.axis_strength < MIN_AXIS_STRENGTH
            acol = ARC_WEAK if weak else ARC_STRONG
            cv2.line(img, tail, loc, acol, 1)
            cv2.arrowedLine(img, loc, tip, acol, 2, tipLength=0.25)
            cv2.putText(img,
                        f"E_m arc  |s|={res.axis_strength:.4f}"
                        + ("  HELD (near-frontal)" if weak else ""),
                        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, acol, 1)

    # HUD
    tau_s = f"{res.tau_s:.2f} s" if res.tau_s is not None else "--"
    d_raw = f"{res.divmax_raw:.5f}" if math.isfinite(res.divmax_raw) else "--"
    cv2.putText(img, f"TTC(flow)  {tau_s}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COL, 2)
    cv2.putText(img, f"D_raw {d_raw}   nvec {res.n_vectors}",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_DIM, 1)

    # Toggle status
    status = (f"F:flow={'ON' if show_flow_arrows else 'OFF'}  "
              f"H:heatmap={'ON' if show_heatmap else 'OFF'}  "
              f"X:arc={'ON' if show_arc_arrow else 'OFF'}")
    cv2.putText(img, status,
                (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    return img

# ─────────────────────────────────────────────
# Draw heatmap view
# ─────────────────────────────────────────────
def draw_heatmap_view(render, res, show_arc_arrow, rot180_flag):
    frame  = render.frame_bgr
    h, w   = frame.shape[:2]
    skip   = render.skip
    gw, gh = render.grid_w, render.grid_h

    positive     = cv2.GaussianBlur(
        np.maximum(render.divergence, 0).astype(np.float32), (3, 3), 0)
    grid_span_w  = (gw - 1) * skip + 1
    grid_span_h  = (gh - 1) * skip + 1
    heatmap_span = cv2.resize(positive, (grid_span_w, grid_span_h),
                              interpolation=cv2.INTER_CUBIC)
    heatmap_large = np.zeros((h, w), dtype=heatmap_span.dtype)
    ph = min(grid_span_h, h)
    pw = min(grid_span_w, w)
    heatmap_large[0:ph, 0:pw] = heatmap_span[0:ph, 0:pw]
    heatmap_u8    = np.uint8(cv2.normalize(heatmap_large, None, 0, 255, cv2.NORM_MINMAX))
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    if rot180_flag:
        out = rot180(out)

    center = (w // 2, h // 2)
    cv2.drawMarker(out, center, CENTER_COL,
                   markerType=cv2.MARKER_CROSS, markerSize=45, thickness=2)
    if res.max_loc is not None:
        loc = rot180_pt(res.max_loc, w, h) if rot180_flag else tuple(int(v) for v in res.max_loc)
        cv2.drawMarker(out, loc, MAXDIV_COL,
                       markerType=cv2.MARKER_CROSS, markerSize=45, thickness=3)
        if show_arc_arrow and res.axis_strength > 0:
            L    = 90
            avx, avy = res.arc_vec
            if rot180_flag:
                avx, avy = -avx, -avy
            tip  = (int(loc[0] + L * avx), int(loc[1] + L * avy))
            tail = (int(loc[0] - L * avx), int(loc[1] - L * avy))
            weak = res.axis_strength < MIN_AXIS_STRENGTH
            acol = ARC_WEAK if weak else ARC_STRONG
            cv2.line(out, tail, loc, acol, 1)
            cv2.arrowedLine(out, loc, tip, acol, 2, tipLength=0.25)
    return out

# ─────────────────────────────────────────────
# M3 — Proximity Panel
# ─────────────────────────────────────────────
def draw_proximity_panel(canvas, res, panel_x, panel_y, panel_w, panel_h):
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

    tau   = res.tau_s if res else None
    bar_x = panel_x + 8
    bar_y = panel_y + 30
    bar_w = panel_w - 16
    bar_hs = 22

    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_hs), (50, 50, 50), -1)
    cv2.rectangle(canvas, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_hs), (100, 100, 100), 1)

    if tau is not None and tau > 0:
        proportion = max(0.0, min(1.0, 1.0 - (tau / TAU_MAX_DISPLAY)))
        fill_w     = int(bar_w * proportion)
        if tau > TAU_WARN * 2:
            bar_col = (0, 200, 80)
        elif tau > TAU_WARN:
            bar_col = (0, 200, 255)
        else:
            bar_col = (0, 60, 255)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_hs), bar_col, -1)
        cv2.putText(canvas, f"TTC: {tau:.2f} s",
                    (panel_x + 8, panel_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        d_str = (f"D_raw: {res.divmax_raw:.5f}"
                 if res and math.isfinite(res.divmax_raw) else "D_raw: --")
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
# M4 — Direction-to-Frontal Panel
# ─────────────────────────────────────────────
def draw_direction_panel(canvas, res, panel_x, panel_y, panel_w, panel_h, rot):
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
        if rot:
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
# S2 — End-of-Clip Summary helpers
# ─────────────────────────────────────────────
def _angular_diff(est, true):
    """Signed circular difference est - true, in [-180, 180)."""
    return (est - true + 180.0) % 360.0 - 180.0


def s2_record_frame(res, frame_num, rot):
    """Record one frame's estimated azimuth if the result is usable.

    Returns the estimated azimuth (deg) if recorded, else None.
    """
    if not res.valid:
        return None
    if res.n_vectors < MIN_VECTORS:
        return None
    if res.axis_strength < MIN_AXIS_STRENGTH:
        return None
    avx, avy = res.arc_vec
    if rot:
        avx, avy = -avx, -avy
    angle_est = math.degrees(math.atan2(avy, avx))
    s2_frame_nums.append(frame_num)
    s2_est_angles.append(angle_est)
    return angle_est


def s2_print_summary(clip_path, true_az, frame_nums, est_angles):
    """Print the end-of-clip accuracy report to the console."""
    n = len(est_angles)
    print()
    print("=" * 60)
    print("S2  END-OF-CLIP SUMMARY")
    print("=" * 60)
    print(f"  clip          : {clip_path}")
    print(f"  true azimuth  : {true_az:.1f} deg")
    print(f"  valid frames  : {n}")
    if n == 0:
        print("  (no valid frames — check MIN_VECTORS / MIN_AXIS_STRENGTH thresholds)")
        print("=" * 60)
        return None, None

    errors = [abs(_angular_diff(e, true_az)) for e in est_angles]
    mean_err  = sum(errors) / n
    max_err   = max(errors)
    median_err = sorted(errors)[n // 2]
    mean_est  = sum(est_angles) / n

    print(f"  mean est. az  : {mean_est:.1f} deg")
    print(f"  mean |error|  : {mean_err:.1f} deg")
    print(f"  median |error|: {median_err:.1f} deg")
    print(f"  max |error|   : {max_err:.1f} deg")
    print("=" * 60)
    return mean_err, errors


def s2_save_excel(clip_path, true_az, frame_nums, est_angles, errors, mean_err):
    """Save per-frame results and summary statistics to an Excel workbook."""
    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]
    xlsx_path = f"summary_{clip_stem}.xlsx"

    wb = openpyxl.Workbook()

    # ── Sheet 1: Per-frame data ──────────────────────────────────────────
    ws_data = wb.active
    ws_data.title = "Frame Data"

    header_fill = PatternFill("solid", fgColor="002855")   # Swinburne navy
    header_font = Font(bold=True, color="FFFFFF")
    centre      = Alignment(horizontal="center")
    thin        = Side(style="thin", color="CCCCCC")
    border      = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ["Frame", "Est. Azimuth (deg)", f"True Azimuth (deg)", "|Error| (deg)"]
    for col, h in enumerate(headers, start=1):
        cell = ws_data.cell(row=1, column=col, value=h)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = centre
        cell.border    = border

    for i, (fn, ea, err) in enumerate(zip(frame_nums, est_angles, errors), start=2):
        row_data = [fn, round(ea, 3), round(true_az, 3), round(err, 3)]
        for col, val in enumerate(row_data, start=1):
            cell = ws_data.cell(row=i, column=col, value=val)
            cell.alignment = centre
            cell.border    = border
            # colour-code error column: green < 5°, orange < 15°, red ≥ 15°
            if col == 4:
                if err < 5:
                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                elif err < 15:
                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                else:
                    cell.fill = PatternFill("solid", fgColor="FFC7CE")

    ws_data.column_dimensions["A"].width = 10
    ws_data.column_dimensions["B"].width = 22
    ws_data.column_dimensions["C"].width = 22
    ws_data.column_dimensions["D"].width = 20

    # ── Sheet 2: Summary statistics ──────────────────────────────────────
    ws_sum = wb.create_sheet("Summary")

    n      = len(errors)
    med    = sorted(errors)[n // 2]
    max_e  = max(errors)
    mean_e = sum(errors) / n

    ws_sum["A1"] = "S2 End-of-Clip Summary"
    ws_sum["A1"].font = Font(bold=True, size=13, color="002855")
    ws_sum.merge_cells("A1:B1")

    stats = [
        ("Clip",              clip_path),
        ("True Azimuth (°)",  round(true_az, 1)),
        ("Valid Frames",      n),
        ("Mean Est. Az. (°)", round(sum(est_angles) / n, 1)),
        ("Mean |Error| (°)",  round(mean_e, 1)),
        ("Median |Error| (°)", round(med, 1)),
        ("Max |Error| (°)",   round(max_e, 1)),
    ]

    label_font  = Font(bold=True)
    label_fill  = PatternFill("solid", fgColor="EBF0F8")

    for row_idx, (label, value) in enumerate(stats, start=3):
        lc = ws_sum.cell(row=row_idx, column=1, value=label)
        vc = ws_sum.cell(row=row_idx, column=2, value=value)
        lc.font      = label_font
        lc.fill      = label_fill
        lc.border    = border
        vc.border    = border

    ws_sum.column_dimensions["A"].width = 26
    ws_sum.column_dimensions["B"].width = 30

    wb.save(xlsx_path)
    print(f"[S2] Excel saved  → {xlsx_path}")


def s2_save_plot(clip_path, true_az, frame_nums, est_angles, errors):
    """Save a matplotlib figure: estimated vs true azimuth + error over frames."""
    if not _MATPLOTLIB_OK:
        print("[S2] matplotlib not available — skipping plot.")
        return

    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]
    plot_path = f"summary_{clip_stem}.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"S2 End-of-Clip Summary — {clip_stem}\n"
        f"True azimuth: {true_az:.1f}°   Mean |error|: {sum(errors)/len(errors):.1f}°",
        fontsize=12
    )

    # Top panel: estimated vs true azimuth
    ax1.plot(frame_nums, est_angles, color="tab:blue", linewidth=1.2, label="Estimated azimuth")
    ax1.axhline(true_az, color="tab:orange", linewidth=1.5, linestyle="--", label=f"True azimuth ({true_az:.1f}°)")
    ax1.set_ylabel("Azimuth (deg)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-190, 190)

    # Bottom panel: absolute angular error per frame
    ax2.fill_between(frame_nums, errors, color="tab:red", alpha=0.5, label="|error|")
    ax2.axhline(sum(errors) / len(errors), color="darkred", linewidth=1.2,
                linestyle="--", label=f"Mean |error| ({sum(errors)/len(errors):.1f}°)")
    ax2.set_ylabel("|Angular error| (deg)")
    ax2.set_xlabel("Frame number")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[S2] Plot saved → {plot_path}")


# ─────────────────────────────────────────────
# Playback state
# ─────────────────────────────────────────────
proc      = MaxDivProcessor()
paused    = False
frame_idx = 0

ret, current_frame = cap.read()
if not ret:
    print("[ERROR] Could not read the first frame.")
    cap.release()
    sys.exit(1)

proc.process(current_frame, dt=0.0, want_render=False)
print("[INFO] Primed. Starting playback...")

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:

    res = proc.process(current_frame, dt=dt, want_render=True)

    # ── S2: record estimated azimuth for this frame ──────────────────
    if validate_mode:
        s2_record_frame(res, frame_idx, rotate_180)
    # ─────────────────────────────────────────────────────────────────

    if res.render is not None:
        h_frame, w_frame = res.render.frame_bgr.shape[:2]

        flow_view = draw_flow_view(res.render, res,
                                   show_flow, show_arc, rotate_180)

        if show_heatmap:
            heat_view = draw_heatmap_view(res.render, res,
                                          show_arc, rotate_180)
        else:
            heat_view = res.render.frame_bgr.copy()
            if rotate_180:
                heat_view = rot180(heat_view)
            center = (w_frame // 2, h_frame // 2)
            cv2.drawMarker(heat_view, center, CENTER_COL,
                           markerType=cv2.MARKER_CROSS, markerSize=45, thickness=2)

        composite = np.hstack([flow_view, heat_view])
        cv2.imshow("Max-Div Validator  (flow | divergence)", composite)

        composite_w = w_frame * 2
        panel_h     = 160
        panel_w     = composite_w // 2
        panel       = np.zeros((panel_h, composite_w, 3), dtype=np.uint8)
        panel[:]    = (20, 20, 20)

        draw_proximity_panel(panel, res,
                             panel_x=0, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h)
        draw_direction_panel(panel, res,
                             panel_x=panel_w, panel_y=0,
                             panel_w=panel_w, panel_h=panel_h,
                             rot=rotate_180)

        cv2.imshow("Max-Div Validator  (proximity | direction)", panel)

        if recording and rec_writer is not None:
            full = np.vstack([composite, panel])
            rec_writer.write(full)

    key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF

    if key in (ord('q'), 27):
        # ── S2: produce summary on manual quit too ────────────────────
        if validate_mode and s2_est_angles:
            mean_err, errors = s2_print_summary(
                args.video, true_azimuth, s2_frame_nums, s2_est_angles)
            if errors:
                s2_save_plot(args.video, true_azimuth,
                             s2_frame_nums, s2_est_angles, errors)
                s2_save_excel(args.video, true_azimuth,
                              s2_frame_nums, s2_est_angles, errors, mean_err)
        # ─────────────────────────────────────────────────────────────
        break
    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'} at frame {frame_idx + 1}")
    elif key == ord('f'):
        show_flow = not show_flow
        print(f"[INFO] Flow arrows: {'ON' if show_flow else 'OFF'}")
    elif key == ord('h'):
        show_heatmap = not show_heatmap
        print(f"[INFO] Heatmap: {'ON' if show_heatmap else 'OFF'}")
    elif key == ord('x'):
        show_arc = not show_arc
        print(f"[INFO] Arc arrow: {'ON' if show_arc else 'OFF'}")
    elif key == ord('r'):
        if not recording:
            if res.render is not None:
                h_f, w_f = res.render.frame_bgr.shape[:2]
                rec_writer = start_recording(w_f * 2, h_f + 160, playback_fps)
                recording  = True
        else:
            rec_writer = stop_recording(rec_writer)
            recording  = False
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
            # ── S2: clip just finished — produce summary then loop ────
            if validate_mode and s2_est_angles:
                mean_err, errors = s2_print_summary(
                    args.video, true_azimuth, s2_frame_nums, s2_est_angles)
                if errors:
                    s2_save_plot(args.video, true_azimuth,
                                 s2_frame_nums, s2_est_angles, errors)
                    s2_save_excel(args.video, true_azimuth,
                                  s2_frame_nums, s2_est_angles, errors, mean_err)
                # reset accumulators so the next loop is a fresh run
                s2_frame_nums.clear()
                s2_est_angles.clear()
            # ─────────────────────────────────────────────────────────
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
if recording:
    rec_writer = stop_recording(rec_writer)
cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")