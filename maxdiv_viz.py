#!/usr/bin/env python3
"""
maxdiv_viz.py

Annotated visualisation + recording for the spherical-flow-divergence cue,
reproducing the Parrot Mambo overlay style on the arm rig.

WHAT IT DRAWS (faithful to mambo_divergence.py)
-----------------------------------------------
  - flow view: green LK flow arrows (tipLength 0.3) on the frame
  - divergence view: positive-divergence field upsampled to the TRUE grid span
    (not stretched to full frame -- that shift-corrects the peak), JET colormap,
    blended 0.6/0.4 over the frame
  - both views: blue center cross, white max-div cross (size 45), and the
    recovered E_m arc drawn as a DIRECTIONAL arrow (cyan when the shear axis is
    strong, grey when weak / held near-frontal)
  - a small HUD: tau (s), depth (m), D_raw, vector count, and the active event

RECORDING (R to toggle, faithful to the Mambo recorder)
-------------------------------------------------------
  trial_NN_HHMMSS_raw.mp4      the plain frames, 1:1 with processed frames
  trial_NN_HHMMSS_overlay.mp4  side-by-side (flow | divergence) composite
Every processed frame is written 1:1 so the recording matches the run.

This module is display/record only. It reads a MaxDivProcessor.RenderBundle and
the scalar Result; it performs NO perception and changes NO calibrated value.
"""

import os
import time
from datetime import datetime

import numpy as np
import cv2


# colours (BGR)
FLOW_COL      = (0, 255, 0)
CENTER_COL    = (255, 0, 0)
MAXDIV_COL    = (255, 255, 255)
ARC_STRONG    = (0, 200, 255)
ARC_WEAK      = (120, 120, 120)
HUD_COL       = (255, 255, 255)
HUD_DIM       = (170, 170, 170)
HUD_WARN      = (0, 165, 255)


def _rot180_pt(pt, w, h):
    """Map a point through a 180 deg image rotation. Self-inverse."""
    return (w - 1 - int(pt[0]), h - 1 - int(pt[1]))


def _rot180(img):
    """180 deg in-plane rotation (flip both axes)."""
    return cv2.flip(img, -1)


# =============================================================================
# DISPLAY ROTATION (presentation only -- NEVER applied to the perception path)
# -----------------------------------------------------------------------------
# The camera is mounted rotated 180 deg about its optical axis, so raw frames
# (and therefore the overlays) appear upside down. This module can present them
# the right way up for viewing and recording.
#
# It is DELIBERATELY confined to display. The perception path is left untouched
# because:
#   * a 180 deg rotation leaves the flow-field invariants unchanged -- both the
#     flow components and the spatial coordinates flip, so divergence and shear
#     magnitude are identical (confirmed on synthetic oblique-plane flow:
#     correlation with tilt stayed positive, flipped or not); and
#   * every empirical calibration on the rig (tilt sign responses, plant
#     Jacobian, probe directions) was measured THROUGH the uncompensated
#     pipeline and therefore already absorbs the mount. Rotating frames in
#     perception would silently invert all of it.
#
# Markers are computed by perception in unrotated coordinates, so their
# positions are mapped through the same rotation before drawing. Annotation then
# happens in upright coordinates, which keeps HUD text readable.
# =============================================================================
def _draw_flow(render, rot180=False):
    """Flow view: green arrows on a copy of the frame.

    When rot180 is set the BASE IMAGE is rotated first and every drawn point is
    mapped through the same rotation, so overlays land on the correct features
    and subsequent text can be drawn upright.
    """
    img = render.frame_bgr.copy()
    h, w = img.shape[:2]
    if rot180:
        img = _rot180(img)
    gp = render.grid_points
    u, v, mask = render.u, render.v, render.valid_lk
    for i in range(len(gp)):
        if not mask[i]:
            continue
        x1 = int(gp[i][0][0]); y1 = int(gp[i][0][1])
        x2 = int(x1 + u[i]);   y2 = int(y1 + v[i])
        if rot180:
            x1, y1 = _rot180_pt((x1, y1), w, h)
            x2, y2 = _rot180_pt((x2, y2), w, h)
        cv2.arrowedLine(img, (x1, y1), (x2, y2), FLOW_COL, 1, tipLength=0.3)
    return img


def _draw_heatmap(render, rot180=False):
    """Divergence view: positive-divergence upsampled to the true grid span,
    pasted at origin (peak-shift-correct), JET, blended over the frame.

    The heatmap is built in perception coordinates and the whole composite is
    rotated at the end, so the field stays registered to the frame.
    """
    frame = render.frame_bgr
    h, w = frame.shape[:2]
    skip = render.skip
    gw, gh = render.grid_w, render.grid_h

    positive = cv2.GaussianBlur(np.maximum(render.divergence, 0).astype(np.float32),
                                (3, 3), 0)

    grid_span_w = (gw - 1) * skip + 1
    grid_span_h = (gh - 1) * skip + 1
    heatmap_span = cv2.resize(positive, (grid_span_w, grid_span_h),
                              interpolation=cv2.INTER_CUBIC)

    heatmap_large = np.zeros((h, w), dtype=heatmap_span.dtype)
    ph, pw = min(grid_span_h, h), min(grid_span_w, w)
    heatmap_large[0:ph, 0:pw] = heatmap_span[0:ph, 0:pw]

    heatmap_uint8 = np.uint8(cv2.normalize(heatmap_large, None, 0, 255,
                                           cv2.NORM_MINMAX))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    return _rot180(out) if rot180 else out


def _annotate_common(imgs, render, res, min_axis_strength=0.005, rot180=False):
    """Center cross, max-div marker, and the directional E_m arc on all views.

    Perception reports max_loc / arc_vec in UNROTATED coordinates, so when the
    display is rotated both the point and the arc direction are mapped through
    the same 180 deg rotation (which negates the direction vector). Text is
    drawn afterwards in display coordinates so it stays readable.
    """
    h, w = render.frame_bgr.shape[:2]
    center = (w // 2, h // 2)          # invariant under 180 deg rotation

    for img in imgs:
        cv2.drawMarker(img, center, CENTER_COL,
                       markerType=cv2.MARKER_CROSS, markerSize=45, thickness=2)

    if res.max_loc is not None:
        loc = _rot180_pt(res.max_loc, w, h) if rot180 else tuple(int(v) for v in res.max_loc)
        for img in imgs:
            cv2.drawMarker(img, loc, MAXDIV_COL,
                           markerType=cv2.MARKER_CROSS, markerSize=45, thickness=3)

        # E_m arc as a directional arrow (sense already resolved by grad(div))
        if res.axis_strength > 0:
            L = 90
            avx, avy = res.arc_vec
            if rot180:                  # 180 deg rotation negates a direction
                avx, avy = -avx, -avy
            tip = (int(loc[0] + L * avx), int(loc[1] + L * avy))
            tail = (int(loc[0] - L * avx), int(loc[1] - L * avy))
            weak = res.axis_strength < min_axis_strength
            acol = ARC_WEAK if weak else ARC_STRONG
            for img in imgs:
                cv2.line(img, tail, loc, acol, 1)
                cv2.arrowedLine(img, loc, tip, acol, 2, tipLength=0.25)
                cv2.putText(img, f"E_m arc  |s|={res.axis_strength:.4f}"
                            + ("  HELD (near-frontal)" if weak else ""),
                            (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, acol, 1)


def _hud(img, tau, depth, res, event):
    tau_s = f"{tau:.2f} s" if tau is not None else "--"
    dep_s = f"{depth:.3f} m" if depth is not None else "--"
    draw = f"{res.divmax_raw:.5f}" if (res and res.divmax_raw == res.divmax_raw) else "--"
    nvec = res.n_vectors if res else 0

    cv2.putText(img, f"TTC(flow)  {tau_s}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COL, 2)
    cv2.putText(img, f"depth(gt)  {dep_s}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_DIM, 1)
    cv2.putText(img, f"D_raw {draw}   nvec {nvec}", (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_DIM, 1)
    if event:
        col = HUD_WARN if event.startswith("STOP") else HUD_DIM
        cv2.putText(img, event, (20, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


class MaxDivVisualizer:
    """Renders flow + divergence views from a RenderBundle, shows them, and
    optionally records a side-by-side composite. Recording toggles with 'R',
    quit with 'q' or ESC (the show() return value signals quit)."""

    def __init__(self, out_dir="recordings", fps=30,
                 window="maxdiv (flow | divergence)", rotate_180=True):
        """:param rotate_180: present frames rotated 180 deg, for a camera
        mounted upside down about its optical axis. DISPLAY/RECORDING ONLY --
        perception is never rotated (see the note at the top of this module)."""
        self.rotate_180 = rotate_180
        self.out_dir = out_dir
        self.fps = fps
        self.window = window
        os.makedirs(out_dir, exist_ok=True)

        self.recording = False
        self.trial = 0
        self.raw_writer = None
        self.ovl_writer = None
        self.raw_path = self.ovl_path = None
        self._size = None      # (w, h) of a single view

    # ---------------------------------------------------------------- recording
    def _start_recording(self, w, h):
        self.trial += 1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        base = f"trial_{self.trial:02d}_{datetime.now().strftime('%H%M%S')}"
        self.raw_path = os.path.join(self.out_dir, base + "_raw.mp4")
        self.ovl_path = os.path.join(self.out_dir, base + "_overlay.mp4")
        self.raw_writer = cv2.VideoWriter(self.raw_path, fourcc, self.fps, (w, h))
        self.ovl_writer = cv2.VideoWriter(self.ovl_path, fourcc, self.fps, (w * 2, h))
        if not self.raw_writer.isOpened() or not self.ovl_writer.isOpened():
            print("ERROR: could not open video writers -- recording NOT started")
            self.raw_writer = self.ovl_writer = None
            return
        self.recording = True
        print(f"\n*** REC START  trial {self.trial:02d} ***")
        print(f"    raw:     {self.raw_path}")
        print(f"    overlay: {self.ovl_path}\n")

    def _stop_recording(self):
        if self.raw_writer is not None:
            self.raw_writer.release()
        if self.ovl_writer is not None:
            self.ovl_writer.release()
        self.raw_writer = self.ovl_writer = None
        self.recording = False
        print("*** REC STOP ***")

    def toggle_recording(self):
        if self.recording:
            self._stop_recording()
        elif self._size is not None:
            self._start_recording(*self._size)

    # ---------------------------------------------------------------- main draw
    def show(self, res, tau, depth, event=""):
        """Draw + display (and record if active) one frame.

        Returns False if the user asked to quit, True otherwise. If res.render
        is None (primed/invalid frame with no bundle), this is a no-op that
        still services the window so 'q'/'R' stay responsive.
        """
        key = cv2.waitKey(1) & 0xFF

        if res is None or res.render is None:
            if key in (ord('q'), 27):
                return False
            if key == ord('r'):
                self.toggle_recording()
            return True

        render = res.render
        h, w = render.frame_bgr.shape[:2]
        self._size = (w, h)

        flow_view = _draw_flow(render, self.rotate_180)
        div_view = _draw_heatmap(render, self.rotate_180)
        _annotate_common((flow_view, div_view), render, res,
                         rot180=self.rotate_180)

        # HUD on the flow view (left)
        _hud(flow_view, tau, depth, res, event)

        # recording indicator
        if self.recording:
            cv2.circle(flow_view, (w - 20, 20), 7, (0, 0, 255), -1)

        composite = np.hstack([flow_view, div_view])

        cv2.imshow(self.window, composite)

        if self.recording and self.raw_writer is not None:
            raw = render.frame_bgr
            self.raw_writer.write(_rot180(raw) if self.rotate_180 else raw)
            self.ovl_writer.write(composite)

        if key in (ord('q'), 27):
            return False
        if key == ord('r'):
            self.toggle_recording()
        return True

    def close(self):
        if self.recording:
            self._stop_recording()
        cv2.destroyAllWindows()