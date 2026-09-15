#!/usr/bin/env python3
"""
realsense_cue.py

RealSense D415 -> spherical-flow-divergence TTC cue provider.

Bridges the live camera to maxdiv_perception.MaxDivProcessor and exposes the
get_tau() interface the tau-dock controller consumes. This is the seam between
perception and control on the arm rig; it mirrors the cue-provider contract used
by the analytic simulator so the SAME control code drives both.

WHAT THIS FIXES VS THE MAMBO
----------------------------
The Mambo used a nominal frame_interval (1/10 Hz fallback). Here we compute the
REAL per-frame dt from the RealSense frame timestamps every tick. TTC scales
linearly with dt, so this is not optional: a wrong dt scales every TTC estimate.

Depth is streamed in PARALLEL as ground truth for validation (flow-derived TTC
vs depth-measured range) and kept STRICTLY OUT of the control cue. depth_scale
and the center-region range are logged, not fed to the controller.
"""

import time
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError("pyrealsense2 not installed. Run: pip install pyrealsense2")

from maxdiv_perception import MaxDivProcessor


class RealSenseCue:
    """Cue provider: get_tau() -> Optional[float] (seconds), or None if invalid.

    Also exposes get_last() for the full perception Result (max-div location,
    raw divergence, arc vector) and get_depth_range() for the parallel
    validation channel.
    """

    def __init__(self,
                 color_w=640, color_h=480, color_fps=60,
                 enable_depth=True,
                 process_size=(640, 360),
                 proc_cfg=None,
                 dt_clip=(1e-3, 0.2),
                 dt_median_window=5,
                 stride=8,
                 d_sample_mode='central'):
        """
        :param process_size: (w, h) the flow runs at. Frames are resized to this
            inside the processor. 640x360 matches the Mambo tuning; the D415
            color is 640x480, so the resize changes aspect slightly -- see the
            note in start() about matching aspect if you care about geometry.
        :param dt_clip: (min, max) seconds. Clamps the measured inter-frame dt to
            reject timestamp glitches / stalls that would otherwise scale TTC
            wildly. A stall producing a huge dt must not inflate TTC.
        :param dt_median_window: number of recent inter-frame intervals to median
            over. The D415 delivers frames on an uneven cadence (observed as a
            ~2:1 alternation in dt, which shows up as a 2x oscillation in TTC
            since TTC scales linearly with dt). Median-filtering the timebase at
            the SOURCE removes that artifact without touching the divergence
            path. The divergence field is already temporally smoothed; the dt
            should be too. Set to 1 to disable.
        """
        self.color_w, self.color_h, self.color_fps = color_w, color_h, color_fps
        self.enable_depth = enable_depth
        self.dt_min, self.dt_max = dt_clip
        self.dt_median_window = max(1, int(dt_median_window))
        self._dt_buf = []

        self.pipeline = None
        self.align = None
        self.depth_scale = None

        # merge process_size + stride into the processor config
        cfg = dict(process_size=process_size, stride=stride,
                   d_sample_mode=d_sample_mode)
        if proc_cfg:
            cfg.update(proc_cfg)
        self.proc = MaxDivProcessor(cfg)

        self._prev_ts = None          # previous frame timestamp (seconds)
        self._last_result = None
        self._last_depth_range = None

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.color_w, self.color_h,
                             rs.format.bgr8, self.color_fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.color_w, self.color_h,
                                 rs.format.z16, self.color_fps)
        profile = self.pipeline.start(config)

        if self.enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            self.align = rs.align(rs.stream.color)

        # discard a few frames so auto-exposure settles before control starts
        for _ in range(20):
            self.pipeline.wait_for_frames()

        self.proc.reset()
        self._prev_ts = None
        self._dt_buf = []

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    # ------------------------------------------------------------------- cue API
    def get_tau(self, want_render=False) -> Optional[float]:
        """Pull the next frame, run perception, return calibrated TTC (s) or None.

        Blocks briefly on wait_for_frames. Returns None on any invalid frame
        (no valid divergence, timestamp glitch, dropped frame) so the controller
        coasts rather than acting on a bad estimate.

        If want_render=True, the stored Result (get_last()) carries a render
        bundle for the visualiser. This does not affect the returned tau.
        """
        if self.pipeline is None:
            return None

        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        color = frames.get_color_frame()
        if not color:
            return None

        # --- measured per-frame dt from the camera's own timestamp (ms -> s) ---
        ts = color.get_timestamp() / 1000.0
        if self._prev_ts is None:
            dt = None                      # first framed pair not yet available
        else:
            raw_dt = ts - self._prev_ts
            if not (self.dt_min <= raw_dt <= self.dt_max):
                # glitch/stall: don't trust this interval; skip TTC this tick
                dt = None
            else:
                # median-filter the timebase to remove the D415's ~2:1 dt
                # alternation (which otherwise oscillates TTC 2x frame-to-frame).
                self._dt_buf.append(raw_dt)
                if len(self._dt_buf) > self.dt_median_window:
                    self._dt_buf.pop(0)
                dt = float(np.median(self._dt_buf))
        self._prev_ts = ts

        color_img = np.asanyarray(color.get_data())

        # --- parallel depth ground truth (validation only, NOT the cue) ---
        if self.enable_depth:
            depth = frames.get_depth_frame()
            if depth:
                self._last_depth_range = self._center_range(depth)

        # --- perception ---
        res = self.proc.process(color_img, dt, want_render=want_render)
        self._last_result = res

        return res.tau_s if res.valid else None

    # --------------------------------------------------------------- validation
    def _center_range(self, depth_frame, half=20):
        """Median depth (m) over a small central window; None if no return."""
        cx, cy = self.color_w // 2, self.color_h // 2
        vals = []
        for dy in range(-half, half + 1, 5):
            for dx in range(-half, half + 1, 5):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0:
                    vals.append(d)
        return float(np.median(vals)) if vals else None

    def get_depth_range(self) -> Optional[float]:
        """Latest center-region range in metres (validation channel), or None."""
        return self._last_depth_range

    def get_last(self):
        """Latest full perception Result (or None)."""
        return self._last_result


# ---------------------------------------------------------------- self-test
def _selftest():
    """Stream for a few seconds and print tau + depth side by side.

    Move the camera slowly toward a textured surface and you should see tau
    fall as depth falls -- the two channels tracking each other is the sanity
    check that perception is calibrated before any arm motion.
    """
    cue = RealSenseCue()
    cue.start()
    print("Streaming cue self-test. Move slowly toward a textured surface.")
    print(f"{'tau_s':>8}  {'depth_m':>8}  {'D_raw':>10}  {'maxloc':>12}  {'nvec':>5}")
    t_end = time.time() + 12.0
    try:
        while time.time() < t_end:
            tau = cue.get_tau()
            res = cue.get_last()
            depth = cue.get_depth_range()
            tau_s = f"{tau:8.3f}" if tau is not None else "    None"
            dep_s = f"{depth:8.3f}" if depth is not None else "    None"
            if res is not None:
                draw = f"{res.divmax_raw:10.6f}" if np.isfinite(res.divmax_raw) else "       nan"
                loc = f"{res.max_loc}" if res.max_loc else "        -"
                nv = res.n_vectors
            else:
                draw, loc, nv = "       nan", "        -", 0
            print(f"{tau_s}  {dep_s}  {draw}  {loc:>12}  {nv:5d}")
    except KeyboardInterrupt:
        pass
    finally:
        cue.stop()
        print("Cue self-test stopped.")


if __name__ == "__main__":
    _selftest()
