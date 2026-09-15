#!/usr/bin/env python3
"""
maxdiv_perception.py

Shared spherical-flow-divergence perception core, packaged as one importable
module so the live arm rig, the analytic simulator's PerceptionCueProvider, and
any offline replay all use ONE copy of the flow/localiser/calibration code.

PROVENANCE
----------
The flow front-end, Sobel-Jacobian divergence, FOE-free shear-axis max-div
localiser, and the raw-divergence -> TTC-in-seconds calibration are lifted
faithfully from Chris McCarthy's `mambo_divergence.py` (the calibrated live
build), NOT from `maxdiv-shear.py` (which min-max-normalises the heatmap and
exposes only a per-frame proxy with no physical units).

THE CALIBRATION THAT MATTERS
----------------------------
Two corrections make TTC physically meaningful, and both are preserved here:

  1. Sobel gain. Sobel kernels are UNNORMALISED. ksize=7 has a gain of 2048
     (response to a unit-slope ramp). The divergence must be divided by
     (skip * SOBEL_GAIN), not just skip. Omitting the 2048 collapses 1/D to ~0
     and TTC floors to zero immediately.

  2. Frames -> seconds. TTC = (2 / D_raw) frames, then * frame_interval to get
     seconds. D = 2V/Z for a frontal plane, so 2/D = Z/V = TTC in frames.

     >>> On the Mambo, frame_interval came from a nominal 10 Hz (DT=0.1) fallback.
     >>> On the arm rig, PASS THE REALSENSE'S MEASURED PER-FRAME dt. TTC scales
     >>> linearly with frame_interval, so a wrong dt scales every TTC estimate.

Note: the scalar TTC in this frontal-plane formulation does NOT depend on camera
intrinsics (fx/fy/pp). Intrinsics matter for spherical reprojection / off-axis
geometry, not for this straight-in docking cue. So the D415's zero-distortion
pinhole is a bonus for later spherical work, not a prerequisite here.
"""

import numpy as np
import cv2
from collections import deque


# ============================================================ calibration
SOBEL_KSIZE = 7
# ksize=3 -> 8,  ksize=5 -> 128,  ksize=7 -> 2048  (unnormalised Sobel gain)
_SOBEL_GAIN_BY_K = {3: 8.0, 5: 128.0, 7: 2048.0}
SOBEL_GAIN = _SOBEL_GAIN_BY_K[SOBEL_KSIZE]


# ============================================================ flow config
DEFAULTS = dict(
    process_size=(640, 360),   # (w, h) the flow runs at; frames are resized to this
    skip=40,                   # grid spacing (px)
    # LOWERED from 0.5 for the shear cue. The gate rejects small-displacement
    # vectors, which is fine for DIVERGENCE (large, isotropic) but discards
    # exactly the vectors carrying accurate flow for SHEAR: LK error grows
    # sharply with displacement (measured sd 0.10 px at ~1 px displacement vs
    # 36 px at ~14 px), so the small vectors are the GOOD ones. Measured effect
    # on shear-direction recovery (ideal slope 0.5, corr 1.0):
    #     min_vector_magnitude=0.5 -> corr -0.39 (BROKEN, sign unstable)
    #     min_vector_magnitude=0.1 -> corr +0.99 (correct)
    min_vector_magnitude=0.1,
    max_vector_magnitude=30.0,
    max_vector_component=30.0,
    max_lk_error=15.0,
    temporal_alpha=0.75,       # divergence field smoothing
    vector_temporal_alpha=0.6, # flow vector smoothing
    sobel_pre_blur_kernel=(3, 3),
    shear_sobel_ksize=3,       # separate (smaller) kernel for SHEAR only
    search_px=200,             # how far ALONG the arc to search (px)
    min_axis_strength=0.005,   # below this the shear axis is ill-conditioned
    ttc_neighbourhood=1,       # half-size of the grid patch averaged for D_raw
    # --- how D is sampled for the TTC estimate -----------------------------
    # 'maxdiv'  : sample at the FOE-free max-div location. Correct for OBLIQUE
    #             surfaces, where divergence genuinely varies across the image
    #             and the E_m arc has a real peak to find.
    # 'central' : robust spatial aggregate over a central region.
    #
    # WHY THIS MATTERS: for a FRONTO-PARALLEL approach along the optical axis,
    # divergence is theoretically CONSTANT over the image (u = xV/Z, v = yV/Z,
    # so du/dx + dv/dy = 2V/Z everywhere). There is no peak. The max-div search
    # then has nothing to lock onto, is driven by noise, and walks to the limit
    # of its search range -- observed pinned at x=0 (the image edge) on a 35
    # mm/s run, and stuck at x~61 and wandering x=146-426 on two others, when
    # the FOE should sit near the image centre (320,180).
    #
    # Sampling at the boundary under-estimates D badly (truncated LK windows,
    # fewer valid neighbours for the Sobel), which is what produced the
    # irreproducible "gain" figures of 0.33-0.92 across runs.
    #
    # For a docking task the surface is frontal by construction, so 'central'
    # is the appropriate estimator.
    d_sample_mode='central',
    central_frac=0.5,          # central region as a fraction of the grid
    # --- adaptive (auto) mode -----------------------------------------------
    # 'auto' measures how PEAKED the divergence field actually is and switches:
    #   peaked  -> sample at the FOE-free max-div location (oblique surface)
    #   flat    -> robust central median (frontal surface)
    #
    # WARNING: 'auto' is NOT yet validated. The discriminant does not reliably
    # separate frontal from oblique in simulation, because the boundary droop
    # (below) gives even a frontal field strong apparent spatial structure.
    # Do not rely on 'auto' until it has been checked against real oblique
    # data. Use 'central' or 'maxdiv' explicitly.
    #
    # Peak contrast is (D_peak - D_median) / IQR(D), a scale-free measure of how
    # far the maximum stands above the bulk of the field relative to its spread.
    # On a frontal plane the true field is constant, so contrast -> small.
    peak_contrast_min=1.5,     # below this, treat the field as flat
    # MEASURED: on a perfectly uniform synthetic frontal scene, where theory
    # says divergence is constant, the computed field droops ~3x at the grid
    # edges (column means 0.0088 / 0.027 / 0.0104 left/centre/right). This is
    # inherent to the Sobel operating at the array boundary, NOT a texture
    # problem. Excluding a margin cuts the field's coefficient of variation
    # from 0.42 (margin 0) to 0.16 (margin 2) to 0.12 (margin 3).
    # Never localise onto grid cells whose Sobel/LK support is truncated. The
    # search is restricted to the interior, excluding this many grid cells at
    # each border. Boundary cells under-estimate divergence (truncated LK
    # windows, missing Sobel neighbours) and the localiser has been observed
    # pinning to them (maxloc_y = 0) once the seed drifts there.
    interior_margin_cells=3,
    # If the seed sits on the interior boundary for this many consecutive
    # frames, re-seed from the image centre rather than compounding the drift.
    reseed_after_stuck=5,
    # --- multi-frame flow stride -------------------------------------------
    # Number of frames back to the LK reference frame. stride=1 is classic
    # frame-to-frame flow (the Mambo behaviour).
    #
    # WHY: LK measures DISPLACEMENT. At slow manipulator speeds the per-frame
    # displacement falls below both LK's resolution and min_vector_magnitude,
    # so the flow is filtered out as noise. Observed: 15 mm/s at 0.8 m gives
    # fx*V/Z/fps = 607*0.015/0.8/60 ~= 0.19 px/frame, well under the 0.5 px gate
    # (measured nvec 5-12, D_raw ~ +/-0.0005, often negative).
    #
    # Matching against a frame N back accumulates displacement ~N x, bringing it
    # back into LK's working range and allowing a slow, safe approach.
    #
    # COSTS (use the SMALLEST stride that clears the floor):
    #   - LK degrades over larger displacements (needs pyramid levels)
    #   - adds latency: tau reflects motion over the stride window, so it lags
    #   - the scene must not change much across the window
    stride=1,
)

LK_PRESETS = {
    "stable": dict(winSize=(31, 31), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)),
    "balanced": dict(winSize=(21, 21), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02)),
    "sensitive": dict(winSize=(15, 15), maxLevel=1,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
}


# ============================================================ helpers (lifted)
def get_vector_mask(u, v, cfg):
    magnitude = np.sqrt(u ** 2 + v ** 2)
    m = np.isfinite(u) & np.isfinite(v)
    m &= (magnitude >= cfg["min_vector_magnitude"])
    m &= (magnitude <= cfg["max_vector_magnitude"])
    m &= (np.abs(u) <= cfg["max_vector_component"])
    m &= (np.abs(v) <= cfg["max_vector_component"])
    return m, magnitude


def get_patch_max_location(heatmap_uint8, patch_size=31):
    patch_average = cv2.blur(heatmap_uint8, (patch_size, patch_size))
    _, max_val, _, max_loc = cv2.minMaxLoc(patch_average)
    return max_val, max_loc


def get_shear_axis_constrained_max(div_field, shear_s1, shear_s2, shear_mag,
                                   seed_xy, skip, w, h,
                                   band_half=1, search_px=200, tensor_half=2):
    """FOE-FREE max-div localisation via the shear axis + divergence gradient.

    Recovers the E_m arc orientation from the deformation-tensor principal axis
    at the seed (no FOE, no verticality assumption), searches 1-D ALONG the arc
    averaging a thin band ACROSS it, and uses grad(div) along the axis to fix the
    180deg sense. Returns (max_val, (x,y), (ax,ay) SIGNED arc unit vector,
    shear strength).

    IMPORTANT: use the returned VECTOR, never reconstruct one from an angle.
    (Faithful extraction from mambo_divergence.py.)
    """
    gh, gw = div_field.shape

    # --- 1. deformation tensor at the seed -> arc orientation (NO FOE) ---
    sx = int(round(seed_xy[0] / skip))
    sy = int(round(seed_xy[1] / skip))
    sx = max(0, min(gw - 1, sx))
    sy = max(0, min(gh - 1, sy))

    x0, x1 = max(0, sx - tensor_half), min(gw, sx + tensor_half + 1)
    y0, y1 = max(0, sy - tensor_half), min(gh, sy + tensor_half + 1)

    wts = shear_mag[y0:y1, x0:x1]
    ws = float(wts.sum())
    if ws < 1e-9:
        return 0.0, (int(seed_xy[0]), int(seed_xy[1])), (1.0, 0.0), 0.0

    s1 = float(np.sum(shear_s1[y0:y1, x0:x1] * wts) / ws)
    s2 = float(np.sum(shear_s2[y0:y1, x0:x1] * wts) / ws)
    strength = float(np.hypot(s1, s2))

    arc = 0.5 * np.arctan2(s2, s1)          # principal axis == E_m direction
    ax, ay = np.cos(arc), np.sin(arc)
    px, py = -ay, ax                        # across-arc (the flat direction)

    # --- 2. 1-D search ALONG the arc, averaging a thin band ACROSS it ---
    ts = np.arange(-search_px, search_px + 1, skip / 2.0)
    prof = np.full(len(ts), -1e18)

    for i, t in enumerate(ts):
        acc, n = 0.0, 0
        for b in range(-band_half, band_half + 1):
            qx = seed_xy[0] + t * ax + b * skip * px
            qy = seed_xy[1] + t * ay + b * skip * py
            gxi = int(round(qx / skip))
            gyi = int(round(qy / skip))
            if 0 <= gxi < gw and 0 <= gyi < gh:
                acc += float(div_field[gyi, gxi])
                n += 1
        if n:
            prof[i] = acc / n

    gi = int(np.argmax(prof))
    best_val = float(prof[gi])
    best_t = float(ts[gi])

    mx = float(np.clip(seed_xy[0] + best_t * ax, 0, w - 1))
    my = float(np.clip(seed_xy[1] + best_t * ay, 0, h - 1))

    # --- 3. divergence GRADIENT along the arc -> resolve the 180deg ambiguity ---
    lo = float(np.mean(prof[max(0, gi - 3):gi])) if gi >= 1 else float(prof[0])
    hi = (float(np.mean(prof[gi + 1:gi + 4]))
          if gi + 1 < len(prof) else float(prof[-1]))
    sense = 1.0 if hi >= lo else -1.0

    return best_val, (int(round(mx)), int(round(my))), \
        (sense * ax, sense * ay), strength


# ============================================================ processor
class MaxDivProcessor:
    """Stateful per-frame spherical-flow-divergence processor (FOE-free).

    Feed it consecutive BGR frames plus the measured inter-frame dt; it returns a
    result carrying calibrated TTC in seconds (or None when no valid estimate is
    available this frame). Holds all temporal state internally: previous frame,
    smoothed flow, smoothed divergence, and the previous max-div seed.

    Usage:
        proc = MaxDivProcessor()
        # first frame just primes the state:
        proc.process(frame0, dt=0.0)
        # subsequent frames return results:
        res = proc.process(frame1, dt=measured_dt)
        if res.tau_s is not None:
            ... use res.tau_s ...
    """

    class Result:
        __slots__ = ("tau_s", "divmax_raw", "max_loc", "arc_vec",
                     "axis_strength", "n_vectors", "valid", "render",
                     "peak_contrast", "sample_mode_used", "seed_stuck",
                     "shear_field_med", "div_field_med", "shear_norm_field",
                     "shear_field_vec")

        def __init__(self):
            self.tau_s = None          # calibrated TTC in seconds, or None
            self.divmax_raw = float("nan")
            self.max_loc = None        # (x, y) in process-size pixels
            self.arc_vec = (1.0, 0.0)  # signed E_m arc unit vector
            self.axis_strength = 0.0
            self.n_vectors = 0
            self.valid = False
            self.render = None         # optional RenderBundle for visualisation
            self.peak_contrast = 0.0   # how peaked the divergence field is
            self.sample_mode_used = "" # 'maxdiv' or 'central' (what auto chose)
            self.seed_stuck = 0        # frames the seed has sat on the border
            # --- field-median shear cue (see notes in process()) ---
            self.shear_field_med = float("nan")   # median |shear| over interior
            self.div_field_med = float("nan")     # median divergence over interior
            self.shear_norm_field = float("nan")  # range-invariant obliquity cue
            self.shear_field_vec = (float("nan"), float("nan"))  # (s1,s2) medians

    class RenderBundle:
        """Intermediate fields needed to draw the annotated view. Populated only
        when process(..., want_render=True). Carrying these does NOT change any
        calibrated computation -- they are the same arrays used internally."""
        __slots__ = ("frame_bgr", "grid_points", "u", "v", "valid_lk",
                     "divergence", "grid_w", "grid_h", "skip", "pw", "ph")

        def __init__(self):
            self.frame_bgr = None      # the (resized) BGR frame drawn on
            self.grid_points = None    # (N,1,2) grid sample points
            self.u = None              # per-point flow u (filtered, for display)
            self.v = None              # per-point flow v
            self.valid_lk = None       # per-point validity mask
            self.divergence = None     # smoothed divergence field (grid_h,grid_w)
            self.grid_w = 0
            self.grid_h = 0
            self.skip = 0
            self.pw = 0
            self.ph = 0

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update(cfg)
        self.pw, self.ph = self.cfg["process_size"]
        # anisotropic-resize compensation (set per frame in process())
        self._src_wh = None
        self._aniso_v_scale = 1.0
        self.skip = self.cfg["skip"]
        self.lk_params = LK_PRESETS["stable"]

        # precompute grid points once
        gx = np.arange(0, self.pw, self.skip)
        gy = np.arange(0, self.ph, self.skip)
        self.grid_w = len(gx)
        self.grid_h = len(gy)
        self.grid_points = np.array(
            [[x, y] for y in gy for x in gx], dtype=np.float32
        ).reshape(-1, 1, 2)

        # temporal state
        # History of (gray_frame, cumulative_time) so LK can match against a
        # frame `stride` back and we can measure the ACTUAL elapsed time across
        # that window (never assume stride * nominal_dt).
        self.stride = max(1, int(self.cfg["stride"]))
        self.frame_hist = deque(maxlen=self.stride + 1)
        self._t_accum = 0.0
        self.prev_u = None
        self.prev_v = None
        self.prev_divergence = None
        self.prev_max_loc = None
        self._seed_stuck = 0

    def reset(self):
        self.frame_hist.clear()
        self._t_accum = 0.0
        self.prev_u = None
        self.prev_v = None
        self.prev_divergence = None
        self.prev_max_loc = None
        self._seed_stuck = 0

    def process(self, frame_bgr, dt, want_render=False):
        """Process one frame. dt = measured seconds since the previous frame.

        Returns a Result. The FIRST call (or any call right after reset()) only
        primes the previous-frame state and returns an invalid Result.

        If want_render=True, res.render is populated with the intermediate
        fields (frame, flow vectors, divergence field) needed to draw the
        annotated view. This does not change any calibrated computation.
        """
        cfg = self.cfg
        res = self.Result()

        self._src_wh = (frame_bgr.shape[1], frame_bgr.shape[0])
        if frame_bgr.shape[1] != self.pw or frame_bgr.shape[0] != self.ph:
            frame_bgr = cv2.resize(frame_bgr, (self.pw, self.ph))
        # --- ANISOTROPY CORRECTION (spurious-shear fix) -----------------------
        # The frame is resized from the camera's native aspect to process_size.
        # When those aspects differ (D415 640x480 -> 640x360: x*1.0, y*0.75) the
        # VERTICAL flow component is compressed relative to the horizontal, so
        # u and v are in DIFFERENT units. The divergence du_dx+dv_dy tolerates
        # this as a mild scale error, but the SHEAR does not:
        #     shear_s1 = du_dx - dv_dy
        # relies on those two terms CANCELLING for an isotropic (frontal) flow.
        # With v compressed by ay, dv_dy is systematically too small and s1 picks
        # up a spurious (1-ay)*du_dx -- a fake shear PROPORTIONAL TO DIVERGENCE,
        # present at exactly zero obliquity.
        #
        # Measured before this fix: normalised shear at a TRUE FRONTAL surface
        # sat at 0.055 and was independent of flow magnitude (corr +0.14 across
        # a 2.5x range of divergence) -- a constant multiplicative artifact. It
        # also made 0 deg read HIGHER than 10 deg in every flow bin, and meant
        # raising approach speed did not improve SNR (signal and artifact scaled
        # together: slope and IQR both doubled from 25->60 mm/s).
        #
        # Rescaling v back to horizontal units restores du_dx == dv_dy for an
        # isotropic field, so frontal shear is genuinely ~0.
        self._aniso_v_scale = 1.0
        if self._src_wh is not None:
            sw, sh = self._src_wh
            ax_ = self.pw / float(sw)
            ay_ = self.ph / float(sh)
            if ay_ > 1e-9 and abs(ax_ - ay_) > 1e-6:
                self._aniso_v_scale = ax_ / ay_

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # advance the running clock and buffer this frame
        if dt is not None and dt > 0:
            self._t_accum += dt
        self.frame_hist.append((gray, self._t_accum))

        # need a reference frame `stride` back before any flow is possible
        if len(self.frame_hist) < self.stride + 1:
            return res   # priming

        gray1, t_ref = self.frame_hist[0]      # oldest = stride frames back
        gray2, t_now = self.frame_hist[-1]     # newest = current

        # ACTUAL elapsed time across the stride window. Never assume
        # stride * nominal_dt -- the divergence is measured over this span, so
        # tau scales directly with it and an assumed value biases every estimate.
        span_dt = t_now - t_ref
        if span_dt <= 1e-6:
            span_dt = None

        # --- grid Lucas-Kanade ---
        new_pts, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, self.grid_points, None, **self.lk_params
        )
        if new_pts is None or status is None or error is None:
            return res

        status = status.flatten()
        error = error.flatten()
        movement = new_pts[:, 0, :] - self.grid_points[:, 0, :]
        valid_status = status == 1

        u = np.zeros(len(self.grid_points), dtype=np.float32)
        v = np.zeros(len(self.grid_points), dtype=np.float32)
        u[valid_status] = movement[valid_status, 0]
        v[valid_status] = movement[valid_status, 1]

        # --- vector temporal smoothing ---
        a = cfg["vector_temporal_alpha"]
        if (self.prev_u is not None and self.prev_v is not None
                and self.prev_u.shape == u.shape):
            u = a * self.prev_u + (1 - a) * u
            v = a * self.prev_v + (1 - a) * v
        self.prev_u = u.copy()
        self.prev_v = v.copy()

        # --- vector filtering ---
        size_mask, mag_all = get_vector_mask(u, v, cfg)
        valid_lk = (status == 1) & (error <= cfg["max_lk_error"]) & size_mask
        res.n_vectors = int(np.sum(valid_lk))

        u_f = np.where(valid_lk, u, 0)
        v_f = np.where(valid_lk, v, 0)
        u_grid = u_f.reshape(self.grid_h, self.grid_w).astype(np.float32)
        v_grid = v_f.reshape(self.grid_h, self.grid_w).astype(np.float32)

        # Put v back into horizontal units (see ANISOTROPY CORRECTION above).
        # 1.0 when the resize preserves aspect, so this is a no-op for square
        # pixels and does not disturb any existing calibration in that case.
        # Keep RAW copies for the shear path before any rescale/blur (see the
        # shear derivative block below for the measured reason).
        u_grid_raw = u_grid.copy()
        v_grid_raw = v_grid.copy()

        if self._aniso_v_scale != 1.0:
            v_grid = v_grid * self._aniso_v_scale

        u_gs = cv2.GaussianBlur(u_grid, cfg["sobel_pre_blur_kernel"], 0)
        v_gs = cv2.GaussianBlur(v_grid, cfg["sobel_pre_blur_kernel"], 0)

        # --- Sobel Jacobian WITH the 2048 gain correction ---
        _sc = self.skip * SOBEL_GAIN
        du_dx = cv2.Sobel(u_gs, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE) / _sc
        dv_dy = cv2.Sobel(v_gs, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE) / _sc
        du_dy = cv2.Sobel(u_gs, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE) / _sc
        dv_dx = cv2.Sobel(v_gs, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE) / _sc

        # --- SHEAR derivatives use their OWN (smaller) kernel ---------------
        # Divergence is the large isotropic part of the Jacobian and is happy
        # with the wide, calibrated ksize=7 kernel. SHEAR is a small DIFFERENCE
        # between nearly-equal derivatives, and a ksize=7 kernel spans about a
        # third of the ~24x18 grid, over-smoothing exactly the anisotropy it is
        # meant to measure.
        #
        # Validated on analytically-generated oblique-plane flow, with image
        # noise, across grid spacings (validate_shear_synthetic.py). Recovery of
        # the shear DIRECTION vs surface tilt azimuth (ideal slope 0.5):
        #     ksize=3 : +0.434 / +0.474 / +0.497,  corr +0.999  (all skips)
        #     ksize=5 : +0.802 / -0.476 / -0.170,  corr erratic, SIGN FLIPS
        #     ksize=7 : +0.764 / +0.803 / -0.328,  corr erratic, SIGN FLIPS
        # The sign instability at wide kernels is the same failure seen on the
        # rig, where the measured shear direction was uncorrelated with the true
        # surface tilt and flipped between runs. Magnitude is insensitive to the
        # kernel (corr >= +0.97 throughout), which is why divergence/tau-dot were
        # never affected.
        _ks = int(self.cfg.get("shear_sobel_ksize", 3))
        # SHEAR uses RAW grids: no anisotropy rescale, no pre-blur.
        #
        # Both were measured to destroy the shear DIRECTION (recovery of tilt
        # azimuth, ideal slope +0.5, on analytic oblique-plane flow through LK):
        #     raw grids            slope +0.638  corr +0.993   mean s1 -0.00002
        #     + Gaussian pre-blur  slope -0.066  corr -0.138
        #     + v anisotropy scale slope -0.024  corr -0.874   mean s1 -0.01457
        #     + both (as shipped)  slope -0.003  corr -0.346   mean s1 -0.01494
        # The v-rescale inflates s1 by ~700x, swamping the real signal and
        # pinning the doubled angle to ~180 deg (phi ~ +/-90) regardless of the
        # true surface tilt -- exactly the failure measured on the rig, where a
        # 180 deg camera roll moved phi by only 5 deg. The pre-blur removes the
        # spatial structure the shear derivative depends on, on a grid that is
        # only ~24x18 points to begin with.
        #
        # Divergence is unaffected by either (it is the large isotropic part),
        # so the calibrated divergence path above keeps blur + the wide kernel.
        _scs = self.skip * _SOBEL_GAIN_BY_K[_ks]
        _u_raw = u_grid_raw
        _v_raw = v_grid_raw
        _du_dx = cv2.Sobel(_u_raw, cv2.CV_64F, 1, 0, ksize=_ks) / _scs
        _dv_dy = cv2.Sobel(_v_raw, cv2.CV_64F, 0, 1, ksize=_ks) / _scs
        _du_dy = cv2.Sobel(_u_raw, cv2.CV_64F, 0, 1, ksize=_ks) / _scs
        _dv_dx = cv2.Sobel(_v_raw, cv2.CV_64F, 1, 0, ksize=_ks) / _scs

        raw_divergence = du_dx + dv_dy
        shear_s1 = _du_dx - _dv_dy
        shear_s2 = _du_dy + _dv_dx

        # --- FIELD-MEDIAN shear (recommended cue for approach-angle control) ---
        # The arc-constrained estimate below band-averages the shear ACROSS the
        # arc around the max-div point. That stabilises the shear DIRECTION, but
        # averaging over a region suppresses exactly the spatial anisotropy it is
        # meant to measure, and it samples only near one point.
        #
        # Validated against analytically-generated oblique-plane flow
        # (validate_shear_synthetic.py), true slope 0.0022 |s|/deg:
        #     arc-constrained : slope 0.00023/deg, corr +0.79  (~1/10 of truth)
        #     field median    : slope 0.00227/deg, corr +0.97  (matches truth)
        # and the noise-equivalent angle improves from tens of degrees to ~1.5
        # deg at skip=30. The field median over the interior of the grid is
        # therefore the cue to regulate.
        #
        # NOTE a constant pedestal: |shear| is positive-definite, so noise biases
        # the median up (~0.15 at true frontal). Regulate CHANGES against a floor
        # set from a measured frontal baseline; it is not an absolute obliquity.
        _in = (slice(2, -2), slice(2, -2))
        if raw_divergence[_in].size:
            _dmed = float(np.median(raw_divergence[_in]))
            _smed = float(np.median(np.hypot(shear_s1, shear_s2)[_in]))
            res.shear_field_med = _smed
            res.div_field_med = _dmed
            res.shear_norm_field = (_smed / _dmed) if abs(_dmed) > 1e-9 else float("nan")
            # Field-level shear DIRECTION: median of the (s1,s2) components over
            # the same interior. Unlike the magnitude these are signed, so the
            # median is an unbiased centre and gives the tilt direction the
            # control law needs (as a doubled-angle vector, phi = atan2(s2,s1)/2).
            _s1m = float(np.median(shear_s1[_in]))
            _s2m = float(np.median(shear_s2[_in]))
            res.shear_field_vec = (_s1m / _dmed, _s2m / _dmed) \
                if abs(_dmed) > 1e-9 else (float("nan"), float("nan"))
        else:
            res.shear_field_med = res.div_field_med = float("nan")
            res.shear_norm_field = float("nan")
        shear_mag = np.sqrt(shear_s1 ** 2 + shear_s2 ** 2)

        # --- divergence temporal smoothing ---
        ta = cfg["temporal_alpha"]
        if (self.prev_divergence is None
                or self.prev_divergence.shape != raw_divergence.shape):
            divergence = raw_divergence.copy()
        else:
            divergence = ta * self.prev_divergence + (1 - ta) * raw_divergence
        self.prev_divergence = divergence.copy()

        # --- seed for the FOE-free search ---
        # Restrict everything to the INTERIOR: boundary grid cells have
        # truncated LK windows and missing Sobel neighbours, so their
        # divergence is unreliable and biased low. The localiser has been
        # observed pinning to them (maxloc_y=0) once the seed drifts there.
        gh_, gw_ = divergence.shape
        mgn = int(cfg.get("interior_margin_cells", 2))
        ix0, ix1 = mgn, max(mgn + 1, gw_ - mgn)
        iy0, iy1 = mgn, max(mgn + 1, gh_ - mgn)
        interior = divergence[iy0:iy1, ix0:ix1]

        # --- peak contrast: is there a genuine SPATIAL STRUCTURE to localise? ---
        # On a frontal plane the true divergence field is CONSTANT (2V/Z), so
        # any apparent peak is noise. On an oblique plane, range varies across
        # the image, so divergence carries a systematic spatial RAMP.
        #
        # Peak height alone is a poor discriminant: it responds to noise as much
        # as to structure. Instead we fit a plane to the divergence field and
        # compare the variance EXPLAINED by that ramp against the residual
        # scatter. Genuine obliquity produces a coherent gradient (high ratio);
        # a flat field plus noise does not (low ratio). The measure is
        # scale-free, so it is independent of approach speed and range.
        if interior.size >= 9:
            ih, iw = interior.shape
            yy, xx = np.mgrid[0:ih, 0:iw]
            A = np.column_stack([xx.ravel().astype(float),
                                 yy.ravel().astype(float),
                                 np.ones(interior.size)])
            b = interior.ravel().astype(float)
            try:
                coef, *_ = np.linalg.lstsq(A, b, rcond=None)
                fit = A @ coef
                ss_res = float(np.sum((b - fit) ** 2))
                ss_tot = float(np.sum((b - b.mean()) ** 2))
                # F-like ratio: explained variance vs residual, per d.o.f.
                if ss_res > 1e-18 and interior.size > 3:
                    res.peak_contrast = ((ss_tot - ss_res) / 2.0) / \
                        (ss_res / (interior.size - 3))
                else:
                    res.peak_contrast = 0.0
            except np.linalg.LinAlgError:
                res.peak_contrast = 0.0
        else:
            res.peak_contrast = 0.0

        # re-seed if the previous seed has been stuck against the interior edge
        def _on_border(pt):
            gx = int(round(pt[0] / self.skip)); gy = int(round(pt[1] / self.skip))
            return (gx <= ix0 or gx >= ix1 - 1 or gy <= iy0 or gy >= iy1 - 1)

        if self.prev_max_loc is not None and _on_border(self.prev_max_loc):
            self._seed_stuck += 1
        else:
            self._seed_stuck = 0
        res.seed_stuck = self._seed_stuck

        if (self.prev_max_loc is not None
                and self._seed_stuck < int(cfg.get("reseed_after_stuck", 5))):
            seed_xy = self.prev_max_loc
        else:
            # re-seed from the interior argmax (falls back to centre if flat)
            if interior.size:
                gi = int(np.argmax(interior))
                gy0, gx0 = divmod(gi, interior.shape[1])
                seed_xy = ((gx0 + ix0) * self.skip, (gy0 + iy0) * self.skip)
            else:
                seed_xy = (self.pw // 2, self.ph // 2)
            self._seed_stuck = 0

        max_val, max_loc, arc_vec, axis_strength = get_shear_axis_constrained_max(
            divergence, shear_s1, shear_s2, shear_mag,
            seed_xy, self.skip, self.pw, self.ph,
            band_half=1, search_px=cfg["search_px"], tensor_half=2
        )

        # clamp the localised maximum into the valid interior -- the arc search
        # can otherwise walk onto boundary cells with unreliable divergence
        _xmin, _xmax = ix0 * self.skip, (ix1 - 1) * self.skip
        _ymin, _ymax = iy0 * self.skip, (iy1 - 1) * self.skip
        max_loc = (int(np.clip(max_loc[0], _xmin, _xmax)),
                   int(np.clip(max_loc[1], _ymin, _ymax)))

        # stabilise the max-div location (same 0.85/0.15 blend as the pilot)
        if self.prev_max_loc is None:
            stable_loc = max_loc
        else:
            stable_loc = (int(0.85 * self.prev_max_loc[0] + 0.15 * max_loc[0]),
                          int(0.85 * self.prev_max_loc[1] + 0.15 * max_loc[1]))
        stable_loc = (int(np.clip(stable_loc[0], _xmin, _xmax)),
                      int(np.clip(stable_loc[1], _ymin, _ymax)))
        self.prev_max_loc = stable_loc

        res.max_loc = stable_loc
        res.arc_vec = arc_vec
        res.axis_strength = axis_strength

        # --- sample divergence for the TTC estimate ---
        gh, gw = divergence.shape
        mode = cfg.get("d_sample_mode", "auto")
        if mode == "auto":
            # Choose on the MEASURED peakedness of the field, not on an
            # assumption about the surface. Frontal -> flat -> central;
            # oblique -> genuine peak -> maxdiv.
            use_central = res.peak_contrast < float(cfg.get("peak_contrast_min", 1.5))
        else:
            use_central = (mode == "central")
        res.sample_mode_used = "central" if use_central else "maxdiv"

        if use_central:
            # Divergence is (theoretically) uniform: robust spatial aggregate
            # over a central region. Median rejects bad-LK outliers; the
            # central crop avoids degraded boundary cells.
            f = float(cfg.get("central_frac", 0.5))
            hw = max(1, int(gw * f / 2))
            hh = max(1, int(gh * f / 2))
            cx, cy = gw // 2, gh // 2
            x0, x1 = max(0, cx - hw), min(gw, cx + hw + 1)
            y0, y1 = max(0, cy - hh), min(gh, cy + hh + 1)
            patch = divergence[y0:y1, x0:x1]
            if patch.size:
                res.divmax_raw = float(np.median(patch))
        else:
            # Genuine peak present: sample at the FOE-free max-div location.
            nb = cfg["ttc_neighbourhood"]
            gx_ = int(round(stable_loc[0] / self.skip))
            gy_ = int(round(stable_loc[1] / self.skip))
            x0, x1 = max(0, gx_ - nb), min(gw, gx_ + nb + 1)
            y0, y1 = max(0, gy_ - nb), min(gh, gy_ + nb + 1)
            if x1 > x0 and y1 > y0:
                res.divmax_raw = float(np.mean(divergence[y0:y1, x0:x1]))

        # --- calibrated TTC in seconds ---
        # D_raw is measured over the STRIDE WINDOW, so 2/D_raw is in units of
        # stride-windows-to-contact. Multiply by the MEASURED span of that
        # window to get seconds. With stride=1 this reduces to the original
        # (2/D) * dt. Using single-frame dt here with stride>1 would scale
        # every tau by 1/stride.
        if (np.isfinite(res.divmax_raw) and res.divmax_raw > 1e-9
                and span_dt is not None):
            ttc_windows = 2.0 / res.divmax_raw
            res.tau_s = ttc_windows * span_dt
            res.valid = True

        # --- optional render bundle (does not affect any computation above) ---
        if want_render:
            rb = self.RenderBundle()
            rb.frame_bgr = frame_bgr
            rb.grid_points = self.grid_points
            rb.u = u
            rb.v = v
            rb.valid_lk = valid_lk
            rb.divergence = divergence
            rb.grid_w = self.grid_w
            rb.grid_h = self.grid_h
            rb.skip = self.skip
            rb.pw = self.pw
            rb.ph = self.ph
            res.render = rb

        return res