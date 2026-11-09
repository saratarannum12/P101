#!/usr/bin/env python3
"""
measure_approach_angle.py

Ground-truth approach angle from the RealSense DEPTH stream, independent of the
optical-flow cue that the controller regulates.

WHY THIS EXISTS
---------------
The controller reports tilt_dev: the tool's rotation from its START pose, read
from joint encoders. That is a useful consistency measure -- four repeat runs
rotated 28.8-33.4 deg with monotone trajectories -- but it is NOT the angle to
the surface normal. It cannot say whether a run finished frontal, only how far
the tool turned. Every convergence claim needs the angle itself.

This tool fits a plane to the depth point cloud and reports the angle between
the surface normal and the camera's optical axis. 0 deg = frontal.

INDEPENDENCE
------------
Depth here comes from stereo disparity, not from optical flow, so this is a
genuinely different measurement path from the divergence/shear cue. It is a
VALIDATION channel only and is never fed to the controller -- the same
separation the docking controller already maintains for its depth logging.

(For a fully vision-independent ground truth, touch three points on the surface
with the tool to establish the plane in base coordinates, then derive the angle
from forward kinematics alone. That is stronger for publication but needs the
touch-off procedure first.)

ACCURACY NOTES
--------------
  * RANSAC, not least squares: a handful of bad depth returns will drag a plain
    fit. RANSAC ignores them.
  * Central region only: D415 depth degrades toward the frame edges, and the
    periphery may see background rather than the target surface.
  * Respect the sensor's working range. Below the D415 minimum (~0.16 m) depth
    is unusable; if the post-dock standoff is too close, retreat a known
    distance and measure there -- a pure translation does not change the angle.

USAGE
    python measure_approach_angle.py                 # single measurement
    python measure_approach_angle.py --watch         # continuous, ctrl-C to stop
    python measure_approach_angle.py --n 20          # average over N frames
"""

import argparse
import sys
import time

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not installed. Run: pip install pyrealsense2")


def fit_plane_ransac(pts, iters=200, thresh=0.004, rng=None):
    """RANSAC plane fit. Returns (normal, inlier_fraction) or (None, 0).

    thresh is the inlier distance in metres (4 mm suits D415 noise at ~0.5 m).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(pts)
    if n < 50:
        return None, 0.0
    best_normal, best_inliers = None, 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        v = np.cross(p1 - p0, p2 - p0)
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v = v / nrm
        d = np.abs((pts - p0) @ v)
        cnt = int((d < thresh).sum())
        if cnt > best_inliers:
            best_inliers, best_normal = cnt, v
    if best_normal is None:
        return None, 0.0
    # refine on the inlier set: least squares (SVD) about their centroid
    d = np.abs((pts - pts[0]) @ best_normal)
    inl = pts[np.abs((pts - pts.mean(0)) @ best_normal) < thresh]
    if len(inl) >= 50:
        c = inl.mean(0)
        _, _, vh = np.linalg.svd(inl - c)
        best_normal = vh[2]
    return best_normal, best_inliers / n


def angle_from_normal(normal):
    """Angle (deg) between the plane normal and the optical axis (0,0,1).

    The normal's sign is arbitrary from a fit, so fold to [0, 90]: we want the
    angle of the SURFACE, not which way the normal happens to point.
    """
    c = abs(float(normal[2])) / (np.linalg.norm(normal) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def measure(pipeline, intr, depth_scale, central=0.6, step=4,
            zmin=0.16, zmax=3.0, rng=None):
    """One measurement. Returns dict or None if the frame is unusable."""
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth:
        return None
    z = np.asanyarray(depth.get_data()).astype(np.float32) * depth_scale
    h, w = z.shape

    # central window only -- edges are noisier and may see past the target
    mh, mw = int(h * (1 - central) / 2), int(w * (1 - central) / 2)
    sub = z[mh:h - mh:step, mw:w - mw:step]
    ys, xs = np.mgrid[mh:h - mh:step, mw:w - mw:step]

    ok = (sub > zmin) & (sub < zmax)
    if ok.sum() < 200:
        return None
    zz = sub[ok]; xx = xs[ok]; yy = ys[ok]
    # deproject to camera-frame metres
    X = (xx - intr.ppx) / intr.fx * zz
    Y = (yy - intr.ppy) / intr.fy * zz
    pts = np.stack([X, Y, zz], axis=1)

    normal, frac = fit_plane_ransac(pts, rng=rng)
    if normal is None:
        return None
    return dict(angle_deg=angle_from_normal(normal),
                normal=normal, inlier_frac=frac,
                median_depth=float(np.median(zz)), n_points=len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="frames to average over (median)")
    ap.add_argument("--watch", action="store_true",
                    help="measure continuously until ctrl-C")
    ap.add_argument("--central", type=float, default=0.6,
                    help="fraction of the frame to use, centred")
    a = ap.parse_args()

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (profile.get_stream(rs.stream.depth)
            .as_video_stream_profile().get_intrinsics())
    rng = np.random.default_rng(0)

    # let auto-exposure settle
    for _ in range(15):
        pipeline.wait_for_frames()

    print("=" * 60)
    print("GROUND-TRUTH APPROACH ANGLE (depth plane fit)")
    print("=" * 60)
    print(f"  intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")
    print(f"  0 deg = frontal (surface normal along the optical axis)")
    print()

    try:
        if a.watch:
            print(f"{'angle_deg':>10} {'depth_m':>9} {'inliers':>8} {'pts':>7}")
            while True:
                m = measure(pipeline, intr, depth_scale, a.central, rng=rng)
                if m is None:
                    print("   (no usable depth -- too close, or untextured?)")
                else:
                    print(f"{m['angle_deg']:10.2f} {m['median_depth']:9.3f} "
                          f"{m['inlier_frac']:8.2f} {m['n_points']:7d}")
                time.sleep(0.2)
        else:
            vals, deps, fracs = [], [], []
            for _ in range(a.n):
                m = measure(pipeline, intr, depth_scale, a.central, rng=rng)
                if m is not None:
                    vals.append(m["angle_deg"]); deps.append(m["median_depth"])
                    fracs.append(m["inlier_frac"])
            if not vals:
                print("  NO USABLE MEASUREMENT.")
                print("  Depth may be out of range (D415 minimum ~0.16 m), the")
                print("  surface may be untextured/reflective, or the target may")
                print("  not fill the central region.")
                return
            v = np.array(vals)
            print(f"  APPROACH ANGLE = {np.median(v):.2f} deg"
                  f"   (spread {v.min():.2f}-{v.max():.2f}, "
                  f"sd {v.std():.2f}, n={len(v)})")
            print(f"  surface distance = {np.median(deps):.3f} m")
            print(f"  plane fit quality = {np.median(fracs):.2f} inlier fraction")
            if np.median(fracs) < 0.5:
                print("  WARNING: low inlier fraction -- the view may not be a")
                print("  single flat surface. Treat the angle with caution.")
            if v.std() > 2.0:
                print("  WARNING: high spread between frames -- noisy depth.")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()