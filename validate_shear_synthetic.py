#!/usr/bin/env python3
"""
validate_shear_synthetic.py

Validate the shear estimator against GROUND TRUTH that cannot be argued with.

WHY
---
Bench characterisation has produced contradictory answers (positive correlation
before an anisotropy fix, negative after), and too many things differ between
runs to isolate the cause: camera mounting, aspect-ratio resize, angle labelling,
surface texture, lighting, range, speed. This harness removes all of them.

It SYNTHESISES the image motion produced by a camera translating toward a plane
of KNOWN obliquity, feeds the resulting frame pairs through the real
MaxDivProcessor, and reports measured shear vs true obliquity. If the estimator
is correct, |s| must be ~0 at frontal and rise monotonically with tilt.

THE GEOMETRY
------------
Camera at the origin looking along +Z, translating with velocity (0,0,Vz).
A plane with unit normal n at distance d satisfies  n . X = d.
For image point (x,y) in normalised camera coords, the ray X = Z*(x,y,1) meets
the plane at
        Z(x,y) = d / (n_x*x + n_y*y + n_z)
The optical-flow field for pure translation T=(0,0,Vz) is
        u = (Tz*x)/Z ,  v = (Tz*y)/Z            (normalised coords)
so the DEPTH GRADIENT induced by plane tilt makes the expansion anisotropic --
which is exactly the shear the controller is meant to regulate. At n=(0,0,1)
(frontal) Z is constant, the flow is a pure isotropic expansion, and shear=0.

We warp a textured image by this flow to make frame 2, so the LK tracker sees a
physically consistent pair.

USAGE
    python validate_shear_synthetic.py
    python validate_shear_synthetic.py --process-size 640x360 --src-size 640x480
    python validate_shear_synthetic.py --flip-v      # simulate an inverted mount
"""
import argparse
import sys

import numpy as np
import cv2

sys.path.insert(0, ".")
from maxdiv_perception import MaxDivProcessor


def textured(w, h, seed=0):
    """Dense aperiodic texture so LK has plenty to track."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # add some larger-scale structure too
    big = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    big = cv2.resize(big, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.addWeighted(img, 0.5, big, 0.5, 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def plane_flow(w, h, fx, fy, cx, cy, tilt_deg, tilt_azimuth_deg,
               d=0.5, Vz=0.9, dt=0.05):
    """Per-pixel flow (du,dv) in PIXELS for a camera translating toward a plane.

    tilt_deg is the angle between the plane normal and the optical axis
    (0 = frontal). tilt_azimuth_deg rotates the tilt direction in the image.
    """
    t = np.radians(tilt_deg)
    a = np.radians(tilt_azimuth_deg)
    n = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = (xs - cx) / fx
    yn = (ys - cy) / fy

    denom = n[0] * xn + n[1] * yn + n[2]
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    Z = d / denom                                   # depth at each pixel

    # flow for pure Z translation, in normalised coords, then to pixels
    du_n = (Vz * dt) * xn / Z
    dv_n = (Vz * dt) * yn / Z
    return du_n * fx, dv_n * fy, Z


def warp_by_flow(img, du, dv):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mapx = (xs - du).astype(np.float32)
    mapy = (ys - dv).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def run(process_size, src_size, flip_v, flip_u, angles, azimuth, reps):
    sw, sh = src_size
    pw, ph = process_size
    # plausible D415-ish intrinsics for the SOURCE resolution
    fx = fy = 607.0
    cx, cy = sw / 2.0, sh / 2.0

    print("=" * 68)
    print("SYNTHETIC SHEAR VALIDATION")
    print("=" * 68)
    print(f"  source frames : {sw}x{sh}")
    print(f"  process size  : {pw}x{ph}"
          f"{'   (ANISOTROPIC resize)' if abs(pw/sw - ph/sh) > 1e-6 else '   (isotropic)'}")
    print(f"  vertical flip : {flip_v}    horizontal flip: {flip_u}")
    print(f"  tilt azimuth  : {azimuth} deg")
    print()
    print(f"{'tilt':>6} {'divmax':>10} {'shear':>10} {'|s|=shear/div':>14} {'nvec':>6}")

    out = []
    for ang in angles:
        rows = []
        for rep in range(reps):
            base = textured(sw, sh, seed=rep)
            du, dv, Z = plane_flow(sw, sh, fx, fy, cx, cy, ang, azimuth)
            du = np.nan_to_num(du); dv = np.nan_to_num(dv)
            f2 = warp_by_flow(base, du, dv)

            f1, f2b = base, f2
            if flip_v:
                f1 = cv2.flip(f1, 0); f2b = cv2.flip(f2b, 0)
            if flip_u:
                f1 = cv2.flip(f1, 1); f2b = cv2.flip(f2b, 1)

            p = MaxDivProcessor(dict(process_size=(pw, ph), skip=40))
            p.reset()
            p.process(f1, 0.05)
            # feed the pair a few times so temporal smoothing settles
            r = None
            for _ in range(6):
                r = p.process(f2b, 0.05)
            if r is None or not np.isfinite(r.divmax_raw) or r.divmax_raw <= 0:
                continue
            rows.append((r.divmax_raw, r.axis_strength,
                         r.axis_strength / r.divmax_raw, r.n_vectors))
        if not rows:
            print(f"{ang:6.0f}  (no valid frames)")
            continue
        a = np.array(rows)
        print(f"{ang:6.0f} {np.median(a[:,0]):10.5f} {np.median(a[:,1]):10.5f} "
              f"{np.median(a[:,2]):14.3f} {np.median(a[:,3]):6.0f}")
        out.append((ang, np.median(a[:, 2])))

    if len(out) >= 3:
        angs = [o[0] for o in out]; s = [o[1] for o in out]
        corr = np.corrcoef(angs, s)[0, 1]
        slope = np.polyfit(angs, s, 1)[0]
        print()
        print(f"  correlation(tilt, |s|) = {corr:+.3f}")
        print(f"  slope                  = {slope:+.5f} |s| per degree")
        print(f"  |s| at frontal         = {s[0]:.4f}")
        if corr > 0.9 and s[0] < 0.05:
            print("  VERDICT: estimator behaves correctly "
                  "(monotone, ~zero at frontal).")
        elif corr < -0.5:
            print("  VERDICT: INVERTED -- shear DECREASES with tilt. Sign error.")
        elif s[0] > 0.05:
            print("  VERDICT: nonzero shear at FRONTAL -- spurious-shear artifact"
                  " still present.")
        else:
            print("  VERDICT: weak/!monotone response -- estimator not resolving tilt.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--process-size", default="640x360")
    ap.add_argument("--src-size", default="640x480")
    ap.add_argument("--flip-v", action="store_true",
                    help="simulate a vertically inverted camera mount")
    ap.add_argument("--flip-u", action="store_true")
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="direction of the surface tilt in the image (deg)")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ps = tuple(int(v) for v in a.process_size.lower().split("x"))
    ss = tuple(int(v) for v in a.src_size.lower().split("x"))
    run(ps, ss, a.flip_v, a.flip_u, [0, 10, 20, 30, 40, 50], a.azimuth, a.reps)
