#!/usr/bin/env python3
"""
realsense_bench_test.py

Camera-only bench test of the Intel RealSense D415.

PURPOSE (mirror of the arm velocity bench test)
-----------------------------------------------
Prove the imaging path in isolation, before any flow, arm, or control loop:
  1. The D415 enumerates and the pipeline starts.
  2. Color streams cleanly at the requested resolution/rate.
  3. (Optional) Depth streams and aligns to color, so you have an independent
     per-pixel range for validating monocular TTC later.
  4. Read and PRINT the factory intrinsics (fx, fy, ppx, ppy, distortion),
     which your spherical-flow reprojection needs.
  5. Report the ACHIEVED framerate over a burst (not just the requested one).

No arm. No optical flow. No maxdiv. Just the camera.

REQUIREMENTS
------------
  pip install pyrealsense2 numpy
  (opencv-python only if you set SHOW_PREVIEW = True)

NOTES ON THE D415
-----------------
- USB 3 device into the PC. Nothing to do with the arm/control box.
- On 400-series cameras the color-stream intrinsics are the ones to use for
  a pinhole reprojection; the depth/IR intrinsics have distortion coeffs
  deliberately zeroed. This script reads the COLOR intrinsics for that reason,
  and also prints depth intrinsics + the depth->color extrinsic for reference.
- depth_scale converts raw depth units to metres; printed so you can log
  ground-truth range alongside your flow-derived TTC.
"""

import time
import sys

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("pyrealsense2 not installed. Run: pip install pyrealsense2")

# ---- CONFIG ---------------------------------------------------------------
COLOR_W, COLOR_H, COLOR_FPS = 640, 480, 60   # start conservative; D415 does this
ENABLE_DEPTH   = True     # also stream depth + align to color (validation aid)
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 480, 60
N_WARMUP       = 30       # frames to discard while auto-exposure settles
N_MEASURE      = 120      # frames to time for achieved-fps measurement
SHOW_PREVIEW   = False    # set True to open an OpenCV window (needs opencv-python)
# ---------------------------------------------------------------------------


def print_intrinsics(tag, intr):
    """intr is an rs.intrinsics."""
    print(f"\n[{tag} intrinsics]")
    print(f"  resolution : {intr.width} x {intr.height}")
    print(f"  fx, fy     : {intr.fx:.4f}, {intr.fy:.4f}")
    print(f"  ppx, ppy   : {intr.ppx:.4f}, {intr.ppy:.4f}")
    print(f"  model      : {intr.model}")
    print(f"  coeffs     : {['%.5f' % c for c in intr.coeffs]}")


def list_devices():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        sys.exit("No RealSense device found. Check the USB 3 connection.")
    print("Connected RealSense devices:")
    for d in devs:
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        fw = d.get_info(rs.camera_info.firmware_version)
        print(f"  - {name}  (S/N {serial}, FW {fw})")
    return devs


def main():
    list_devices()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H,
                         rs.format.bgr8, COLOR_FPS)
    if ENABLE_DEPTH:
        config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H,
                             rs.format.z16, DEPTH_FPS)

    print("\nStarting pipeline...")
    profile = pipeline.start(config)

    # depth scale (raw units -> metres)
    if ENABLE_DEPTH:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"depth_scale (m per unit): {depth_scale:.6f}")

    # aligner: bring depth into the color frame so pixels correspond
    align = rs.align(rs.stream.color) if ENABLE_DEPTH else None

    # --- intrinsics: read from the started stream profiles ---
    color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    print_intrinsics("COLOR", color_prof.get_intrinsics())

    if ENABLE_DEPTH:
        depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        print_intrinsics("DEPTH", depth_prof.get_intrinsics())
        # depth -> color extrinsic (rotation + translation)
        extr = depth_prof.get_extrinsics_to(color_prof)
        print("\n[depth -> color extrinsic]")
        print("  rotation   :", ['%.5f' % r for r in extr.rotation])
        print("  translation:", ['%.5f' % t for t in extr.translation])

    if SHOW_PREVIEW:
        import cv2

    try:
        # --- warmup (let auto-exposure settle) ---
        print(f"\nWarming up ({N_WARMUP} frames)...")
        for _ in range(N_WARMUP):
            pipeline.wait_for_frames()

        # --- measure achieved framerate over a burst ---
        print(f"Measuring achieved framerate over {N_MEASURE} frames...")
        n_ok = 0
        n_depth_ok = 0
        center_ranges = []
        t_start = time.time()

        for _ in range(N_MEASURE):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color = frames.get_color_frame()
            if not color:
                continue
            n_ok += 1
            color_img = np.asanyarray(color.get_data())

            if ENABLE_DEPTH:
                depth = frames.get_depth_frame()
                if depth:
                    n_depth_ok += 1
                    # sample the center pixel's range as a sanity check
                    cx, cy = COLOR_W // 2, COLOR_H // 2
                    d = depth.get_distance(cx, cy)   # metres
                    if d > 0:
                        center_ranges.append(d)

            if SHOW_PREVIEW:
                cv2.imshow("D415 color", color_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        elapsed = time.time() - t_start
        fps = n_ok / elapsed if elapsed > 0 else 0.0

        print("\n--- results ---")
        print(f"  color frames ok : {n_ok}/{N_MEASURE}")
        print(f"  elapsed         : {elapsed:.3f} s")
        print(f"  achieved fps    : {fps:.1f}  (requested {COLOR_FPS})")
        print(f"  frame shape     : {color_img.shape}  (H, W, C)")
        if ENABLE_DEPTH:
            print(f"  depth frames ok : {n_depth_ok}/{N_MEASURE}")
            if center_ranges:
                arr = np.array(center_ranges)
                print(f"  center range    : {arr.mean():.3f} m "
                      f"(min {arr.min():.3f}, max {arr.max():.3f}) "
                      f"over {len(arr)} valid samples")
            else:
                print("  center range    : no valid depth at center "
                      "(too close/far, or pointed at nothing)")

        print("\nCamera bench test complete.")

    except KeyboardInterrupt:
        print("\nAborted by user (Ctrl-C).")
    finally:
        pipeline.stop()
        if SHOW_PREVIEW:
            import cv2
            cv2.destroyAllWindows()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()
