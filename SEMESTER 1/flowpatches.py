import cv2
import numpy as np
import argparse
import time
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Command Line Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='Path to input video file')
parser.add_argument('--output', type=str, help='Path to save output video file')
parser.add_argument('--window', type=int, default=5, help='Patch window size around each keypoint (e.g. 5 = 5x5, 9 = 9x9)')
args = parser.parse_args()

# ─────────────────────────────────────────────
# Input Source
# ─────────────────────────────────────────────
if args.video:
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        exit()
    cap = cv2.VideoCapture(args.video)
    print(f"[INFO] Loading video file: {args.video}")
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] Using live webcam")

# ─────────────────────────────────────────────
# Output Video Writer
# ─────────────────────────────────────────────
writer = None
if args.output:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"[INFO] Saving output to: {args.output}")

# ─────────────────────────────────────────────
# Load YOLO Pose Model
# ─────────────────────────────────────────────
print("[INFO] Loading YOLO Pose model...")
model = YOLO("yolov8n-pose.pt")
print("[INFO] YOLO Pose model loaded!")

# ─────────────────────────────────────────────
# Optical Flow Setup
# ─────────────────────────────────────────────
ret, prev_frame = cap.read()
if not ret:
    print("[ERROR] Could not read from input source.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
h, w = prev_frame.shape[:2]

# Half the window size — used to calculate patch boundaries
half_win = args.window // 2

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Skeleton connections — pairs of keypoint indices to draw lines between
skeleton = [
    [0, 1], [0, 2],       # nose to eyes
    [1, 3], [2, 4],       # eyes to ears
    [5, 6],               # shoulders
    [5, 7], [7, 9],       # left arm
    [6, 8], [8, 10],      # right arm
    [5, 11], [6, 12],     # shoulders to hips
    [11, 12],             # hips
    [11, 13], [13, 15],   # left leg
    [12, 14], [14, 16],   # right leg
]

print(f"[INFO] Window size: {args.window}x{args.window} pixels around each keypoint")
print("[INFO] Press Q to quit | Press S to save photo | Press SPACE to pause")

paused      = False
frame_count = 0

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ─────────────────────────────────────
        # Step 1 — Run YOLO Pose
        # ─────────────────────────────────────
        yolo_start = time.time()
        results = model(frame, verbose=False)
        yolo_time = (time.time() - yolo_start) * 1000

        # Collect all keypoints from all detected people
        all_keypoints = []

        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()

                for person_kps in keypoints:

                    # Draw skeleton lines
                    for pair in skeleton:
                        pt1 = person_kps[pair[0]]
                        pt2 = person_kps[pair[1]]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame,
                                     (int(pt1[0]), int(pt1[1])),
                                     (int(pt2[0]), int(pt2[1])),
                                     (0, 255, 255), 2)  # yellow lines

                    # Draw joint dots and collect keypoint locations
                    for kp in person_kps:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0:
                            cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)  # red dots
                            all_keypoints.append((kx, ky))

        # ─────────────────────────────────────
        # Step 2 — Draw patch boxes on each keypoint
        # ─────────────────────────────────────
        for (kx, ky) in all_keypoints:
            # Calculate patch boundaries — clamp to frame edges
            x1 = max(0, kx - half_win)
            y1 = max(0, ky - half_win)
            x2 = min(w - 1, kx + half_win)
            y2 = min(h - 1, ky + half_win)

            # Draw the patch box so you can see where the window is
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)  # blue box

        # ─────────────────────────────────────
        # Step 3 — Calculate Optical Flow only inside each patch
        # ─────────────────────────────────────
        flow_start = time.time()

        for (kx, ky) in all_keypoints:
            # Calculate patch boundaries — clamp to frame edges
            x1 = max(0, kx - half_win)
            y1 = max(0, ky - half_win)
            x2 = min(w - 1, kx + half_win)
            y2 = min(h - 1, ky + half_win)

            # Skip if patch is too small
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            # Extract the patch region from previous and current frames
            prev_patch = prev_gray[y1:y2, x1:x2]
            curr_patch = curr_gray[y1:y2, x1:x2]

            # Create a grid of tracking points within this patch
            patch_h, patch_w = prev_patch.shape[:2]
            patch_points = np.array(
                [[x1 + x, y1 + y]
                 for y in range(0, patch_h, max(1, patch_h // 3))
                 for x in range(0, patch_w, max(1, patch_w // 3))],
                dtype=np.float32
            ).reshape(-1, 1, 2)

            if len(patch_points) == 0:
                continue

            # Calculate Lucas-Kanade flow on just these patch points
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, patch_points, None, **lk_params
            )

            # Draw arrows for each successfully tracked point
            for i, (new, old) in enumerate(zip(new_points, patch_points)):
                if status[i][0] == 1:
                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()
                    dx = x_new - x_old
                    dy = y_new - y_old
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 1.0:
                        cv2.arrowedLine(frame,
                                        (int(x_old), int(y_old)),
                                        (int(x_new), int(y_new)),
                                        (0, 255, 0), 1, tipLength=0.4)  # green arrows

        flow_time = (time.time() - flow_start) * 1000

        # ─────────────────────────────────────
        # HUD Overlay
        # ─────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"Mode: Pose-Centred Optical Flow Patches",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Window Size: {args.window}x{args.window} px | Keypoints: {len(all_keypoints)}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(frame, f"YOLO: {yolo_time:.1f}ms | Flow: {flow_time:.1f}ms | Frame: {frame_count}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Legend
        cv2.putText(frame, "Yellow=Skeleton  Red=Joints  Blue=Patch  Green=Flow",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Bottom bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 30), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Q: Quit  |  S: Save Photo  |  SPACE: Pause",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ─────────────────────────────────────
        # Update and Show
        # ─────────────────────────────────────
        prev_gray = curr_gray.copy()

        if writer:
            writer.write(frame)

        cv2.imshow("Pose-Centred Optical Flow Patches", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"patch_screenshot_{frame_count}.jpg", frame)
        print(f"[INFO] Screenshot saved: patch_screenshot_{frame_count}.jpg")
    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'}")

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
if writer:
    writer.release()
    print(f"[INFO] Output saved to: {args.output}")
cv2.destroyAllWindows()
print("[INFO] Done.")