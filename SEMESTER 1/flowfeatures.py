import cv2
import numpy as np
import argparse
import time
import os
import csv
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Command Line Arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, help='Path to input video file')
parser.add_argument('--output', type=str, help='Path to save output video file')
parser.add_argument('--window', type=int, default=5, help='Patch window size around each keypoint')
parser.add_argument('--skip', type=int, default=0, help='Pixel skip between tracked points. 0=every pixel, 1=every 2nd, 2=every 3rd')
parser.add_argument('--csv', type=str, default='features.csv', help='Path to save CSV feature output')
parser.add_argument('--action', type=str, default='unknown', help='Label for the action being performed')
args = parser.parse_args()

# ─────────────────────────────────────────────
# Joint Names
# ─────────────────────────────────────────────
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

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
# CSV Setup
# ─────────────────────────────────────────────
csv_file = open(args.csv, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'action', 'frame', 'joint_id', 'joint_name',
    'x', 'y',
    'mean_flow_x', 'mean_flow_y',
    'mean_magnitude', 'divergence'
])
print(f"[INFO] Saving features to: {args.csv}")

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

half_win = args.window // 2
step = args.skip + 1

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Skeleton connections
skeleton = [
    [0, 1], [0, 2],
    [1, 3], [2, 4],
    [5, 6],
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    [5, 11], [6, 12],
    [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
]

print(f"[INFO] Window: {args.window}x{args.window} | Skip: {args.skip} | Step: {step}px | Action: {args.action}")
print("[INFO] Press Q to quit | Press S to save screenshot | Press SPACE to pause")

paused = False
frame_count = 0

# ─────────────────────────────────────────────
# Feature Extraction Function
# ─────────────────────────────────────────────
def extract_features(prev_gray, curr_gray, kx, ky, half_win, step, w, h, lk_params):
    """
    For a patch centred on keypoint (kx, ky):
    1. Build a grid of tracking points using step size (controlled by skip)
    2. Run Lucas-Kanade to get flow vectors at those points
    3. Compute mean_flow_x, mean_flow_y, mean_magnitude
    4. Compute divergence using Sobel filter: u_x + v_y
       - u_x = Sobel gradient of U (horizontal flow) in x direction
       - v_y = Sobel gradient of V (vertical flow) in y direction
       - sum across the patch to get one divergence value
    """

    # Clamp patch boundaries to frame edges
    x1 = max(0, kx - half_win)
    y1 = max(0, ky - half_win)
    x2 = min(w - 1, kx + half_win)
    y2 = min(h - 1, ky + half_win)

    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    patch_h = y2 - y1
    patch_w = x2 - x1

    # Build grid of tracking points using step size
    patch_points = np.array(
        [[x1 + x, y1 + y]
         for y in range(0, patch_h, step)
         for x in range(0, patch_w, step)],
        dtype=np.float32
    ).reshape(-1, 1, 2)

    if len(patch_points) == 0:
        return None

    # Run Lucas-Kanade on patch points
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, patch_points, None, **lk_params
    )

    # Collect valid flow vectors
    flow_u = []
    flow_v = []
    valid_old = []
    valid_new = []

    for i, (new, old) in enumerate(zip(new_points, patch_points)):
        if status[i][0] == 1:
            dx = float(new[0][0] - old[0][0])
            dy = float(new[0][1] - old[0][1])
            flow_u.append(dx)
            flow_v.append(dy)
            valid_old.append(old)
            valid_new.append(new)

    if len(flow_u) == 0:
        return None

    u = np.array(flow_u)
    v = np.array(flow_v)

    # ── Mean flow x and y
    mean_flow_x = float(np.mean(u))
    mean_flow_y = float(np.mean(v))

    # ── Mean magnitude
    magnitude = np.sqrt(u**2 + v**2)
    mean_magnitude = float(np.mean(magnitude))

    # ── Divergence using Sobel filter (u_x + v_y)
    # We need to reshape u and v into a 2D grid to apply Sobel
    grid_rows = len(range(0, patch_h, step))
    grid_cols = len(range(0, patch_w, step))

    if grid_rows >= 3 and grid_cols >= 3 and len(u) == grid_rows * grid_cols:
        # Reshape into 2D grids
        u_grid = u.reshape(grid_rows, grid_cols).astype(np.float32)
        v_grid = v.reshape(grid_rows, grid_cols).astype(np.float32)

        # Apply Sobel filter to get gradients
        # u_x = gradient of horizontal flow in the x (horizontal) direction
        u_x = cv2.Sobel(u_grid, cv2.CV_64F, 1, 0, ksize=3)

        # v_y = gradient of vertical flow in the y (vertical) direction
        v_y = cv2.Sobel(v_grid, cv2.CV_64F, 0, 1, ksize=3)

        # Sum u_x + v_y across the whole patch to get divergence
        divergence = float(np.sum(u_x + v_y))
    else:
        # Not enough points for Sobel — set to zero
        divergence = 0.0

    return {
        'mean_flow_x': round(mean_flow_x, 4),
        'mean_flow_y': round(mean_flow_y, 4),
        'mean_magnitude': round(mean_magnitude, 4),
        'divergence': round(divergence, 4),
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'valid_old': valid_old,
        'valid_new': valid_new
    }

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Run YOLO Pose
        yolo_start = time.time()
        results = model(frame, verbose=False)
        yolo_time = (time.time() - yolo_start) * 1000

        flow_start = time.time()

        for result in results:
            if result.keypoints is None:
                continue

            keypoints = result.keypoints.xy.cpu().numpy()

            for person_kps in keypoints:

                # Draw skeleton
                for pair in skeleton:
                    pt1 = person_kps[pair[0]]
                    pt2 = person_kps[pair[1]]
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame,
                                 (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0]), int(pt2[1])),
                                 (0, 255, 255), 2)

                # ── Step 2: For each joint extract features and write to CSV
                for joint_id, kp in enumerate(person_kps):
                    kx, ky = int(kp[0]), int(kp[1])

                    if kx <= 0 or ky <= 0:
                        continue

                    # Draw joint dot
                    cv2.circle(frame, (kx, ky), 4, (0, 0, 255), -1)

                    # Extract features
                    features = extract_features(
                        prev_gray, curr_gray, kx, ky,
                        half_win, step, w, h, lk_params
                    )

                    if features is None:
                        continue

                    # ── Draw patch box
                    # Colour based on divergence
                    # Red = positive divergence (expanding = toward camera)
                    # Blue = negative divergence (contracting = away from camera)
                    # Orange = neutral
                    div = features['divergence']
                    if div > 0.05:
                        box_colour = (0, 0, 255)
                    elif div < -0.05:
                        box_colour = (255, 0, 0)
                    else:
                        box_colour = (255, 100, 0)

                    cv2.rectangle(frame,
                                  (features['x1'], features['y1']),
                                  (features['x2'], features['y2']),
                                  box_colour, 1)

                    # ── Draw flow arrows inside patch
                    for old, new in zip(features['valid_old'], features['valid_new']):
                        x_old, y_old = old.ravel()
                        x_new, y_new = new.ravel()
                        dx = x_new - x_old
                        dy = y_new - y_old
                        magnitude = np.sqrt(dx**2 + dy**2)
                        if magnitude > 1.0:
                            cv2.arrowedLine(frame,
                                            (int(x_old), int(y_old)),
                                            (int(x_new), int(y_new)),
                                            (0, 255, 0), 1, tipLength=0.4)

                    # ── Write to CSV
                    joint_name = JOINT_NAMES[joint_id] if joint_id < len(JOINT_NAMES) else f"joint_{joint_id}"
                    csv_writer.writerow([
                        args.action,
                        frame_count,
                        joint_id,
                        joint_name,
                        kx,
                        ky,
                        features['mean_flow_x'],
                        features['mean_flow_y'],
                        features['mean_magnitude'],
                        features['divergence']
                    ])

        flow_time = (time.time() - flow_start) * 1000

        # ── HUD Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 115), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"Action: {args.action.upper()}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, f"Window: {args.window}x{args.window}px | Skip: {args.skip} | Step: {step}px",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(frame, f"YOLO: {yolo_time:.1f}ms | Flow: {flow_time:.1f}ms | Frame: {frame_count}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
        cv2.putText(frame, f"CSV: {args.csv}",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(frame, "Red=toward cam  Blue=away  Orange=neutral  Green=flow arrows",
                    (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Bottom bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 30), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Q: Quit  |  S: Screenshot  |  SPACE: Pause",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        prev_gray = curr_gray.copy()

        if writer:
            writer.write(frame)

        cv2.imshow("Pose-Centred Optical Flow Patches", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"patch_screenshot_{args.action}_{frame_count}.jpg", frame)
        print(f"[INFO] Screenshot saved: patch_screenshot_{args.action}_{frame_count}.jpg")
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
csv_file.close()
print(f"[INFO] Features saved to: {args.csv}")
cv2.destroyAllWindows()
print("[INFO] Done.")