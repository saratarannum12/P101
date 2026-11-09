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
parser.add_argument('--camera', action='store_true', help='Use live webcam as input')
parser.add_argument('--video', type=str, help='Path to input video file')
parser.add_argument('--output', type=str, help='Path to save output video file')
parser.add_argument('--flow', type=str, default='lk', choices=['lk', 'farneback', 'none'], help='Optical flow method: lk, farneback, or none')
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

# Grid of points for Lucas-Kanade
step = 20
points = np.array([[x, y] for y in range(0, h, step)
                           for x in range(0, w, step)],
                   dtype=np.float32).reshape(-1, 1, 2)

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# ─────────────────────────────────────────────
# Timing Log Setup
# ─────────────────────────────────────────────
if args.flow != 'none':
    log_filename = f"flow_log_{args.flow}.txt"
    log_file = open(log_filename, 'w')
    log_file.write(f"Optical Flow Timing Log\n")
    log_file.write(f"Method: {args.flow}\n")
    log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'Frame':<10} {'Flow Time (ms)':<20} {'YOLO Time (ms)':<20} {'FPS':<10}\n")
    log_file.write("-" * 60 + "\n")
    print(f"[INFO] Logging times to: {log_filename}")

print(f"[INFO] Optical flow method: {args.flow}")
print("[INFO] Press Q to quit | Press S to save photo | Press SPACE to pause | Press F to toggle flow | Press Y to toggle YOLO")

# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
paused       = False
frame_count  = 0
show_yolo    = True   # toggle YOLO skeleton on/off with Y
show_flow    = True   # toggle optical flow on/off with F
elapsed_flow = 0
elapsed_yolo = 0

while True:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ─────────────────────────────────────
        # YOLO Pose
        # ─────────────────────────────────────
        if show_yolo:
            yolo_start = time.time()

            results = model(frame, verbose=False)

            # Draw skeleton on frame
            # We draw manually so we can still see the optical flow arrows on top
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    
                    # Skeleton connections - pairs of keypoint indices to connect with lines
                    skeleton = [
                        [0, 1], [0, 2],           # nose to eyes
                        [1, 3], [2, 4],           # eyes to ears
                        [5, 6],                   # shoulders
                        [5, 7], [7, 9],           # left arm
                        [6, 8], [8, 10],          # right arm
                        [5, 11], [6, 12],         # shoulders to hips
                        [11, 12],                 # hips
                        [11, 13], [13, 15],       # left leg
                        [12, 14], [14, 16],       # right leg
                    ]

                    for person_kps in keypoints:
                        # Draw lines between connected joints
                        for pair in skeleton:
                            pt1 = person_kps[pair[0]]
                            pt2 = person_kps[pair[1]]
                            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                cv2.line(frame,
                                         (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])),
                                         (0, 255, 255), 2)  # yellow lines

                        # Draw dots on each joint
                        for kp in person_kps:
                            if kp[0] > 0 and kp[1] > 0:
                                cv2.circle(frame,
                                           (int(kp[0]), int(kp[1])),
                                           4, (0, 0, 255), -1)  # red dots

            elapsed_yolo = (time.time() - yolo_start) * 1000

        # ─────────────────────────────────────
        # Optical Flow
        # ─────────────────────────────────────
        if show_flow and args.flow != 'none':

            # Lucas-Kanade
            if args.flow == 'lk':
                flow_start = time.time()

                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, points, None, **lk_params
                )

                elapsed_flow = (time.time() - flow_start) * 1000

                for i, (new, old) in enumerate(zip(new_points, points)):
                    if status[i][0] == 1:
                        x_new, y_new = new.ravel()
                        x_old, y_old = old.ravel()
                        dx = x_new - x_old
                        dy = y_new - y_old
                        magnitude = np.sqrt(dx**2 + dy**2)
                        if magnitude > 1.5:
                            cv2.arrowedLine(frame,
                                            (int(x_old), int(y_old)),
                                            (int(x_new), int(y_new)),
                                            (0, 255, 0), 1, tipLength=0.3)

                # Reset points grid
                points = np.array([[x, y] for y in range(0, h, step)
                                           for x in range(0, w, step)],
                                   dtype=np.float32).reshape(-1, 1, 2)

            # Farneback
            elif args.flow == 'farneback':
                flow_start = time.time()

                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                elapsed_flow = (time.time() - flow_start) * 1000

                for y in range(0, h, step):
                    for x in range(0, w, step):
                        dx, dy = flow[y, x]
                        magnitude = np.sqrt(dx**2 + dy**2)
                        if magnitude > 1.5:
                            cv2.arrowedLine(frame,
                                            (x, y),
                                            (int(x + dx), int(y + dy)),
                                            (0, 255, 0), 1, tipLength=0.3)

            # Write to log file
            fps_flow = 1000 / elapsed_flow if elapsed_flow > 0 else 0
            log_file.write(f"{frame_count:<10} {elapsed_flow:<20.2f} {elapsed_yolo:<20.2f} {fps_flow:<10.1f}\n")

        # ─────────────────────────────────────
        # HUD Overlay
        # ─────────────────────────────────────
        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Method and status info
        cv2.putText(frame, f"Flow: {args.flow.upper()} ({'ON' if show_flow else 'OFF'})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if show_flow else (100, 100, 100), 2)
        cv2.putText(frame, f"YOLO Pose: ({'ON' if show_yolo else 'OFF'})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if show_yolo else (100, 100, 100), 2)
        cv2.putText(frame, f"Flow: {elapsed_flow:.1f}ms | YOLO: {elapsed_yolo:.1f}ms",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Bottom controls bar
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 30), (w, h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "Q: Quit | S: Save Photo | SPACE: Pause | F: Toggle Flow | Y: Toggle YOLO",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ─────────────────────────────────────
        # Update previous frame and show
        # ─────────────────────────────────────
        prev_gray = curr_gray.copy()

        if writer:
            writer.write(frame)

        cv2.imshow("PoseOFF Demonstrator", frame)

    # ─────────────────────────────────────────
    # Key Controls
    # ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"screenshot_{frame_count}.jpg", frame)
        print(f"[INFO] Screenshot saved: screenshot_{frame_count}.jpg")
    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'}")
    elif key == ord('f'):
        show_flow = not show_flow
        print(f"[INFO] Flow {'ON' if show_flow else 'OFF'}")
    elif key == ord('y'):
        show_yolo = not show_yolo
        print(f"[INFO] YOLO {'ON' if show_yolo else 'OFF'}")

# ─────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────
cap.release()
if writer:
    writer.release()
    print(f"[INFO] Output saved to: {args.output}")
if args.flow != 'none':
    log_file.close()
    print(f"[INFO] Timing log saved to: {log_filename}")
cv2.destroyAllWindows()
print("[INFO] Done.")