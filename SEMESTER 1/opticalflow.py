import cv2
import numpy as np
import argparse
import time
import os

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--camera', action='store_true', help='Use live webcam as input')
parser.add_argument('--video', type=str, help='Path to input video file')
parser.add_argument('--output', type=str, help='Path to save output video file')
parser.add_argument('--flow', type=str, default='lk', choices=['lk', 'farneback'], help='Optical flow method: lk or farneback')
args = parser.parse_args()

# Input source
if args.video:
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        exit()
    cap = cv2.VideoCapture(args.video)
    print(f"[INFO] Loading video file: {args.video}")
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] Using live webcam")

# Output video writer setup
writer = None
if args.output:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"[INFO] Saving output to: {args.output}")

# Log file setup
log_filename = f"flow_log_{args.flow}.txt"
log_file = open(log_filename, 'w')
log_file.write(f"Optical Flow Timing Log\n")
log_file.write(f"Method: {args.flow}\n")
log_file.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write(f"{'Frame':<10} {'Time (ms)':<15} {'FPS':<10}\n")
log_file.write("-" * 40 + "\n")
print(f"[INFO] Logging times to: {log_filename}")

# Read the first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Grid of points for Lucas-Kanade
h, w = prev_frame.shape[:2]
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

print(f"[INFO] Optical flow method: {args.flow}")
print("[INFO] Press Q to quit | Press S to save photo | Press SPACE to pause")

paused = False
frame_count = 0

while True:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── LUCAS-KANADE METHOD ──
        if args.flow == 'lk':

            # Start timer
            start_time = time.time()

            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, points, None, **lk_params
            )

            # Stop timer
            elapsed_ms = (time.time() - start_time) * 1000
            fps_flow = 1000 / elapsed_ms if elapsed_ms > 0 else 0

            # Draw arrows
            for i, (new, old) in enumerate(zip(new_points, points)):
                if status[i][0] == 1:
                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()
                    dx = x_new - x_old
                    dy = y_new - y_old
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 1.5:
                        cv2.arrowedLine(
                            frame,
                            (int(x_old), int(y_old)),
                            (int(x_new), int(y_new)),
                            (0, 255, 0), 1, tipLength=0.3
                        )

            # Reset points grid
            points = np.array([[x, y] for y in range(0, h, step)
                                       for x in range(0, w, step)],
                               dtype=np.float32).reshape(-1, 1, 2)

        # ── FARNEBACK METHOD ──
        elif args.flow == 'farneback':

            # Start timer
            start_time = time.time()

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Stop timer
            elapsed_ms = (time.time() - start_time) * 1000
            fps_flow = 1000 / elapsed_ms if elapsed_ms > 0 else 0

            # Draw arrows
            for y in range(0, h, step):
                for x in range(0, w, step):
                    dx, dy = flow[y, x]
                    magnitude = np.sqrt(dx**2 + dy**2)
                    if magnitude > 1.5:
                        cv2.arrowedLine(
                            frame,
                            (x, y),
                            (int(x + dx), int(y + dy)),
                            (0, 255, 0), 1, tipLength=0.3
                        )

        # Write timing to log file
        log_file.write(f"{frame_count:<10} {elapsed_ms:<15.2f} {fps_flow:<10.1f}\n")

        # Show timing on screen
        cv2.putText(frame, f"Method: {args.flow.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {elapsed_ms:.1f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps_flow:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update previous frame
        prev_gray = curr_gray.copy()

        if writer:
            writer.write(frame)

        cv2.imshow("Optical Flow Demo", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("photo.jpg", frame)
        print("Photo saved!")
    elif key == ord(' '):
        paused = not paused
        print(f"[INFO] {'Paused' if paused else 'Resumed'}")

cap.release()
if writer:
    writer.release()
    print(f"[INFO] Output saved to: {args.output}")
log_file.close()
print(f"[INFO] Timing log saved to: {log_filename}")
cv2.destroyAllWindows()