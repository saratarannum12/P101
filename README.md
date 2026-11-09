# Max-Div Alignment Validator

A Python tool that wraps Chris McCarthy's MaxDivProcessor algorithm to display and validate the max-div shear cue on recorded video footage or a live Intel RealSense D415 depth camera stream.

The tool's job is to **display and validate** the existing max-div shear cue. It does not re-implement or improve the perception engine. Everything here is camera-only. 

---

## Table of Contents

1. [What the Tool Does](#1-what-the-tool-does)
2. [File Overview](#2-file-overview)
3. [How Files Map to Project Milestones](#3-how-files-map-to-project-milestones)
4. [Setup and Installation](#4-setup-and-installation)
5. [Running the Tool](#5-running-the-tool)
6. [Output Files](#6-output-files)
7. [Known Limitations](#7-known-limitations)

---

## 1. What the Tool Does

The Max-Div Alignment Validator takes video footage (or a live camera stream) and runs it through the MaxDivProcessor perception engine. It then displays an overlay showing:

- The divergence heatmap and max-div point
- A proximity readout (time-to-contact in seconds)
- A direction-to-frontal arrow derived from the shear cue
- A confidence flag indicating whether the surface is near-frontal (the ill-conditioned case where the shear estimate becomes unreliable)

It also supports a validation mode where you provide the true azimuth angle of the surface and the tool computes per-frame angular error, saves a summary chart, and exports the results to an Excel workbook for further analysis.

---

## 2. File Overview

### Files You Provided (Do Not Modify)

**`maxdiv_perception.py`**
The core perception engine. Contains the `MaxDivProcessor` class, which takes consecutive video frames and a measured time-step and returns a `Result` object containing proximity, divergence, max-div point, and shear cue data. This is the single source of truth for all perception outputs. The validator tool calls it and reads its `Result`. It is never modified.

**`maxdiv_viz.py`**
The existing overlay renderer provided with the project. Draws the divergence heatmap, max-div point marker, and shear arc onto a frame using the `Result` object from `maxdiv_perception.py`. Used directly in the video and live modes to produce the visual overlay.

**`realsense_cue.py`**
The camera-to-perception bridge for the Intel RealSense D415 depth camera. Wraps the RealSense stream and exposes `get_tau()` (time-to-contact in seconds, or None) and `get_last()` (the full Result object). Used by the live camera modes (c1.py and newc1.py).

**`measure_approach_angle.py`**
An independent ground-truth surface angle estimator that uses depth data from the RealSense camera. Fits a plane to the depth frame and returns the true surface angle. Used as a validation reference for real recorded clips. Never feed its output back into the perception cue -- depth is for validation only. The shear cue must remain purely monocular.

**`realsense_bench_test.py`**
A diagnostic script that proves a RealSense camera is streaming correctly. Prints camera intrinsics and frame rate. Run this first before attempting any live camera mode to confirm the camera is recognised and streaming at the expected rate.

**`cat.mov`**
A sample video file for testing the tool without needing a camera. Use this to verify the tool runs correctly on your machine before connecting hardware. Any standard video file (.mp4, .mov, .avi) works in the same way.

---

### Files Built for This Project

**`m3m4.py`**
The primary video file mode. Accepts a video file path and runs it through the MaxDivProcessor pipeline, displaying the overlay in an OpenCV window. Covers Milestones M1 (video playback), M2 (overlay visualisation), M3 (proximity readout), and M4 (direction-to-frontal arrow). This is the main entry point for processing recorded footage.

**`m2.py`**
An earlier standalone implementation of Milestone M2 (overlay visualisation). Developed during Sprint 4 to isolate and test the overlay integration before it was merged into the main pipeline. Retained for reference.

**`newm4.py`**
A refined version of the direction-to-frontal arrow (Milestone M4) with an updated direction panel design. The panel was redesigned during Sprint 5 for clarity after supervisor review. This version's panel layout is the one used in the final tool.

**`s2new.py`**
The S2 True-Azimuth Validation module. Accepts a video file and a known true azimuth angle (in degrees) provided by the user, runs the clip through the MaxDivProcessor, computes the per-frame angular error between the estimated direction and the true azimuth, and saves a PNG accuracy chart and a CSV file of per-frame results. This is the core validation capability of the tool.

**`s2summary.py`**
An extension of s2new.py that adds Excel export. After running S2 validation, this script produces an Excel workbook (.xlsx) with two sheets: a Frame Data sheet containing colour-coded per-frame error values, and a Summary sheet containing clip-level statistics (mean error, standard deviation, number of valid frames). Added at the client's request after the Sprint 6 demo. Use this version instead of s2new.py when you want the Excel output.

**`s3batchmode.py`**
The S3 Batch Processing mode. Accepts a folder of video clips and a manifest CSV file listing the true azimuth for each clip. Processes every clip automatically and writes a summary CSV with one row per clip showing mean error, standard deviation, and number of valid frames. Designed for overnight processing of large clip sets without manual intervention.

**`c1.py`**
The first implementation of the C1 Live Camera mode. Connects to an Intel RealSense D415 depth camera via realsense_cue.py and displays the overlay in real time. Falls back to a standard webcam if no RealSense device is detected.

**`newc1.py`**
The refined and final version of the C1 Live Camera mode. Incorporates the M5 confidence flag display, the updated direction panel from newm4.py, and the M5 confidence state logic (NEAR FRONTAL / SLIGHT TILT / SIGNIFICANT TILT). This is the version to use for live camera operation. c1.py is retained for reference only.

---

### Documentation Files

**`maxdiv_alignment_validator_spec.docx`**
The original project specification document provided at the start of the project. Defines all milestones (M1-M5), stretch deliverables (S1-S3), and could-tier deliverables (C1-C2) with acceptance criteria. Retained as a record of the original agreed scope.

**`README_alignment_validator.md`**
This file.

---

## 3. How Files Map to Project Milestones

| Milestone | Description | File(s) |
|-----------|-------------|---------|
| M1 | Video file playback in an OpenCV window | m3m4.py |
| M2 | Overlay visualisation (heatmap, max-div point, shear arc) | m3m4.py, m2.py (reference), maxdiv_viz.py |
| M3 | Proximity readout (time-to-contact in seconds) | m3m4.py |
| M4 | Direction-to-frontal arrow from shear cue | m3m4.py, newm4.py (refined panel) |
| M5 | Confidence flag (NEAR FRONTAL / SLIGHT TILT / SIGNIFICANT TILT) | newc1.py, m3m4.py |
| S2 | True-azimuth validation with per-frame error chart and CSV | s2new.py, s2summary.py |
| S2 (Excel) | Excel workbook export of per-frame and summary validation data | s2summary.py |
| S3 | Batch processing of a folder of clips against a manifest CSV | s3batchmode.py |
| C1 | Live camera mode via Intel RealSense D415 | newc1.py, c1.py (reference), realsense_cue.py |

**Not delivered:**

| Milestone | Description | Reason |
|-----------|-------------|--------|
| S1 | Synthetic clip generation with known ground truth | Deprioritised in Sprint 4 at Chris's direction. S2 (true-azimuth validation on real clips) is more directly useful for his experiment because his rig already produces footage with known camera angles. |
| C2 | Built-in clip generator using validate_shear_synthetic.py | Deprioritised in favour of S2 and S3, which deliver higher research value. validate_shear_synthetic.py remains in the codebase and can be used directly from the command line. |

---

## 4. Setup and Installation

Full step-by-step instructions for Windows are in the System Installation Manual. The quick version is below.

### Requirements

- Python 3.10 or later
- Windows 10 or later (the installation manual covers Windows only; macOS and Linux should work but are not documented)
- Intel RealSense D415 depth camera (only required for live camera modes c1.py and newc1.py)
- USB 3.0 port (required for RealSense; USB 2.0 will not provide sufficient bandwidth)

### Step 1: Create a virtual environment

```
python -m venv .venv
```

Activate it:

- Windows Command Prompt: `.venv\Scripts\activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS / Linux: `source .venv/bin/activate`

You should see `(.venv)` at the start of your terminal prompt once it is active.

### Step 2: Install dependencies

```
pip install numpy opencv-python matplotlib openpyxl
```

For live camera modes only (requires a connected RealSense D415):

```
pip install pyrealsense2
```

You do not need pyrealsense2 to run the video file modes (m3m4.py, s2new.py, s2summary.py, s3batchmode.py).

### Step 3: Verify the installation

Run the synthetic validation script to confirm the perception engine works on your machine without needing a camera:

```
python validate_shear_synthetic.py
```

This exercises the real MaxDivProcessor with known ground-truth tilt values and prints results to the terminal. If it runs without error, the installation is working.

### Step 4 (live camera only): Verify the RealSense camera

If you plan to use live camera mode, connect the RealSense D415 via USB 3.0 and run:

```
python realsense_bench_test.py
```

This prints camera intrinsics and frame rate. If it does not detect a camera, check the USB 3.0 connection and confirm pyrealsense2 installed correctly.

---

## 5. Running the Tool

### Video file mode (M1, M2, M3, M4, M5)

Processes a video file and displays the overlay in an OpenCV window.

```
python m3m4.py path/to/your/video.mp4
```

To use the included sample file:

```
python m3m4.py cat.mov
```

**Controls during playback:**
- `Q` -- quit
- `R` -- toggle recording (saves output frames to a file)
- `P` -- pause / resume

---

### S2 True-Azimuth Validation with Excel export (S2)

Runs a single video clip through validation using a known true azimuth angle. Produces a PNG accuracy chart, a CSV of per-frame results, and an Excel workbook.

```
python s2summary.py path/to/your/clip.mp4 --true-azimuth 45
```

Replace `45` with the actual azimuth angle in degrees for your clip. The angle is the horizontal surface tilt measured from the camera's forward axis.

Output files are saved in the same folder as the input clip:
- `accuracy_<clipname>.png` -- per-frame error chart
- `results_<clipname>.csv` -- per-frame error data
- `summary_<clipname>.xlsx` -- Excel workbook with Frame Data and Summary sheets

If you do not need the Excel file, you can run s2new.py instead, which produces only the PNG and CSV:

```
python s2new.py path/to/your/clip.mp4 --true-azimuth 45
```

---

### S3 Batch Processing (S3)

Processes a folder of video clips automatically using a manifest CSV file that lists the true azimuth for each clip. Designed for running a large clip set overnight without manual intervention.

**Step 1: Create a manifest CSV file.**

The manifest must have two columns with these exact headers:

```
filename,true_azimuth
clip_001.mp4,30
clip_002.mp4,45
clip_003.mp4,0
```

Save the manifest CSV in the same folder as your video clips, or provide a path to it.

**Step 2: Run the batch:**

```
python s3batchmode.py path/to/clips/folder --manifest path/to/manifest.csv --save-plots --true-azimuth
```

The `--save-plots` flag saves a PNG accuracy chart for each clip. Remove it if you only want the summary CSV.

Output is a summary CSV saved to the clips folder with one row per clip showing mean error, standard deviation, and number of valid frames.

---

### Live camera mode (C1)

Connects to an Intel RealSense D415 depth camera and displays the overlay in real time. Falls back to a standard webcam if no RealSense is detected.

```
python newc1.py
```

Before running, confirm your camera is working with:

```
python realsense_bench_test.py
```

**Note:** The `--live` RealSense mode has not been tested with a physical D415 connected, as the laboratory rig was not available during development. The code path reaches the correct point and raises a clear error message if no device is present. The System Installation Manual troubleshooting section covers what to check if the camera is not detected.

---

## 6. Output Files

| File | Produced by | Contents |
|------|-------------|----------|
| `accuracy_<clipname>.png` | s2new.py, s2summary.py | Per-frame angular error chart |
| `results_<clipname>.csv` | s2new.py, s2summary.py | Per-frame error data (frame number, estimated azimuth, true azimuth, error in degrees) |
| `summary_<clipname>.xlsx` | s2summary.py | Excel workbook: Frame Data sheet (colour-coded per-frame errors) and Summary sheet (clip-level statistics) |
| `batch_summary.csv` | s3batchmode.py | One row per clip: filename, mean error, standard deviation, valid frame count |
| `accuracy_<clipname>.png` (per clip) | s3batchmode.py with --save-plots | Per-clip accuracy chart for each clip in the batch |

---

## 7. Known Limitations

**Near-frontal is a genuinely hard case, not a bug.**
When the camera faces a surface square-on, the divergence field is flat and the shear magnitude shrinks to near zero. The max-div point wanders and the direction estimate becomes meaningless. The tool detects and flags this condition as NEAR FRONTAL (Milestone M5). A clip that reads "low confidence" near frontal is the tool working correctly, not a failure.

**The shear direction has known limits on the real rig.**
There are cases where the recovered direction does not fully resolve which way the surface is tilted. This is a known property of the perception algorithm. Documenting where the estimate fails is a valid and useful result.

**Live mode has not been tested with physical hardware.**
The `--live` RealSense path was implemented and verified to compile and reach the correct entry point, but was not tested with a physical D415 connected. Test on Lewis's rig and refer to the System Installation Manual troubleshooting section if the camera is not detected.

**Windows only (installation documentation).**
The System Installation Manual covers Windows. The tool should run on macOS and Linux but no installation guide exists for those platforms.

---

*Max-Div Alignment Validator -- COS40006 Computing Technology Project B*
*Student: Sara Tarannum | Client: Chris McCarthy | Supervisor: Naveed Ali*
