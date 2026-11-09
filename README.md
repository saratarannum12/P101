# Max-Div Alignment Validator - Code Handover

The code you build on for the **Max-Div Alignment Validator** project. See the
`maxdiv_alignment_validator_spec` document for what to build; this file explains
the code you're given and how to use it.

Your tool's job is to **display and validate** the existing max-div / shear cue
on video or a live camera - not to re-implement or improve it. Everything here
is camera-only; **no robot is involved at any stage.**

> **Read `read_this_first_paper_vs_code` before the paper draft.** The paper and
> this code describe overlapping but *different* systems. For your work,
> `maxdiv_perception.py` is ground truth; the paper is background.

---

## 1. What each file is

| File | Role | You will… |
|---|---|---|
| `maxdiv_perception.py` | The perception engine. `MaxDivProcessor` takes consecutive frames + a measured time-step and returns a `Result` (proximity, divergence, max-div point, shear cue). **The single source of truth.** | Call it; read its `Result`. Don't modify it. |
| `realsense_cue.py` | Camera → perception bridge. Wraps a RealSense camera and gives you `get_tau()` / `get_last()`. | Use it for the live-camera tier (C1), or as a reference for pulling frames. |
| `maxdiv_viz.py` | The existing overlay renderer (divergence heatmap, max-div marker, shear arc). | Reuse it for the overlay (requirement M2). |
| `validate_shear_synthetic.py` | Synthesises video of a camera approaching a plane of **known tilt**, and feeds it through the real `MaxDivProcessor`. | Your exact ground-truth source (S1) and the basis of the built-in generator (C2). |
| `measure_approach_angle.py` | Independent ground-truth surface angle from the depth camera (a plane fit). | Ground truth for validating **real** recorded clips (S1). |
| `realsense_bench_test.py` | Proves a camera streams; prints its intrinsics + frame rate. | Run first if you use a live camera. |

---

## 2. The `Result` object — what your tool reads

Every overlay and readout comes from `MaxDivProcessor`'s `Result`. The fields
you'll actually use:

| Field | Meaning | Used for |
|---|---|---|
| `tau_s` | Time-to-contact in **seconds**, or `None` if invalid this frame. | Proximity readout (M3). |
| `divmax_raw` | Raw divergence (proximity). `nan` until valid. | Proximity readout (M3). |
| `max_loc` | `(x, y)` of the max-div point, or `None`. | Overlay (M2). |
| `arc_vec` | Shear-axis direction (the recovered arc). | Direction-to-frontal arrow (M4). |
| `axis_strength` | Shear magnitude; **low ⇒ near-frontal / ill-conditioned**. | Confidence flag (M5). |
| `n_vectors` | Number of good flow vectors this frame. | Confidence flag (M5). |
| `valid` | Whether `tau_s` is trustworthy this frame. | Gating readouts. |
| `render` | Arrays for the overlay, when you ask for them (see below). | Overlay (M2). |

> **The direction-to-frontal arrow (M4)** comes from the shear cue: `arc_vec`
> gives the axis, and `axis_strength` tells you whether to trust it. When
> `axis_strength` is low (near frontal), show "low confidence" instead of a
> confident arrow — that's requirement M5, and it's the honest behaviour the
> project is really about.

---

## 3. Minimal usage

### Run perception on frames you already have (video file)

```python
import cv2
from maxdiv_perception import MaxDivProcessor

proc = MaxDivProcessor()
proc.reset()

cap = cv2.VideoCapture("clip.mp4")
prev_t = None
ok, frame = cap.read()
proc.process(frame, dt=0.0)          # first frame primes state — result not valid yet

while True:
    ok, frame = cap.read()
    if not ok:
        break
    dt = 1.0 / cap.get(cv2.CAP_PROP_FPS)   # time between frames
    res = proc.process(frame, dt, want_render=True)
    if res.valid:
        print(res.tau_s, res.divmax_raw, res.arc_vec, res.axis_strength)
    # res.render carries what maxdiv_viz.py draws
```

> **Two things that will bite you if you skip them:**
> - The **first frame primes the pipeline** and returns an invalid result. Always
>   feed at least two frames before expecting anything valid.
> - **Time-step (`dt`) matters.** Time-to-contact scales with the time between
>   frames, so pass a real `dt` (from the video's frame rate, or measured), not a
>   guess.

### Draw the overlay

```python
from maxdiv_viz import MaxDivVisualizer
viz = MaxDivVisualizer()
viz.show(res, res.tau_s, depth=None, event="")   # needs res.render (want_render=True)
```

### Live camera (Could-tier C1)

```python
from realsense_cue import RealSenseCue
cue = RealSenseCue()
cue.start()
try:
    while True:
        tau = cue.get_tau()      # seconds, or None
        res = cue.get_last()     # full Result
finally:
    cue.stop()
```

### Make a known-ground-truth test clip (for validation, S1 / C2)

`validate_shear_synthetic.py` already synthesises approach flow for a plane of
**known tilt** and runs it through the real processor. Read it to see how the
geometry is built — you can reuse that to generate your own validation clips
where the true "direction to frontal" is known exactly.

```bash
python validate_shear_synthetic.py                 # runs a tilt sweep
python validate_shear_synthetic.py --azimuth 90    # tilt in a different direction
```

### Depth ground truth for real clips (S1)

```bash
python measure_approach_angle.py --watch     # prints the true surface angle live
```

Use this as the independent "truth" to compare your estimated direction against
on **real** footage. It uses depth only — never feed depth back into the cue.

---

## 4. Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install numpy opencv-python matplotlib
pip install pyrealsense2                             # only if you use a live camera / depth
```

- A recorded video needs only `numpy` + `opencv-python`. `pyrealsense2` is needed
  only for live camera input and for `measure_approach_angle.py`.
- Once it runs, freeze your versions: `pip freeze > requirements.txt` and commit it.
- If you use a live camera, run `python realsense_bench_test.py` first to confirm
  it streams and to read its intrinsics. Don't hard-code intrinsics.

---

## 5. Things worth knowing (so you don't chase phantoms)

- **Frontal is a genuinely hard case, not a bug.** When the camera faces a
  surface square-on, the divergence field is flat and the shear shrinks to near
  zero  so the max-div point wanders and the direction estimate becomes
  meaningless. Your tool's job is to **detect and flag** this (M5), not to fix it.
  A clip that reads "low confidence" near frontal is the tool working correctly.
- **The shear direction has known limits on the real rig.** The supervisor has
  measured cases where the recovered direction doesn't fully resolve which way the
  surface is tilted. Don't be alarmed if validation shows the estimate failing in
  some conditions  **documenting where it fails is a valid, valuable result.**
- **Depth is validation only.** Use it as independent truth; never feed it into
  the cue. Keeping the cue purely monocular is the whole point.
- **Use `arc_vec` + `axis_strength` for the direction cue**, not the raw max-div
  point position (which wanders at frontal).

---

## 6. First-week checklist

1. `pip install numpy opencv-python matplotlib`.
2. Run `python validate_shear_synthetic.py` — it exercises the real perception
   engine with no camera needed, and shows the cue responding to known tilt. This
   proves the code works on your machine before you build anything.
3. Open `maxdiv_perception.py` and find the `Result` class — the fields in §2 are
   what your tool consumes.
4. Get a single video file playing in an OpenCV window (Milestone 1). That's your
   first demonstrable step — don't add perception until it plays cleanly.
5. Then wire `MaxDivProcessor` into the loop (Milestone 2) using the snippet in §3.

---

