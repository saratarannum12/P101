import csv
import os
import glob
from collections import defaultdict

# ─────────────────────────────────────────────
# Load all CSV files from current folder
# ─────────────────────────────────────────────

csv_files = glob.glob("*.csv")

if not csv_files:
    print("[ERROR] No CSV files found in current folder.")
    print("Make sure your CSV files are in the same folder as this script.")
    exit()

print(f"[INFO] Found {len(csv_files)} CSV files: {csv_files}")
print()

# Store data grouped by action
action_data = defaultdict(lambda: defaultdict(list))

for csv_path in csv_files:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row['action']
            action_data[action]['mean_flow_x'].append(float(row['mean_flow_x']))
            action_data[action]['mean_flow_y'].append(float(row['mean_flow_y']))
            action_data[action]['mean_magnitude'].append(float(row['mean_magnitude']))
            action_data[action]['divergence'].append(float(row['divergence']))

# ─────────────────────────────────────────────
# Per-joint breakdown for key joints
# ─────────────────────────────────────────────

key_joints = ['right_wrist', 'left_wrist', 'right_knee', 'left_knee', 'right_hip', 'left_hip']

action_joint_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for csv_path in csv_files:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row['action']
            joint = row['joint_name']
            if joint in key_joints:
                action_joint_data[action][joint]['mean_magnitude'].append(float(row['mean_magnitude']))
                action_joint_data[action][joint]['divergence'].append(float(row['divergence']))

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0

def interpret_divergence(d):
    if d > 0.1:
        return "toward camera"
    elif d < -0.1:
        return "away from camera"
    else:
        return "neutral"

# ─────────────────────────────────────────────
# Overall Summary Table
# ─────────────────────────────────────────────

col_w = 16

print("=" * 95)
print(" OVERALL FEATURE SUMMARY — AVERAGED ACROSS ALL JOINTS AND FRAMES")
print("=" * 95)
print(f"{'Action':<14} {'Mean Flow X':>{col_w}} {'Mean Flow Y':>{col_w}} {'Mean Magnitude':>{col_w}} {'Divergence':>{col_w}}")
print("-" * 95)

for action in sorted(action_data.keys()):
    d = action_data[action]
    print(f"{action:<14} "
          f"{avg(d['mean_flow_x']):>{col_w}.4f} "
          f"{avg(d['mean_flow_y']):>{col_w}.4f} "
          f"{avg(d['mean_magnitude']):>{col_w}.4f} "
          f"{avg(d['divergence']):>{col_w}.4f}")

print("=" * 95)
print()

# ─────────────────────────────────────────────
# Divergence Interpretation
# ─────────────────────────────────────────────

print("=" * 60)
print(" DIVERGENCE INTERPRETATION (toward/away from camera)")
print("=" * 60)
for action in sorted(action_data.keys()):
    d = avg(action_data[action]['divergence'])
    interp = interpret_divergence(d)
    print(f"  {action:<20} divergence = {d:>8.4f}  ->  {interp}")
print("=" * 60)
print()

# ─────────────────────────────────────────────
# Per-Joint Magnitude Breakdown
# ─────────────────────────────────────────────

print("=" * 95)
print(" PER-JOINT MAGNITUDE BREAKDOWN — KEY JOINTS ONLY")
print("=" * 95)

header = f"{'Joint':<20}"
for action in sorted(action_data.keys()):
    header += f" {action:>{col_w}}"
print(header)
print("-" * 95)

for joint in key_joints:
    row_str = f"{joint:<20}"
    for action in sorted(action_data.keys()):
        mag_list = action_joint_data[action][joint]['mean_magnitude']
        val = avg(mag_list) if mag_list else 0.0
        row_str += f" {val:>{col_w}.4f}"
    print(row_str)

print("=" * 95)
print()

# ─────────────────────────────────────────────
# Key Observations
# ─────────────────────────────────────────────

print("=" * 70)
print(" KEY OBSERVATIONS")
print("=" * 70)

all_actions = sorted(action_data.keys())

wrist_mags = {}
for action in all_actions:
    mags = (action_joint_data[action]['right_wrist']['mean_magnitude'] +
            action_joint_data[action]['left_wrist']['mean_magnitude'])
    wrist_mags[action] = avg(mags) if mags else 0.0

highest_wrist = max(wrist_mags, key=wrist_mags.get)
print(f"  Highest wrist motion:     {highest_wrist} (magnitude = {wrist_mags[highest_wrist]:.4f})")

knee_mags = {}
for action in all_actions:
    mags = (action_joint_data[action]['right_knee']['mean_magnitude'] +
            action_joint_data[action]['left_knee']['mean_magnitude'])
    knee_mags[action] = avg(mags) if mags else 0.0

highest_knee = max(knee_mags, key=knee_mags.get)
print(f"  Highest knee motion:      {highest_knee} (magnitude = {knee_mags[highest_knee]:.4f})")

divs = {a: avg(action_data[a]['divergence']) for a in all_actions}
highest_div = max(divs, key=divs.get)
lowest_div  = min(divs, key=divs.get)
print(f"  Most positive divergence: {highest_div} ({divs[highest_div]:.4f}) -- moving toward camera")
print(f"  Most negative divergence: {lowest_div} ({divs[lowest_div]:.4f}) -- moving away from camera")

print("=" * 70)
print()
print("[INFO] Analysis complete.")
print(f"[INFO] CSV files analysed: {', '.join(csv_files)}")