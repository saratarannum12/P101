#!/usr/bin/env python3

import csv
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import cv2

# make sure the project files are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maxdiv_perception import MaxDivProcessor


# ================================
# UNIT TESTS
# ================================

class TestAngularDiff(unittest.TestCase):
    """[UNIT] _angular_diff — circular wraparound subtraction."""

    def _diff(self, est, true):
        """Replicate the formula from s3_batch.py and m3m4.py."""
        return (est - true + 180.0) % 360.0 - 180.0

    def test_zero_error(self):
        """Same angle should give zero error."""
        self.assertAlmostEqual(self._diff(45, 45), 0.0)

    def test_positive_offset(self):
        """30 deg ahead of truth should give +30."""
        self.assertAlmostEqual(self._diff(60, 30), 30.0)

    def test_negative_offset(self):
        """30 deg behind truth should give -30."""
        self.assertAlmostEqual(self._diff(0, 30), -30.0)

    def test_wraparound_across_180(self):
        """170 vs -170 should be a 20 deg difference, not 340."""
        diff = abs(self._diff(170, -170))
        self.assertAlmostEqual(diff, 20.0, places=5)

    def test_wraparound_across_zero(self):
        """350 vs 10 should be 20 deg apart going the short way."""
        diff = abs(self._diff(350, 10))
        self.assertAlmostEqual(diff, 20.0, places=5)

    def test_exactly_180(self):
        """180 deg error is the maximum possible."""
        diff = abs(self._diff(180, 0))
        self.assertLessEqual(diff, 180.0)

    def test_negative_angles(self):
        """Negative angles should work correctly."""
        self.assertAlmostEqual(self._diff(-45, -45), 0.0)


class TestAzimuthFromArcVec(unittest.TestCase):
    """[UNIT] Azimuth angle extraction from arc_vec using atan2."""

    def _angle(self, avx, avy):
        return math.degrees(math.atan2(avy, avx))

    def test_right(self):
        """(1, 0) points right → 0 deg."""
        self.assertAlmostEqual(self._angle(1, 0), 0.0)

    def test_up(self):
        """(0, -1) points up in image coords → -90 deg."""
        self.assertAlmostEqual(self._angle(0, -1), -90.0)

    def test_down(self):
        """(0, 1) points down → 90 deg."""
        self.assertAlmostEqual(self._angle(0, 1), 90.0)

    def test_left(self):
        """(-1, 0) points left → 180 deg."""
        self.assertAlmostEqual(abs(self._angle(-1, 0)), 180.0)

    def test_diagonal(self):
        """(1, 1) normalised → 45 deg."""
        v = 1 / math.sqrt(2)
        self.assertAlmostEqual(self._angle(v, v), 45.0, places=4)

    def test_rotate180_negates_vector(self):
        """rotate_180=True should negate arc_vec before computing angle."""
        avx, avy = 1.0, 0.0          # points right
        avx, avy = -avx, -avy        # after rotation: points left
        self.assertAlmostEqual(abs(self._angle(avx, avy)), 180.0)


class TestS2RecordLogic(unittest.TestCase):
    """[UNIT] S2 per-frame recording guard conditions."""

    MIN_VECTORS       = 20
    MIN_AXIS_STRENGTH = 0.005

    def _should_record(self, valid, n_vectors, axis_strength):
        """Replicate the guard from s2_record_frame()."""
        if not valid:
            return False
        if n_vectors < self.MIN_VECTORS:
            return False
        if axis_strength < self.MIN_AXIS_STRENGTH:
            return False
        return True

    def test_records_good_frame(self):
        self.assertTrue(self._should_record(True, 50, 0.02))

    def test_skips_invalid_result(self):
        self.assertFalse(self._should_record(False, 50, 0.02))

    def test_skips_low_vectors(self):
        self.assertFalse(self._should_record(True, 10, 0.02))

    def test_skips_weak_axis(self):
        self.assertFalse(self._should_record(True, 50, 0.001))

    def test_boundary_vectors_exact(self):
        """Exactly MIN_VECTORS should pass."""
        self.assertTrue(self._should_record(True, self.MIN_VECTORS, 0.02))

    def test_boundary_axis_exact(self):
        """Exactly MIN_AXIS_STRENGTH should pass."""
        self.assertTrue(self._should_record(True, 50, self.MIN_AXIS_STRENGTH))

    def test_boundary_vectors_one_below(self):
        """One below MIN_VECTORS should fail."""
        self.assertFalse(self._should_record(True, self.MIN_VECTORS - 1, 0.02))


class TestMeanErrorCalculation(unittest.TestCase):
    """[UNIT] Mean absolute angular error calculation."""

    def _mean_abs_error(self, est_angles, true_az):
        errors = [abs((e - true_az + 180) % 360 - 180) for e in est_angles]
        return sum(errors) / len(errors)

    def test_perfect_estimate(self):
        """All estimates equal to truth → 0 error."""
        self.assertAlmostEqual(self._mean_abs_error([45, 45, 45], 45), 0.0)

    def test_constant_offset(self):
        """All estimates 10 deg off → mean error = 10."""
        self.assertAlmostEqual(self._mean_abs_error([55, 55, 55], 45), 10.0)

    def test_symmetric_errors_cancel_in_mean(self):
        """Errors of +10 and -10 → mean abs error = 10 (not 0)."""
        self.assertAlmostEqual(self._mean_abs_error([55, 35], 45), 10.0)

    def test_wraparound_error(self):
        """350 vs truth 10 → error should be 20, not 340."""
        err = self._mean_abs_error([350], 10)
        self.assertAlmostEqual(err, 20.0, places=4)

    def test_single_frame(self):
        """Single frame, 5 deg off."""
        self.assertAlmostEqual(self._mean_abs_error([5], 0), 5.0)


class TestVideoExtensions(unittest.TestCase):
    """[UNIT] Video file extension filtering used in s3_batch.py."""

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MOV", ".MP4"}

    def test_mp4_accepted(self):
        self.assertIn(".mp4", self.VIDEO_EXTENSIONS)

    def test_mov_accepted(self):
        self.assertIn(".mov", self.VIDEO_EXTENSIONS)

    def test_uppercase_MOV_accepted(self):
        self.assertIn(".MOV", self.VIDEO_EXTENSIONS)

    def test_txt_rejected(self):
        self.assertNotIn(".txt", self.VIDEO_EXTENSIONS)

    def test_py_rejected(self):
        self.assertNotIn(".py", self.VIDEO_EXTENSIONS)

    def test_csv_rejected(self):
        self.assertNotIn(".csv", self.VIDEO_EXTENSIONS)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def _make_synthetic_frame(w=320, h=240, seed=0):
    """Generate a repeatable textured BGR frame for testing."""
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 220, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestProcessorIntegration(unittest.TestCase):
    """[INTEGRATION] MaxDivProcessor + result fields."""

    def setUp(self):
        self.proc = MaxDivProcessor()
        self.proc.reset()

    def test_processor_returns_result(self):
        """Processor must return a result object, not None."""
        frame = _make_synthetic_frame()
        result = self.proc.process(frame, dt=0.05)
        self.assertIsNotNone(result)

    def test_result_has_required_fields(self):
        """Result object must have the fields the validator reads."""
        frame = _make_synthetic_frame()
        result = self.proc.process(frame, dt=0.05)
        self.assertTrue(hasattr(result, "valid"))
        self.assertTrue(hasattr(result, "tau_s"))
        self.assertTrue(hasattr(result, "divmax_raw"))
        self.assertTrue(hasattr(result, "arc_vec"))
        self.assertTrue(hasattr(result, "axis_strength"))
        self.assertTrue(hasattr(result, "n_vectors"))
        self.assertTrue(hasattr(result, "max_loc"))

    def test_prime_then_process(self):
        """First frame primes (dt=0.0), second frame should produce a result."""
        f1 = _make_synthetic_frame(seed=0)
        f2 = _make_synthetic_frame(seed=1)
        self.proc.process(f1, dt=0.0)
        result = self.proc.process(f2, dt=0.05)
        self.assertIsNotNone(result)

    def test_arc_vec_is_unit_or_zero(self):
        """arc_vec must be a 2-element sequence with magnitude ≤ 1 + epsilon."""
        f1 = _make_synthetic_frame(seed=0)
        f2 = _make_synthetic_frame(seed=1)
        self.proc.process(f1, dt=0.0)
        result = self.proc.process(f2, dt=0.05)
        avx, avy = result.arc_vec
        magnitude = math.sqrt(avx**2 + avy**2)
        self.assertLessEqual(magnitude, 1.0 + 1e-5)

    def test_reset_clears_state(self):
        """After reset(), the processor should accept a fresh prime."""
        frame = _make_synthetic_frame()
        self.proc.process(frame, dt=0.05)
        self.proc.reset()
        result = self.proc.process(frame, dt=0.0)
        self.assertIsNotNone(result)

    def test_n_vectors_non_negative(self):
        """n_vectors should never be negative."""
        frame = _make_synthetic_frame()
        result = self.proc.process(frame, dt=0.05)
        self.assertGreaterEqual(result.n_vectors, 0)

    def test_axis_strength_non_negative(self):
        """axis_strength should never be negative."""
        frame = _make_synthetic_frame()
        result = self.proc.process(frame, dt=0.05)
        self.assertGreaterEqual(result.axis_strength, 0.0)


class TestS2RecordingIntegration(unittest.TestCase):
    """[INTEGRATION] Per-frame recording + summary calculation working together."""

    def setUp(self):
        self.frame_nums = []
        self.est_angles = []

    def _record(self, avx, avy, frame_num,
                valid=True, n_vectors=50, axis_strength=0.02):
        MIN_VECTORS = 20
        MIN_AXIS_STRENGTH = 0.005
        if not valid or n_vectors < MIN_VECTORS or axis_strength < MIN_AXIS_STRENGTH:
            return
        angle = math.degrees(math.atan2(avy, avx))
        self.frame_nums.append(frame_num)
        self.est_angles.append(angle)

    def _mean_abs_error(self, true_az):
        errors = [abs((e - true_az + 180) % 360 - 180) for e in self.est_angles]
        return sum(errors) / len(errors)

    def test_records_accumulate(self):
        """Multiple valid frames should accumulate in the lists."""
        for i in range(5):
            self._record(1.0, 0.0, i)
        self.assertEqual(len(self.est_angles), 5)

    def test_invalid_frames_not_recorded(self):
        """Invalid frames must not appear in the accumulator."""
        self._record(1.0, 0.0, 0, valid=False)
        self._record(1.0, 0.0, 1, n_vectors=5)
        self._record(1.0, 0.0, 2, axis_strength=0.001)
        self.assertEqual(len(self.est_angles), 0)

    def test_mean_error_correct_after_recording(self):
        """After recording frames pointing at 0 deg, error vs 0 should be ~0."""
        for i in range(10):
            self._record(1.0, 0.0, i)   # all pointing at 0 deg
        err = self._mean_abs_error(true_az=0.0)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_mixed_valid_invalid(self):
        """Only valid frames should count toward the mean error."""
        self._record(1.0, 0.0, 0)           # valid, 0 deg
        self._record(1.0, 0.0, 1, valid=False)  # skipped
        self._record(1.0, 0.0, 2)           # valid, 0 deg
        self.assertEqual(len(self.est_angles), 2)
        self.assertAlmostEqual(self._mean_abs_error(0.0), 0.0, places=4)


class TestCSVWriteIntegration(unittest.TestCase):
    """[INTEGRATION] CSV row generation + file write + read-back."""

    CSV_FIELDS = [
        "clip", "total_frames", "valid_frames", "coverage_pct",
        "mean_est_az_deg", "true_azimuth_deg",
        "mean_error_deg", "median_error_deg", "max_error_deg",
        "mean_tau_s", "status",
    ]

    def _write_csv(self, rows, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_csv_is_created(self):
        """CSV file should exist after writing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._write_csv([], path)
            self.assertTrue(os.path.exists(path))
        finally:
            os.unlink(path)

    def test_csv_has_correct_header(self):
        """Header row should match CSV_FIELDS exactly."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                         mode="w", newline="") as f:
            path = f.name
        try:
            self._write_csv([], path)
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, self.CSV_FIELDS)
        finally:
            os.unlink(path)

    def test_csv_row_roundtrip(self):
        """A written row should read back with the same values."""
        row = {
            "clip": "test.mp4", "total_frames": 100, "valid_frames": 80,
            "coverage_pct": 80.0, "mean_est_az_deg": 12.3,
            "true_azimuth_deg": 0.0, "mean_error_deg": 12.3,
            "median_error_deg": 11.0, "max_error_deg": 30.0,
            "mean_tau_s": 1.234, "status": "OK",
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                         mode="w", newline="") as f:
            path = f.name
        try:
            self._write_csv([row], path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["clip"], "test.mp4")
            self.assertEqual(rows[0]["status"], "OK")
            self.assertAlmostEqual(float(rows[0]["mean_error_deg"]), 12.3)
        finally:
            os.unlink(path)

    def test_csv_multiple_rows(self):
        """Multiple clips should each get their own row."""
        rows = [
            {"clip": "a.mp4", "total_frames": 100, "valid_frames": 90,
             "coverage_pct": 90.0, "mean_est_az_deg": 5.0,
             "true_azimuth_deg": 0.0, "mean_error_deg": 5.0,
             "median_error_deg": 4.5, "max_error_deg": 10.0,
             "mean_tau_s": 2.0, "status": "OK"},
            {"clip": "b.mov", "total_frames": 200, "valid_frames": 0,
             "coverage_pct": 0.0, "mean_est_az_deg": "",
             "true_azimuth_deg": 0.0, "mean_error_deg": "",
             "median_error_deg": "", "max_error_deg": "",
             "mean_tau_s": "", "status": "NO_VALID_FRAMES"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                         mode="w", newline="") as f:
            path = f.name
        try:
            self._write_csv(rows, path)
            with open(path, newline="") as f:
                read_rows = list(csv.DictReader(f))
            self.assertEqual(len(read_rows), 2)
            self.assertEqual(read_rows[1]["status"], "NO_VALID_FRAMES")
        finally:
            os.unlink(path)


# =============================================================================
# SYSTEM TESTS
# =============================================================================

def _make_synthetic_video(path, n_frames=30, w=320, h=240, fps=30):
    """Write a short synthetic .mp4 for system-level testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        img = rng.integers(40, 200, (h, w), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()


class TestSystemPipelineEndToEnd(unittest.TestCase):
    """[SYSTEM] Full pipeline: video file → processor → results → CSV."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "test_clip.mp4")
        _make_synthetic_video(self.video_path, n_frames=40)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_video_file_created(self):
        """Synthetic video file should exist and be non-empty."""
        self.assertTrue(os.path.exists(self.video_path))
        self.assertGreater(os.path.getsize(self.video_path), 0)

    def test_full_pipeline_runs_without_error(self):
        """Running the whole pipeline on a real video file should not raise."""
        cap = cv2.VideoCapture(self.video_path)
        self.assertTrue(cap.isOpened())

        proc = MaxDivProcessor()
        proc.reset()
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / video_fps

        ok, frame = cap.read()
        self.assertTrue(ok)
        proc.process(frame, dt=0.0, want_render=False)

        results = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = proc.process(frame, dt=dt, want_render=False)
            results.append(res)

        cap.release()
        self.assertGreater(len(results), 0)

    def test_all_results_have_valid_field(self):
        """Every result from the processor must have a boolean valid field."""
        cap = cv2.VideoCapture(self.video_path)
        proc = MaxDivProcessor()
        proc.reset()
        dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)

        ok, frame = cap.read()
        proc.process(frame, dt=0.0, want_render=False)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = proc.process(frame, dt=dt, want_render=False)
            self.assertIsInstance(res.valid, (bool, np.bool_))

        cap.release()

    def test_csv_produced_from_batch(self):
        """Processing a folder with one clip should produce a CSV file."""
        # replicate the core of process_clip() + write_csv()
        cap = cv2.VideoCapture(self.video_path)
        proc = MaxDivProcessor()
        proc.reset()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)

        ok, frame = cap.read()
        proc.process(frame, dt=0.0, want_render=False)

        valid_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = proc.process(frame, dt=dt, want_render=False)
            if res.valid and res.n_vectors >= 20 and res.axis_strength >= 0.005:
                valid_frames += 1
        cap.release()

        csv_path = os.path.join(self.tmpdir, "batch_results.csv")
        row = {
            "clip": "test_clip.mp4",
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "coverage_pct": round(valid_frames / total_frames * 100, 1),
            "mean_est_az_deg": "",
            "true_azimuth_deg": "",
            "mean_error_deg": "",
            "median_error_deg": "",
            "max_error_deg": "",
            "mean_tau_s": "",
            "status": "OK" if valid_frames > 0 else "NO_VALID_FRAMES",
        }
        fields = list(row.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path) as f:
            content = f.read()
        self.assertIn("test_clip.mp4", content)
        self.assertIn("total_frames", content)

    def test_s2_summary_numbers_are_finite(self):
        """Any recorded azimuth estimates should be finite numbers."""
        cap = cv2.VideoCapture(self.video_path)
        proc = MaxDivProcessor()
        proc.reset()
        dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)

        ok, frame = cap.read()
        proc.process(frame, dt=0.0, want_render=False)

        est_angles = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = proc.process(frame, dt=dt, want_render=False)
            if res.valid and res.n_vectors >= 20 and res.axis_strength >= 0.005:
                avx, avy = res.arc_vec
                angle = math.degrees(math.atan2(avy, avx))
                self.assertTrue(math.isfinite(angle),
                                msg=f"Estimated azimuth is not finite: {angle}")
                est_angles.append(angle)

        cap.release()
        # if we got any valid frames, mean error calculation should work
        if est_angles:
            true_az = 0.0
            errors = [abs((e - true_az + 180) % 360 - 180) for e in est_angles]
            mean_err = sum(errors) / len(errors)
            self.assertTrue(math.isfinite(mean_err))
            self.assertGreaterEqual(mean_err, 0.0)
            self.assertLessEqual(mean_err, 180.0)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Max-Div Alignment Validator — Test Suite")
    print("Unit  |  Integration  |  System")
    print("=" * 65)
    print()
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    # Add in order: unit → integration → system
    unit_classes = [
        TestAngularDiff,
        TestAzimuthFromArcVec,
        TestS2RecordLogic,
        TestMeanErrorCalculation,
        TestVideoExtensions,
    ]
    integration_classes = [
        TestProcessorIntegration,
        TestS2RecordingIntegration,
        TestCSVWriteIntegration,
    ]
    system_classes = [
        TestSystemPipelineEndToEnd,
    ]

    for cls in unit_classes + integration_classes + system_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 65)
    total   = result.testsRun
    passed  = total - len(result.failures) - len(result.errors)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {len(result.failures)}")
    print(f"  Errors      : {len(result.errors)}")
    print("=" * 65)

    sys.exit(0 if result.wasSuccessful() else 1)