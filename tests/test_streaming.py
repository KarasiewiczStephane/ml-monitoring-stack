"""Tests for sliding window drift detection on streaming data."""

import numpy as np
import pandas as pd
import pytest

from src.drift.streaming import SlidingWindowDriftDetector


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create a small reference dataset."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "f0": rng.standard_normal(200),
            "f1": rng.standard_normal(200),
        }
    )


@pytest.fixture
def small_detector(reference_data) -> SlidingWindowDriftDetector:
    """Create a detector with a small window for testing."""
    return SlidingWindowDriftDetector(
        reference_data=reference_data,
        window_size=50,
        step_size=10,
        threshold=0.05,
    )


class TestWindowFilling:
    """Tests for buffer behaviour before the window is full."""

    def test_no_check_before_window_full(self, small_detector):
        """No drift check should occur until the window is full."""
        rng = np.random.default_rng(99)
        for _ in range(49):
            result = small_detector.add_sample(
                {"f0": rng.standard_normal(), "f1": rng.standard_normal()}
            )
            assert result is None

    def test_check_triggers_when_window_full(self, small_detector):
        """A drift check should trigger once window_size samples are added."""
        rng = np.random.default_rng(99)
        results = []
        for _ in range(50):
            result = small_detector.add_sample(
                {"f0": rng.standard_normal(), "f1": rng.standard_normal()}
            )
            if result is not None:
                results.append(result)

        assert len(results) == 1
        assert "drift_detected" in results[0]
        assert "feature_scores" in results[0]


class TestStepSize:
    """Tests for step-size triggered checks."""

    def test_respects_step_size(self, small_detector):
        """After initial check, next check should wait for step_size samples."""
        rng = np.random.default_rng(99)
        # Fill window and trigger first check
        for _ in range(50):
            small_detector.add_sample({"f0": rng.standard_normal(), "f1": rng.standard_normal()})

        # Add 9 more (step_size=10, should NOT trigger)
        for _ in range(9):
            result = small_detector.add_sample(
                {"f0": rng.standard_normal(), "f1": rng.standard_normal()}
            )
            assert result is None

        # 10th sample should trigger
        result = small_detector.add_sample(
            {"f0": rng.standard_normal(), "f1": rng.standard_normal()}
        )
        assert result is not None


class TestDriftDetection:
    """Tests for actual drift detection in streaming mode."""

    def test_detects_drift_with_shifted_data(self, reference_data):
        """Drift should be detected when streaming shifted data."""
        detector = SlidingWindowDriftDetector(
            reference_data=reference_data,
            window_size=100,
            step_size=100,
            threshold=0.05,
        )
        rng = np.random.default_rng(99)
        # Stream shifted data
        for _ in range(100):
            detector.add_sample(
                {
                    "f0": rng.standard_normal() + 5.0,  # Large shift
                    "f1": rng.standard_normal(),
                }
            )

        result = detector._last_drift_result
        assert result is not None
        assert "f0" in result["drifted_features"]

    def test_no_drift_with_same_distribution(self, reference_data):
        """No drift should be detected with matching distributions."""
        detector = SlidingWindowDriftDetector(
            reference_data=reference_data,
            window_size=100,
            step_size=100,
            threshold=0.05,
        )
        rng = np.random.default_rng(99)
        for _ in range(100):
            detector.add_sample(
                {
                    "f0": rng.standard_normal(),
                    "f1": rng.standard_normal(),
                }
            )

        result = detector._last_drift_result
        assert result is not None
        assert len(result["drifted_features"]) == 0


class TestBatchProcessing:
    """Tests for batch sample addition."""

    def test_add_batch(self, small_detector):
        """add_batch should process all samples and return results."""
        rng = np.random.default_rng(99)
        samples = [{"f0": rng.standard_normal(), "f1": rng.standard_normal()} for _ in range(60)]

        results = small_detector.add_batch(samples)
        assert len(results) >= 1
        assert all("drift_detected" in r for r in results)


class TestStatusAndReset:
    """Tests for status reporting and reset."""

    def test_initial_status(self, small_detector):
        """Initial status should show empty buffer."""
        status = small_detector.get_current_status()
        assert status["buffer_size"] == 0
        assert not status["ready"]
        assert status["last_result"] is None

    def test_status_after_samples(self, small_detector):
        """Status should reflect buffer filling."""
        rng = np.random.default_rng(99)
        for _ in range(25):
            small_detector.add_sample({"f0": rng.standard_normal(), "f1": rng.standard_normal()})

        status = small_detector.get_current_status()
        assert status["buffer_size"] == 25
        assert not status["ready"]

    def test_reset_clears_state(self, small_detector):
        """reset should clear the buffer and last result."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            small_detector.add_sample({"f0": rng.standard_normal(), "f1": rng.standard_normal()})

        small_detector.reset()
        status = small_detector.get_current_status()
        assert status["buffer_size"] == 0
        assert status["last_result"] is None

    def test_buffer_overflow(self, reference_data):
        """Buffer should not exceed window_size due to deque maxlen."""
        detector = SlidingWindowDriftDetector(
            reference_data=reference_data,
            window_size=20,
            step_size=5,
            threshold=0.05,
        )
        rng = np.random.default_rng(99)
        for _ in range(50):
            detector.add_sample({"f0": rng.standard_normal(), "f1": rng.standard_normal()})

        assert len(detector.buffer) == 20
