"""Tests for performance degradation detection (CUSUM and Page-Hinkley)."""

import numpy as np

from src.performance.degradation_detector import (
    CUSUMDetector,
    DegradationDetector,
    PageHinkleyDetector,
)


class TestCUSUMDetector:
    """Tests for the CUSUM detector."""

    def test_no_alert_on_stable_values(self):
        """Stable values around the target should not trigger alerts."""
        detector = CUSUMDetector(target_mean=0.9, threshold=5.0, drift_allowance=0.5)
        rng = np.random.default_rng(42)

        for _ in range(100):
            value = 0.9 + rng.standard_normal() * 0.01
            alert = detector.update(value)
            assert alert is None

    def test_detects_positive_shift(self):
        """A large positive shift should trigger an alert."""
        detector = CUSUMDetector(target_mean=0.9, threshold=3.0, drift_allowance=0.1)
        alerts = []

        # Feed values above target
        for _ in range(50):
            alert = detector.update(1.5)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].method == "CUSUM"
        assert "Positive shift" in alerts[0].message

    def test_detects_negative_shift(self):
        """A large negative shift should trigger an alert."""
        detector = CUSUMDetector(target_mean=0.9, threshold=3.0, drift_allowance=0.1)
        alerts = []

        for _ in range(50):
            alert = detector.update(0.1)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert "Negative shift" in alerts[0].message

    def test_reset_clears_state(self):
        """Reset should zero accumulators."""
        detector = CUSUMDetector(target_mean=0.9)
        detector.update(5.0)
        detector.reset()

        assert detector.s_pos == 0.0
        assert detector.s_neg == 0.0
        assert len(detector.values) == 0

    def test_threshold_sensitivity(self):
        """Higher threshold should be harder to trigger."""
        low_threshold = CUSUMDetector(target_mean=0.9, threshold=2.0, drift_allowance=0.1)
        high_threshold = CUSUMDetector(target_mean=0.9, threshold=50.0, drift_allowance=0.1)

        low_alerts = 0
        high_alerts = 0
        for _ in range(100):
            if low_threshold.update(1.2):
                low_alerts += 1
            if high_threshold.update(1.2):
                high_alerts += 1

        assert low_alerts > high_alerts


class TestPageHinkleyDetector:
    """Tests for the Page-Hinkley detector."""

    def test_no_alert_on_stable_values(self):
        """Stable values should not trigger alerts."""
        detector = PageHinkleyDetector(threshold=50.0, delta=0.005)
        rng = np.random.default_rng(42)

        for _ in range(200):
            value = rng.standard_normal() * 0.01
            alert = detector.update(value)
            assert alert is None

    def test_detects_gradual_drift(self):
        """A gradual upward trend should eventually trigger."""
        detector = PageHinkleyDetector(threshold=10.0, delta=0.001)
        alerts = []

        for i in range(500):
            value = 0.5 + i * 0.01  # Linearly increasing
            alert = detector.update(value)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].method == "PageHinkley"

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        detector = PageHinkleyDetector()
        for i in range(10):
            detector.update(float(i))
        detector.reset()

        assert detector.cumulative_sum == 0.0
        assert detector.count == 0
        assert detector.mean == 0.0


class TestDegradationDetector:
    """Tests for the combined degradation detector."""

    def test_no_alerts_on_stable(self):
        """Stable accuracy should not generate alerts."""
        detector = DegradationDetector(baseline_accuracy=0.9)
        rng = np.random.default_rng(42)

        for _ in range(100):
            alerts = detector.update(0.9 + rng.standard_normal() * 0.001)
            assert len(alerts) == 0

    def test_collects_alerts(self):
        """Alerts from both detectors should be collected."""
        detector = DegradationDetector(
            baseline_accuracy=0.9,
            config={"cusum_threshold": 2.0, "cusum_drift": 0.1},
        )

        all_alerts = []
        for _ in range(100):
            alerts = detector.update(1.5)  # Large positive shift
            all_alerts.extend(alerts)

        assert len(all_alerts) > 0
        assert len(detector.alerts) > 0

    def test_reset(self):
        """Reset should clear all detectors and alerts."""
        detector = DegradationDetector(baseline_accuracy=0.9)
        detector.update(5.0)
        detector.reset()

        assert len(detector.alerts) == 0
        assert detector.cusum.s_pos == 0.0

    def test_custom_config(self):
        """Custom config values should be applied."""
        config = {
            "cusum_threshold": 10.0,
            "cusum_drift": 1.0,
            "ph_threshold": 100.0,
            "ph_delta": 0.01,
        }
        detector = DegradationDetector(baseline_accuracy=0.5, config=config)

        assert detector.cusum.threshold == 10.0
        assert detector.cusum.drift_allowance == 1.0
        assert detector.page_hinkley.threshold == 100.0
        assert detector.page_hinkley.delta == 0.01
