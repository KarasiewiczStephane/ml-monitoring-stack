"""Tests for prediction distribution monitoring without ground truth."""

import numpy as np
import pytest

from src.performance.prediction_monitor import PredictionMonitor


@pytest.fixture
def monitor() -> PredictionMonitor:
    """Create a PredictionMonitor with baseline set."""
    mon = PredictionMonitor(window_size=500)
    rng = np.random.default_rng(42)
    confidences = rng.uniform(0.7, 0.95, 200).tolist()
    predictions = rng.choice([0, 1], size=200, p=[0.5, 0.5]).tolist()
    mon.set_baseline(confidences, predictions)
    return mon


class TestBaseline:
    """Tests for baseline configuration."""

    def test_set_baseline(self, monitor):
        """Baseline stats should be computed after setting."""
        assert monitor._baseline_confidence_stats is not None
        assert "mean" in monitor._baseline_confidence_stats
        assert monitor.reference_distribution is not None

    def test_baseline_distribution_sums_to_one(self, monitor):
        """Reference distribution should sum to approximately 1."""
        total = sum(monitor.reference_distribution.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestConfidenceShift:
    """Tests for confidence distribution shift detection."""

    def test_no_anomaly_with_similar_confidence(self, monitor):
        """Similar confidence distribution should not trigger anomaly."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            monitor.log_prediction(0, rng.uniform(0.7, 0.95))

        anomaly = monitor.check_confidence_distribution()
        assert anomaly is None

    def test_detects_confidence_shift(self, monitor):
        """Large confidence shift should be detected."""
        for _ in range(200):
            monitor.log_prediction(0, 0.1)  # Very low confidence

        anomaly = monitor.check_confidence_distribution()
        assert anomaly is not None
        assert anomaly.anomaly_type == "confidence_shift"
        assert anomaly.details["z_score"] > 3.0

    def test_insufficient_data_returns_none(self):
        """Not enough data should return None."""
        mon = PredictionMonitor(window_size=200)
        mon.set_baseline([0.9] * 100, [0] * 100)
        for _ in range(50):
            mon.log_prediction(0, 0.1)

        assert mon.check_confidence_distribution() is None


class TestClassBalance:
    """Tests for prediction class balance detection."""

    def test_no_anomaly_with_balanced_predictions(self, monitor):
        """Balanced predictions matching baseline should not trigger."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            monitor.log_prediction(rng.choice([0, 1]), 0.85)

        anomaly = monitor.check_class_balance()
        assert anomaly is None

    def test_detects_class_imbalance(self, monitor):
        """A heavily skewed distribution should be detected."""
        for _ in range(200):
            monitor.log_prediction(0, 0.85)  # All class 0

        anomaly = monitor.check_class_balance()
        assert anomaly is not None
        assert anomaly.anomaly_type == "class_imbalance"

    def test_severity_levels(self, monitor):
        """Very significant shift should be high severity."""
        for _ in range(300):
            monitor.log_prediction(0, 0.85)

        anomaly = monitor.check_class_balance()
        assert anomaly is not None
        assert anomaly.severity in ("medium", "high")


class TestDetectAnomalies:
    """Tests for the combined anomaly detection."""

    def test_detects_multiple_anomalies(self, monitor):
        """Both confidence and balance anomalies should be detected."""
        for _ in range(200):
            monitor.log_prediction(0, 0.1)  # Low confidence + all class 0

        anomalies = monitor.detect_anomalies()
        types = {a.anomaly_type for a in anomalies}
        assert "confidence_shift" in types
        assert "class_imbalance" in types

    def test_no_anomalies_on_normal_data(self, monitor):
        """Normal data should produce no anomalies."""
        rng = np.random.default_rng(42)
        for _ in range(200):
            monitor.log_prediction(rng.choice([0, 1]), rng.uniform(0.7, 0.95))

        anomalies = monitor.detect_anomalies()
        assert len(anomalies) == 0


class TestGetCurrentStats:
    """Tests for current stats reporting."""

    def test_empty_stats(self):
        """Empty monitor should return empty dict."""
        mon = PredictionMonitor()
        assert mon.get_current_stats() == {}

    def test_stats_with_data(self, monitor):
        """Stats should include confidence and distribution info."""
        for _ in range(50):
            monitor.log_prediction(1, 0.85)

        stats = monitor.get_current_stats()
        assert "confidence_mean" in stats
        assert "confidence_std" in stats
        assert "prediction_distribution" in stats
        assert stats["buffer_size"] == 50
