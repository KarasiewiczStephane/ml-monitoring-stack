"""Tests for Prometheus metrics implementation."""

import time

import pytest
from prometheus_client import CollectorRegistry

from src.metrics.collector import MetricsCollector
from src.metrics.prometheus_metrics import MLMetrics


@pytest.fixture
def registry() -> CollectorRegistry:
    """Create a fresh Prometheus registry for each test."""
    return CollectorRegistry()


@pytest.fixture
def metrics(registry) -> MLMetrics:
    """Create MLMetrics with an isolated registry."""
    return MLMetrics(registry=registry)


class TestMLMetrics:
    """Tests for Prometheus metric definitions."""

    def test_prediction_counter(self, metrics):
        """Prediction counter should increment correctly."""
        metrics.record_prediction("class_0", "model_a")
        metrics.record_prediction("class_0", "model_a")
        metrics.record_prediction("class_1", "model_a")

        value = metrics.predictions_total.labels(
            predicted_class="class_0", model_name="model_a"
        )._value.get()
        assert value == 2.0

    def test_accuracy_gauge(self, metrics):
        """Accuracy gauge should reflect the last set value."""
        metrics.update_accuracy(0.95, "model_a")
        value = metrics.accuracy_score.labels(model_name="model_a")._value.get()
        assert value == 0.95

    def test_drift_score_gauge(self, metrics):
        """Drift score gauge should be set per feature."""
        metrics.update_drift_score("feature_0", 0.02, "model_a")
        metrics.update_drift_score("feature_1", 0.15, "model_a")

        f0 = metrics.drift_score.labels(feature_name="feature_0", model_name="model_a")._value.get()
        f1 = metrics.drift_score.labels(feature_name="feature_1", model_name="model_a")._value.get()
        assert f0 == 0.02
        assert f1 == 0.15

    def test_quality_score_gauge(self, metrics):
        """Quality score gauge should be set correctly."""
        metrics.update_quality_score(0.98, "model_a")
        value = metrics.data_quality_score.labels(model_name="model_a")._value.get()
        assert value == 0.98

    def test_error_rate_gauge(self, metrics):
        """Error rate gauge should support different error types."""
        metrics.update_error_rate(0.05, "prediction", "model_a")
        value = metrics.error_rate.labels(
            model_name="model_a", error_type="prediction"
        )._value.get()
        assert value == 0.05

    def test_latency_histogram(self, metrics):
        """Latency histogram should record observations."""
        metrics.prediction_latency.observe(0.015)
        metrics.prediction_latency.observe(0.025)

        # Verify the count of observations
        output = metrics.get_metrics_output().decode()
        assert "model_prediction_latency_seconds_count 2.0" in output

    def test_track_prediction_time_context_manager(self, metrics):
        """Context manager should record latency and manage in-flight counter."""
        with metrics.track_prediction_time("model_a"):
            time.sleep(0.01)

        # Should have recorded one observation
        output = metrics.get_metrics_output().decode()
        assert "model_prediction_latency_seconds_count 1.0" in output

        # In-flight should be back to 0
        in_flight = metrics.predictions_in_flight.labels(model_name="model_a")._value.get()
        assert in_flight == 0.0

    def test_metrics_output_format(self, metrics):
        """Output should be valid Prometheus exposition format."""
        metrics.record_prediction("class_0")
        metrics.update_accuracy(0.9)

        output = metrics.get_metrics_output()
        assert isinstance(output, bytes)
        text = output.decode()
        assert "model_predictions_total" in text
        assert "model_accuracy_score" in text


class TestMetricsCollector:
    """Tests for the periodic metrics collector."""

    def test_collect_once_no_error(self, metrics):
        """collect_once should run without error when no detectors set."""
        collector = MetricsCollector(metrics=metrics)
        collector.collect_once()

    def test_collect_with_performance_tracker(self, metrics, mock_db):
        """Collector should update accuracy from performance tracker."""
        from src.performance.tracker import PerformanceTracker

        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1, 0, 1], [0, 1, 0, 1])

        collector = MetricsCollector(
            metrics=metrics,
            performance_tracker=tracker,
            model_name="test_model",
        )
        collector.collect_once()

        value = metrics.accuracy_score.labels(model_name="test_model")._value.get()
        assert value == 1.0
