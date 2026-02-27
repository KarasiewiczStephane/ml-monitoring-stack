"""Prometheus metrics definitions for ML model monitoring."""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)


class MLMetrics:
    """Custom Prometheus metrics for ML model monitoring.

    Provides histograms, counters, and gauges for tracking prediction
    latency, accuracy, drift scores, data quality, and error rates.

    Args:
        registry: Prometheus collector registry. Defaults to the global
            REGISTRY.
    """

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.registry = registry

        self.prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Time spent processing prediction request",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry,
        )

        self.predictions_total = Counter(
            "model_predictions_total",
            "Total number of predictions",
            ["predicted_class", "model_name"],
            registry=registry,
        )

        self.accuracy_score = Gauge(
            "model_accuracy_score",
            "Current model accuracy",
            ["model_name"],
            registry=registry,
        )

        self.drift_score = Gauge(
            "data_drift_score",
            "Drift score for feature",
            ["feature_name", "model_name"],
            registry=registry,
        )

        self.data_quality_score = Gauge(
            "data_quality_score",
            "Overall data quality score (0-1)",
            ["model_name"],
            registry=registry,
        )

        self.error_rate = Gauge(
            "model_error_rate",
            "Model error rate",
            ["model_name", "error_type"],
            registry=registry,
        )

        self.predictions_in_flight = Gauge(
            "model_predictions_in_flight",
            "Number of predictions currently being processed",
            ["model_name"],
            registry=registry,
        )

    @contextmanager
    def track_prediction_time(self, model_name: str = "default") -> Generator[None, None, None]:
        """Context manager to track prediction latency.

        Args:
            model_name: Name of the model being tracked.

        Yields:
            None. Latency is recorded on exit.
        """
        self.predictions_in_flight.labels(model_name=model_name).inc()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.prediction_latency.observe(time.perf_counter() - start)
            self.predictions_in_flight.labels(model_name=model_name).dec()

    def record_prediction(self, predicted_class: str, model_name: str = "default") -> None:
        """Increment the prediction counter for a class.

        Args:
            predicted_class: The predicted class label.
            model_name: Name of the model.
        """
        self.predictions_total.labels(
            predicted_class=str(predicted_class),
            model_name=model_name,
        ).inc()

    def update_accuracy(self, accuracy: float, model_name: str = "default") -> None:
        """Update the current accuracy gauge.

        Args:
            accuracy: Current accuracy value.
            model_name: Name of the model.
        """
        self.accuracy_score.labels(model_name=model_name).set(accuracy)

    def update_drift_score(self, feature: str, score: float, model_name: str = "default") -> None:
        """Update the drift score gauge for a feature.

        Args:
            feature: Feature name.
            score: Drift score value.
            model_name: Name of the model.
        """
        self.drift_score.labels(
            feature_name=feature,
            model_name=model_name,
        ).set(score)

    def update_quality_score(self, score: float, model_name: str = "default") -> None:
        """Update the data quality score gauge.

        Args:
            score: Quality score between 0 and 1.
            model_name: Name of the model.
        """
        self.data_quality_score.labels(model_name=model_name).set(score)

    def update_error_rate(
        self,
        rate: float,
        error_type: str = "prediction",
        model_name: str = "default",
    ) -> None:
        """Update the error rate gauge.

        Args:
            rate: Error rate value.
            error_type: Type of error being tracked.
            model_name: Name of the model.
        """
        self.error_rate.labels(
            model_name=model_name,
            error_type=error_type,
        ).set(rate)

    def get_metrics_output(self) -> bytes:
        """Generate Prometheus-format metrics output.

        Returns:
            Encoded metrics string in Prometheus exposition format.
        """
        return generate_latest(self.registry)
