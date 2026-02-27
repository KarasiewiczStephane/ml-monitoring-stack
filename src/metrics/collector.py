"""Periodic metrics collection and Prometheus gauge updates."""

import asyncio
import logging

from src.metrics.prometheus_metrics import MLMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Periodically collect metrics and update Prometheus gauges.

    Bridges drift detection and performance tracking with Prometheus
    by reading the latest values and updating gauges at a configured
    interval.

    Args:
        metrics: MLMetrics instance for Prometheus gauge updates.
        drift_detector: Optional drift detector to read scores from.
        performance_tracker: Optional performance tracker to read metrics from.
        model_name: Model name label for Prometheus metrics.
    """

    def __init__(
        self,
        metrics: MLMetrics,
        drift_detector: object | None = None,
        performance_tracker: object | None = None,
        model_name: str = "default",
    ) -> None:
        self.metrics = metrics
        self.drift_detector = drift_detector
        self.performance_tracker = performance_tracker
        self.model_name = model_name

    async def collect_loop(self, interval: int = 60) -> None:
        """Run the collection loop at the specified interval.

        Args:
            interval: Seconds between collection cycles.
        """
        while True:
            self._collect_metrics()
            await asyncio.sleep(interval)

    def _collect_metrics(self) -> None:
        """Gather current metrics from detectors and update Prometheus."""
        if self.performance_tracker is not None:
            try:
                perf = self.performance_tracker.compute_metrics()
                if "accuracy" in perf:
                    self.metrics.update_accuracy(perf["accuracy"], self.model_name)
            except Exception:
                logger.exception("Error collecting performance metrics")

        if self.drift_detector is not None:
            try:
                stats = getattr(self.drift_detector, "_reference_stats", None)
                if stats:
                    logger.debug("Drift stats available for collection")
            except Exception:
                logger.exception("Error collecting drift metrics")

    def collect_once(self) -> None:
        """Run a single collection cycle (non-async convenience method)."""
        self._collect_metrics()
