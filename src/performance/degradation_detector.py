"""Statistical methods for detecting gradual performance degradation."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class DegradationAlert:
    """Represents a detected performance degradation event.

    Attributes:
        method: Detection method name (CUSUM or PageHinkley).
        timestamp: ISO timestamp when the alert was raised.
        metric_name: Name of the metric being tracked.
        current_value: The statistic value that triggered the alert.
        threshold: The configured threshold that was exceeded.
        message: Human-readable description of the alert.
    """

    method: str
    timestamp: str
    metric_name: str
    current_value: float
    threshold: float
    message: str


class CUSUMDetector:
    """CUSUM (Cumulative Sum) detector for mean shift detection.

    Tracks both positive and negative cumulative sums to detect
    upward and downward shifts in a monitored metric.

    Args:
        target_mean: Expected mean value of the metric.
        threshold: CUSUM threshold for triggering an alert.
        drift_allowance: Allowable slack before accumulating.
    """

    def __init__(
        self,
        target_mean: float,
        threshold: float = 5.0,
        drift_allowance: float = 0.5,
    ) -> None:
        self.target_mean = target_mean
        self.threshold = threshold
        self.drift_allowance = drift_allowance
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.values: list[float] = []

    def update(self, value: float) -> DegradationAlert | None:
        """Update CUSUM with a new observation.

        Args:
            value: New metric observation.

        Returns:
            A DegradationAlert if the threshold was exceeded, else None.
        """
        self.values.append(value)
        self.s_pos = max(0, self.s_pos + value - self.target_mean - self.drift_allowance)
        self.s_neg = max(0, self.s_neg - value + self.target_mean - self.drift_allowance)

        if self.s_pos > self.threshold:
            alert = DegradationAlert(
                method="CUSUM",
                timestamp=datetime.now(tz=UTC).isoformat(),
                metric_name="performance",
                current_value=self.s_pos,
                threshold=self.threshold,
                message=f"Positive shift detected: CUSUM={self.s_pos:.3f}",
            )
            self.s_pos = 0.0
            return alert

        if self.s_neg > self.threshold:
            alert = DegradationAlert(
                method="CUSUM",
                timestamp=datetime.now(tz=UTC).isoformat(),
                metric_name="performance",
                current_value=self.s_neg,
                threshold=self.threshold,
                message=f"Negative shift detected: CUSUM={self.s_neg:.3f}",
            )
            self.s_neg = 0.0
            return alert

        return None

    def reset(self) -> None:
        """Reset CUSUM accumulators and value history."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.values.clear()


class PageHinkleyDetector:
    """Page-Hinkley test for gradual drift detection.

    Detects slow changes in the mean of a sequence by tracking the
    cumulative deviation from a running mean.

    Args:
        delta: Magnitude of allowed change per observation.
        threshold: Page-Hinkley threshold for triggering an alert.
        alpha: Forgetting factor for the running mean.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.cumulative_sum: float = 0.0
        self.min_sum: float = float("inf")
        self.count: int = 0
        self.mean: float = 0.0

    def update(self, value: float) -> DegradationAlert | None:
        """Update the Page-Hinkley test with a new observation.

        Args:
            value: New metric observation.

        Returns:
            A DegradationAlert if the threshold was exceeded, else None.
        """
        self.count += 1
        self.mean = self.mean + (value - self.mean) / self.count
        self.cumulative_sum = self.alpha * self.cumulative_sum + (value - self.mean - self.delta)
        self.min_sum = min(self.min_sum, self.cumulative_sum)

        ph_value = self.cumulative_sum - self.min_sum
        if ph_value > self.threshold:
            alert = DegradationAlert(
                method="PageHinkley",
                timestamp=datetime.now(tz=UTC).isoformat(),
                metric_name="performance",
                current_value=ph_value,
                threshold=self.threshold,
                message=f"Gradual drift detected: PH={ph_value:.3f}",
            )
            self.cumulative_sum = 0.0
            self.min_sum = float("inf")
            return alert

        return None

    def reset(self) -> None:
        """Reset the Page-Hinkley test state."""
        self.cumulative_sum = 0.0
        self.min_sum = float("inf")
        self.count = 0
        self.mean = 0.0


class DegradationDetector:
    """Combined degradation detection using CUSUM and Page-Hinkley.

    Runs both detectors in parallel and collects alerts from either.

    Args:
        baseline_accuracy: Expected baseline metric value.
        config: Configuration dict with optional keys 'cusum_threshold',
            'cusum_drift', 'ph_threshold', 'ph_delta'.
    """

    def __init__(self, baseline_accuracy: float, config: dict | None = None) -> None:
        config = config or {}
        self.cusum = CUSUMDetector(
            target_mean=baseline_accuracy,
            threshold=config.get("cusum_threshold", 5.0),
            drift_allowance=config.get("cusum_drift", 0.5),
        )
        self.page_hinkley = PageHinkleyDetector(
            threshold=config.get("ph_threshold", 50.0),
            delta=config.get("ph_delta", 0.005),
        )
        self.alerts: list[DegradationAlert] = field(default_factory=list)
        self.alerts = []

    def update(self, accuracy: float) -> list[DegradationAlert]:
        """Update all detectors with a new metric observation.

        Args:
            accuracy: Current metric value (e.g. accuracy score).

        Returns:
            List of alerts raised by any detector.
        """
        new_alerts: list[DegradationAlert] = []
        cusum_alert = self.cusum.update(accuracy)
        if cusum_alert:
            new_alerts.append(cusum_alert)
        ph_alert = self.page_hinkley.update(accuracy)
        if ph_alert:
            new_alerts.append(ph_alert)
        self.alerts.extend(new_alerts)
        return new_alerts

    def reset(self) -> None:
        """Reset all detectors and clear alert history."""
        self.cusum.reset()
        self.page_hinkley.reset()
        self.alerts.clear()
