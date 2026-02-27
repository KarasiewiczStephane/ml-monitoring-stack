"""Prediction distribution monitoring without ground truth."""

import logging
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PredictionAnomaly:
    """Represents an anomaly detected in prediction patterns.

    Attributes:
        anomaly_type: Type of anomaly (confidence_shift, class_imbalance).
        timestamp: ISO timestamp when the anomaly was detected.
        details: Dictionary with specific anomaly information.
        severity: Alert severity level (low, medium, high).
    """

    anomaly_type: str
    timestamp: str
    details: dict
    severity: str


class PredictionMonitor:
    """Monitor prediction patterns without requiring ground truth.

    Tracks confidence score distributions and class balance to detect
    shifts from a baseline period.

    Args:
        window_size: Number of recent predictions to monitor.
        reference_distribution: Baseline class distribution as proportions.
    """

    def __init__(
        self,
        window_size: int = 1000,
        reference_distribution: dict | None = None,
    ) -> None:
        self.window_size = window_size
        self.reference_distribution = reference_distribution
        self.confidence_buffer: deque = deque(maxlen=window_size)
        self.prediction_buffer: deque = deque(maxlen=window_size)
        self._baseline_confidence_stats: dict | None = None

    def set_baseline(self, confidences: list[float], predictions: list) -> None:
        """Set baseline statistics from a reference period.

        Args:
            confidences: Confidence scores from the reference period.
            predictions: Predictions from the reference period.
        """
        self._baseline_confidence_stats = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "p25": float(np.percentile(confidences, 25)),
            "p75": float(np.percentile(confidences, 75)),
        }
        pred_counts = Counter(predictions)
        total = len(predictions)
        self.reference_distribution = {str(k): v / total for k, v in pred_counts.items()}
        logger.info("Baseline set from %d samples", total)

    def log_prediction(self, prediction: object, confidence: float) -> None:
        """Log a prediction for monitoring.

        Args:
            prediction: Predicted class or value.
            confidence: Confidence score for the prediction.
        """
        self.confidence_buffer.append(confidence)
        self.prediction_buffer.append(prediction)

    def check_confidence_distribution(self) -> PredictionAnomaly | None:
        """Check if the confidence score distribution has shifted.

        Uses a Z-test comparing the current mean confidence against
        the baseline.

        Returns:
            A PredictionAnomaly if a significant shift is detected,
            else None.
        """
        if len(self.confidence_buffer) < 100 or not self._baseline_confidence_stats:
            return None

        current_mean = float(np.mean(self.confidence_buffer))
        baseline = self._baseline_confidence_stats

        if baseline["std"] == 0:
            return None

        z_score = abs(current_mean - baseline["mean"]) / (
            baseline["std"] / np.sqrt(len(self.confidence_buffer))
        )

        if z_score > 3.0:
            return PredictionAnomaly(
                anomaly_type="confidence_shift",
                timestamp=datetime.now(tz=UTC).isoformat(),
                details={
                    "baseline_mean": baseline["mean"],
                    "current_mean": current_mean,
                    "z_score": float(z_score),
                },
                severity="high" if z_score > 5.0 else "medium",
            )
        return None

    def check_class_balance(self) -> PredictionAnomaly | None:
        """Check if the prediction class distribution has shifted.

        Uses a chi-square test comparing current distribution against
        the baseline.

        Returns:
            A PredictionAnomaly if a significant shift is detected,
            else None.
        """
        if len(self.prediction_buffer) < 100 or not self.reference_distribution:
            return None

        current_counts = Counter(self.prediction_buffer)
        total = len(self.prediction_buffer)
        current_dist = {str(k): v / total for k, v in current_counts.items()}

        classes = sorted(set(self.reference_distribution.keys()) | set(current_dist.keys()))
        observed = [current_dist.get(c, 0) * total for c in classes]
        expected = [max(self.reference_distribution.get(c, 0.01) * total, 1) for c in classes]

        chi2, p_value = stats.chisquare(observed, expected)

        if p_value < 0.01:
            return PredictionAnomaly(
                anomaly_type="class_imbalance",
                timestamp=datetime.now(tz=UTC).isoformat(),
                details={
                    "reference_dist": self.reference_distribution,
                    "current_dist": current_dist,
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                },
                severity="high" if p_value < 0.001 else "medium",
            )
        return None

    def detect_anomalies(self) -> list[PredictionAnomaly]:
        """Run all anomaly detection checks.

        Returns:
            List of detected anomalies (may be empty).
        """
        anomalies: list[PredictionAnomaly] = []
        conf_anomaly = self.check_confidence_distribution()
        if conf_anomaly:
            anomalies.append(conf_anomaly)
        balance_anomaly = self.check_class_balance()
        if balance_anomaly:
            anomalies.append(balance_anomaly)
        return anomalies

    def get_current_stats(self) -> dict:
        """Get current prediction statistics.

        Returns:
            Dictionary with current confidence and distribution stats.
        """
        if not self.confidence_buffer:
            return {}
        return {
            "confidence_mean": float(np.mean(self.confidence_buffer)),
            "confidence_std": float(np.std(self.confidence_buffer)),
            "prediction_distribution": dict(Counter(self.prediction_buffer)),
            "buffer_size": len(self.confidence_buffer),
        }
