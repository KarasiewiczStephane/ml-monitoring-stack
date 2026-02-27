"""Sliding window drift detection for streaming data."""

import logging
from collections import deque
from datetime import UTC, datetime

import pandas as pd

from src.drift.evidently_runner import EvidentlyRunner

logger = logging.getLogger(__name__)


class SlidingWindowDriftDetector:
    """Detect drift on streaming data using a sliding window approach.

    Maintains a buffer of recent samples and periodically runs drift
    detection against a reference dataset when the buffer is full.

    Args:
        reference_data: Baseline dataset for comparison.
        window_size: Number of samples in the sliding window.
        step_size: Number of new samples before triggering a check.
        threshold: P-value threshold for drift detection.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        window_size: int = 1000,
        step_size: int = 100,
        threshold: float = 0.05,
    ) -> None:
        self.reference = reference_data
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold
        self.buffer: deque = deque(maxlen=window_size)
        self.evidently = EvidentlyRunner(threshold=threshold)
        self._samples_since_check: int = 0
        self._last_drift_result: dict | None = None

    def add_sample(self, sample: dict) -> dict | None:
        """Add a single sample and check drift if the window is ready.

        Args:
            sample: A dictionary representing one data point, keyed by
                feature name.

        Returns:
            Drift result dictionary if a check was triggered, else None.
        """
        self.buffer.append(sample)
        self._samples_since_check += 1

        if len(self.buffer) >= self.window_size and self._samples_since_check >= self.step_size:
            return self._check_drift()
        return None

    def add_batch(self, samples: list[dict]) -> list[dict]:
        """Add a batch of samples, returning any drift results.

        Args:
            samples: List of data point dictionaries.

        Returns:
            List of drift result dictionaries (one per triggered check).
        """
        results: list[dict] = []
        for sample in samples:
            result = self.add_sample(sample)
            if result is not None:
                results.append(result)
        return results

    def _check_drift(self) -> dict:
        """Run drift detection on the current window contents.

        Returns:
            Dictionary with timestamp, drift status, and per-feature scores.
        """
        current_df = pd.DataFrame(list(self.buffer))
        report = self.evidently.run_drift_report(self.reference, current_df)
        scores = self.evidently.get_feature_drift_scores(report)
        dataset_drift = self.evidently.get_dataset_drift(report)

        self._samples_since_check = 0
        self._last_drift_result = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "window_size": len(self.buffer),
            "drift_detected": dataset_drift,
            "feature_scores": scores,
            "drifted_features": [f for f, s in scores.items() if s < self.threshold],
        }
        logger.info(
            "Window drift check: detected=%s, drifted_features=%d",
            dataset_drift,
            len(self._last_drift_result["drifted_features"]),
        )
        return self._last_drift_result

    def get_current_status(self) -> dict:
        """Get the current detector status without triggering a check.

        Returns:
            Dictionary with buffer size, readiness, and last result.
        """
        return {
            "buffer_size": len(self.buffer),
            "window_size": self.window_size,
            "ready": len(self.buffer) >= self.window_size,
            "samples_since_check": self._samples_since_check,
            "last_result": self._last_drift_result,
        }

    def reset(self) -> None:
        """Clear the buffer and reset internal state."""
        self.buffer.clear()
        self._samples_since_check = 0
        self._last_drift_result = None
        logger.info("Sliding window detector reset")
