"""Main drift detection orchestrator combining Evidently and database logging."""

import logging

import pandas as pd

from src.drift.evidently_runner import EvidentlyRunner
from src.drift.reference_manager import ReferenceManager
from src.utils.config import DriftConfig
from src.utils.database import MetricsDB

logger = logging.getLogger(__name__)


class DriftDetector:
    """Orchestrate drift detection using Evidently and log results.

    Combines reference data management, Evidently drift analysis,
    and database persistence into a single high-level interface.

    Args:
        config: Drift detection configuration.
        db: Metrics database for persisting drift scores.
        reference_manager: Manager for reference dataset access.
    """

    def __init__(
        self,
        config: DriftConfig,
        db: MetricsDB,
        reference_manager: ReferenceManager,
    ) -> None:
        self.config = config
        self.db = db
        self.reference_manager = reference_manager
        self.evidently = EvidentlyRunner(
            stattest=config.stattest,
            threshold=config.threshold,
        )
        self._reference_data: pd.DataFrame | None = None
        self._reference_stats: dict | None = None

    def initialize(self, reference_name: str = "wine") -> None:
        """Load reference data for comparison.

        Args:
            reference_name: Name of the saved reference dataset.
        """
        self._reference_data, self._reference_stats = self.reference_manager.load_reference(
            reference_name
        )
        logger.info(
            "DriftDetector initialized with reference '%s' (shape=%s)",
            reference_name,
            self._reference_data.shape,
        )

    @property
    def is_initialized(self) -> bool:
        """Whether reference data has been loaded."""
        return self._reference_data is not None

    def detect(self, current_data: pd.DataFrame) -> dict:
        """Run drift detection and log results to the database.

        Args:
            current_data: Current production data to compare against reference.

        Returns:
            Dictionary with 'drift_detected' boolean and per-feature 'scores'.

        Raises:
            RuntimeError: If the detector has not been initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("DriftDetector not initialized. Call initialize() first.")

        report = self.evidently.run_drift_report(self._reference_data, current_data)
        scores = self.evidently.get_feature_drift_scores(report)
        dataset_drift = self.evidently.get_dataset_drift(report)

        for feature, score in scores.items():
            detected = score < self.config.threshold
            self.db.log_drift_score(feature, score, drift_detected=detected)

        logger.info(
            "Drift detection complete: dataset_drift=%s, features_checked=%d",
            dataset_drift,
            len(scores),
        )
        return {
            "drift_detected": dataset_drift,
            "scores": scores,
            "drifted_features": [f for f, s in scores.items() if s < self.config.threshold],
        }

    def detect_target_drift(
        self,
        current_data: pd.DataFrame,
        target_col: str = "target",
    ) -> dict:
        """Run target drift detection.

        Args:
            current_data: Current data containing a target column.
            target_col: Name of the target column.

        Returns:
            Target drift analysis results.

        Raises:
            RuntimeError: If the detector has not been initialized.
        """
        if not self.is_initialized:
            raise RuntimeError("DriftDetector not initialized. Call initialize() first.")

        return self.evidently.run_target_drift(
            self._reference_data,
            current_data,
            target_col=target_col,
        )

    def get_reference_stats(self) -> dict | None:
        """Return the loaded reference statistics, if available."""
        return self._reference_stats
