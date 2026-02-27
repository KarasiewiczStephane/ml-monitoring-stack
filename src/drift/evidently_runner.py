"""Evidently AI integration for dataset and target drift detection."""

import logging

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)


class EvidentlyRunner:
    """Run drift analysis using Evidently AI reports.

    Wraps Evidently's legacy Report API to provide a simplified interface
    for dataset-level and per-feature drift detection.

    Args:
        stattest: Statistical test to use ('ks', 'psi', 'wasserstein').
        threshold: P-value threshold for drift detection.
    """

    def __init__(self, stattest: str = "ks", threshold: float = 0.05) -> None:
        self.stattest = stattest
        self.threshold = threshold

    def run_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        column_mapping: dict | None = None,
    ) -> dict:
        """Run a full dataset drift analysis.

        Args:
            reference: Baseline dataset.
            current: Current production dataset to compare.
            column_mapping: Optional column role mapping for Evidently.

        Returns:
            Dictionary containing the Evidently report results.
        """
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()
        logger.info("Drift report completed for %d columns", len(current.columns))
        return result

    def get_feature_drift_scores(self, report_dict: dict) -> dict[str, float]:
        """Extract per-feature drift scores (p-values) from a report.

        Args:
            report_dict: Output from ``run_drift_report``.

        Returns:
            Dictionary mapping feature names to p-value drift scores.
            Lower values indicate stronger evidence of drift.
        """
        scores: dict[str, float] = {}
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            drift_by_columns = result.get("drift_by_columns", {})
            if drift_by_columns:
                for col, data in drift_by_columns.items():
                    scores[col] = data.get("drift_score", 1.0)
        return scores

    def get_dataset_drift(self, report_dict: dict) -> bool:
        """Check whether overall dataset drift was detected.

        Args:
            report_dict: Output from ``run_drift_report``.

        Returns:
            True if dataset-level drift was detected.
        """
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            if "dataset_drift" in result:
                return bool(result["dataset_drift"])
        return False

    def run_target_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        target_col: str = "target",
    ) -> dict:
        """Detect prediction/target distribution shift.

        Args:
            reference: Baseline dataset with target column.
            current: Current dataset with target column.
            target_col: Name of the target column.

        Returns:
            Dictionary with target drift analysis results including
            'drift_score' and 'drift_detected'.
        """
        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result = report.as_dict()

        target_result = {}
        for metric in result.get("metrics", []):
            metric_result = metric.get("result", {})
            if metric_result.get("column_name") == target_col:
                target_result = {
                    "drift_score": metric_result.get("drift_score", 1.0),
                    "drift_detected": metric_result.get("drift_detected", False),
                    "stattest_name": metric_result.get("stattest_name", ""),
                }
                break

        logger.info(
            "Target drift analysis: detected=%s, score=%s",
            target_result.get("drift_detected"),
            target_result.get("drift_score"),
        )
        return target_result

    def detect_drift_per_feature(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: list[str],
        thresholds: dict[str, float],
    ) -> dict[str, bool]:
        """Check drift per feature against configurable thresholds.

        Uses p-value comparison: drift is detected when the p-value
        falls below the configured threshold for that feature.

        Args:
            reference: Baseline dataset.
            current: Current dataset.
            features: List of feature names to check.
            thresholds: Per-feature p-value thresholds.

        Returns:
            Dictionary mapping feature names to drift detection boolean.
        """
        report = self.run_drift_report(reference, current)
        scores = self.get_feature_drift_scores(report)
        return {
            feat: scores.get(feat, 1.0) < thresholds.get(feat, self.threshold) for feat in features
        }
