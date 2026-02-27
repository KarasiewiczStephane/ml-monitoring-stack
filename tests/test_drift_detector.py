"""Tests for drift detection using Evidently AI."""

import numpy as np
import pandas as pd
import pytest

from src.drift.detector import DriftDetector
from src.drift.evidently_runner import EvidentlyRunner
from src.drift.reference_manager import ReferenceManager
from src.utils.config import DriftConfig


@pytest.fixture
def evidently_runner() -> EvidentlyRunner:
    """Create an EvidentlyRunner with default settings."""
    return EvidentlyRunner(stattest="ks", threshold=0.05)


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Create a reference dataset with known distributions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_0": rng.standard_normal(300),
            "feature_1": rng.standard_normal(300),
            "feature_2": rng.standard_normal(300),
        }
    )


@pytest.fixture
def drifted_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataset with drift in feature_0."""
    drifted = reference_df.copy()
    drifted["feature_0"] += 3.0  # Large mean shift
    return drifted


@pytest.fixture
def drift_detector(tmp_path, mock_db) -> DriftDetector:
    """Create a DriftDetector with temporary reference storage."""
    config = DriftConfig()
    ref_manager = ReferenceManager(reference_dir=str(tmp_path / "reference"))

    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame(
        {
            "feature_0": rng.standard_normal(300),
            "feature_1": rng.standard_normal(300),
        }
    )
    ref_manager.save_reference(ref_df, "test_ref")

    detector = DriftDetector(config=config, db=mock_db, reference_manager=ref_manager)
    detector.initialize("test_ref")
    return detector


class TestEvidentlyRunner:
    """Tests for the EvidentlyRunner wrapper."""

    def test_drift_detected_with_shifted_data(self, evidently_runner, reference_df, drifted_df):
        """Drift should be detected when a feature is shifted by 3 std."""
        report = evidently_runner.run_drift_report(reference_df, drifted_df)
        scores = evidently_runner.get_feature_drift_scores(report)

        assert "feature_0" in scores
        # p-value for drifted feature should be very small
        assert scores["feature_0"] < 0.01

    def test_no_drift_with_same_distribution(self, evidently_runner, reference_df):
        """No drift should be detected with identical distributions."""
        report = evidently_runner.run_drift_report(reference_df, reference_df.copy())
        scores = evidently_runner.get_feature_drift_scores(report)

        for feature, score in scores.items():
            assert score > 0.05, f"False drift detected for {feature}"

    def test_get_dataset_drift(self, evidently_runner, reference_df, drifted_df):
        """get_dataset_drift should return a boolean."""
        report = evidently_runner.run_drift_report(reference_df, drifted_df)
        result = evidently_runner.get_dataset_drift(report)
        assert isinstance(result, bool)

    def test_per_feature_drift_detection(self, evidently_runner, reference_df, drifted_df):
        """Per-feature detection should flag only the drifted feature."""
        thresholds = {"feature_0": 0.05, "feature_1": 0.05, "feature_2": 0.05}
        result = evidently_runner.detect_drift_per_feature(
            reference_df, drifted_df, list(thresholds.keys()), thresholds
        )

        assert result["feature_0"]  # Drifted
        assert not result["feature_1"]  # Not drifted
        assert not result["feature_2"]  # Not drifted

    def test_target_drift_with_shifted_labels(self, evidently_runner):
        """Target drift should be detected when label distribution shifts."""
        rng = np.random.default_rng(42)
        ref = pd.DataFrame(
            {
                "f0": rng.standard_normal(300),
                "target": np.concatenate([np.zeros(150), np.ones(150)]),
            }
        )
        cur = pd.DataFrame(
            {
                "f0": rng.standard_normal(300),
                "target": np.concatenate([np.zeros(50), np.ones(250)]),
            }
        )

        result = evidently_runner.run_target_drift(ref, cur)
        assert result["drift_detected"] is True
        assert result["drift_score"] < 0.05


class TestDriftDetector:
    """Tests for the DriftDetector orchestrator."""

    def test_detect_returns_drift_status(self, drift_detector):
        """detect() should return drift_detected and scores."""
        rng = np.random.default_rng(99)
        current = pd.DataFrame(
            {
                "feature_0": rng.standard_normal(300) + 3.0,
                "feature_1": rng.standard_normal(300),
            }
        )

        result = drift_detector.detect(current)
        assert "drift_detected" in result
        assert "scores" in result
        assert "drifted_features" in result
        assert isinstance(result["scores"], dict)

    def test_detect_logs_to_database(self, drift_detector, mock_db):
        """Drift scores should be persisted to the database."""
        rng = np.random.default_rng(99)
        current = pd.DataFrame(
            {
                "feature_0": rng.standard_normal(300),
                "feature_1": rng.standard_normal(300),
            }
        )

        drift_detector.detect(current)
        scores = mock_db.get_drift_scores()
        assert len(scores) > 0

    def test_detect_without_init_raises(self, tmp_path, mock_db):
        """Calling detect before initialize should raise RuntimeError."""
        config = DriftConfig()
        ref_manager = ReferenceManager(reference_dir=str(tmp_path / "ref"))
        detector = DriftDetector(config=config, db=mock_db, reference_manager=ref_manager)

        with pytest.raises(RuntimeError, match="not initialized"):
            detector.detect(pd.DataFrame({"x": [1.0]}))

    def test_is_initialized_property(self, tmp_path, mock_db):
        """is_initialized should reflect initialization state."""
        config = DriftConfig()
        ref_manager = ReferenceManager(reference_dir=str(tmp_path / "ref"))
        detector = DriftDetector(config=config, db=mock_db, reference_manager=ref_manager)

        assert not detector.is_initialized

    def test_reference_stats_available(self, drift_detector):
        """Reference stats should be accessible after initialization."""
        stats = drift_detector.get_reference_stats()
        assert stats is not None
        assert "feature_0" in stats
