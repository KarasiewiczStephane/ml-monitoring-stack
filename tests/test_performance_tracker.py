"""Tests for model performance tracking."""

import pytest

from src.performance.tracker import PerformanceTracker


class TestClassificationMetrics:
    """Tests for classification performance tracking."""

    def test_perfect_classification(self, mock_db):
        """Perfect predictions should yield 100% accuracy."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1, 2, 0, 1], [0, 1, 2, 0, 1])

        metrics = tracker.compute_metrics()
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_imperfect_classification(self, mock_db):
        """Metrics should reflect prediction errors."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1, 1, 0], [0, 0, 1, 0])

        metrics = tracker.compute_metrics()
        assert 0 < metrics["accuracy"] < 1.0
        assert metrics["sample_count"] == 4

    def test_per_class_metrics(self, mock_db):
        """Per-class breakdown should be computed."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1, 0, 1], [0, 1, 1, 0])

        metrics = tracker.compute_metrics()
        assert "per_class" in metrics
        assert "0" in metrics["per_class"]
        assert "1" in metrics["per_class"]
        for cls_metrics in metrics["per_class"].values():
            assert "precision" in cls_metrics
            assert "recall" in cls_metrics
            assert "f1" in cls_metrics

    def test_single_prediction(self, mock_db):
        """Metrics should work with a single prediction."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_prediction(1, 1, confidence=0.9)

        metrics = tracker.compute_metrics()
        assert metrics["accuracy"] == 1.0

    def test_confusion_matrix(self, mock_db):
        """Confusion matrix should have correct shape."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1, 0, 1], [0, 1, 1, 0])

        cm = tracker.get_confusion_matrix()
        assert cm.shape == (2, 2)


class TestRegressionMetrics:
    """Tests for regression performance tracking."""

    def test_perfect_regression(self, mock_db):
        """Perfect predictions should yield RMSE=0 and R²=1."""
        tracker = PerformanceTracker("regression", mock_db)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        tracker.log_batch(values, values)

        metrics = tracker.compute_metrics()
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)

    def test_imperfect_regression(self, mock_db):
        """Metrics should reflect prediction errors."""
        tracker = PerformanceTracker("regression", mock_db)
        tracker.log_batch([1.0, 2.0, 3.0], [1.1, 2.2, 2.8])

        metrics = tracker.compute_metrics()
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] < 1.0

    def test_confusion_matrix_raises_for_regression(self, mock_db):
        """Confusion matrix should not be available for regression."""
        tracker = PerformanceTracker("regression", mock_db)
        with pytest.raises(ValueError, match="classification"):
            tracker.get_confusion_matrix()


class TestBatchVsIndividual:
    """Tests for consistency between batch and individual logging."""

    def test_same_metrics(self, mock_db, tmp_path):
        """Batch and individual logging should produce identical metrics."""
        preds = [0, 1, 0, 1, 1]
        truths = [0, 0, 0, 1, 1]

        tracker_batch = PerformanceTracker("classification", mock_db)
        tracker_batch.log_batch(preds, truths)
        batch_metrics = tracker_batch.compute_metrics()

        from src.utils.database import MetricsDB

        db2 = MetricsDB(str(tmp_path / "db2.db"))
        tracker_individual = PerformanceTracker("classification", db2)
        for p, t in zip(preds, truths):
            tracker_individual.log_prediction(p, t)
        individual_metrics = tracker_individual.compute_metrics()

        assert batch_metrics["accuracy"] == individual_metrics["accuracy"]
        assert batch_metrics["f1_macro"] == individual_metrics["f1_macro"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_predictions(self, mock_db):
        """Empty tracker should return empty dict."""
        tracker = PerformanceTracker("classification", mock_db)
        assert tracker.compute_metrics() == {}

    def test_empty_confusion_matrix(self, mock_db):
        """Empty tracker should return empty confusion matrix."""
        tracker = PerformanceTracker("classification", mock_db)
        cm = tracker.get_confusion_matrix()
        assert len(cm) == 0

    def test_reset(self, mock_db):
        """Reset should clear all state."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1], [0, 1])
        tracker.reset()

        assert tracker.compute_metrics() == {}
        assert len(tracker.predictions) == 0

    def test_log_with_confidence(self, mock_db):
        """Confidences should be passed through to the database."""
        tracker = PerformanceTracker("classification", mock_db)
        tracker.log_batch([0, 1], [0, 1], confidences=[0.9, 0.8])

        predictions = mock_db.get_predictions()
        assert len(predictions) == 2
