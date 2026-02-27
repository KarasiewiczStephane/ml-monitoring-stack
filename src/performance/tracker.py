"""Online model performance tracking with classification and regression metrics."""

import logging
from collections import defaultdict
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.utils.database import MetricsDB

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track model performance metrics online as predictions arrive.

    Supports classification (accuracy, precision, recall, F1) and
    regression (RMSE, MAE, R²) with per-class breakdown.

    Args:
        task_type: Either 'classification' or 'regression'.
        db: Database for persisting metrics.
    """

    def __init__(
        self,
        task_type: Literal["classification", "regression"],
        db: MetricsDB,
    ) -> None:
        self.task_type = task_type
        self.db = db
        self.predictions: list = []
        self.ground_truths: list = []
        self._per_class_metrics: dict = defaultdict(list)

    def log_prediction(
        self,
        prediction: object,
        ground_truth: object,
        confidence: float | None = None,
    ) -> None:
        """Log a single prediction with its ground truth.

        Args:
            prediction: Predicted value or class.
            ground_truth: True value or label.
            confidence: Optional confidence score.
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
        self.db.log_prediction(prediction, confidence, ground_truth)

    def log_batch(
        self,
        predictions: list,
        ground_truths: list,
        confidences: list | None = None,
    ) -> None:
        """Log a batch of predictions.

        Args:
            predictions: List of predicted values.
            ground_truths: List of true values.
            confidences: Optional list of confidence scores.
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            conf = confidences[i] if confidences else None
            self.db.log_prediction(pred, conf, gt)

    def compute_metrics(self) -> dict:
        """Compute current performance metrics.

        Returns:
            Dictionary of computed metrics appropriate for the task type.
            Returns empty dict if no predictions have been logged.
        """
        if not self.predictions:
            return {}

        if self.task_type == "classification":
            metrics = self._compute_classification_metrics()
        else:
            metrics = self._compute_regression_metrics()

        for name, value in metrics.items():
            if isinstance(value, int | float):
                self.db.log_performance_metric(name, float(value))

        logger.info("Computed %s metrics: %d predictions", self.task_type, len(self.predictions))
        return metrics

    def _compute_classification_metrics(self) -> dict:
        """Compute classification metrics with per-class breakdown."""
        y_true, y_pred = self.ground_truths, self.predictions
        metrics: dict = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "sample_count": len(y_true),
        }

        classes = sorted(set(y_true) | set(y_pred))
        per_class: dict = {}
        for cls in classes:
            cls_true = [1 if y == cls else 0 for y in y_true]
            cls_pred = [1 if y == cls else 0 for y in y_pred]
            per_class[str(cls)] = {
                "precision": float(precision_score(cls_true, cls_pred, zero_division=0)),
                "recall": float(recall_score(cls_true, cls_pred, zero_division=0)),
                "f1": float(f1_score(cls_true, cls_pred, zero_division=0)),
            }
        metrics["per_class"] = per_class
        return metrics

    def _compute_regression_metrics(self) -> dict:
        """Compute regression metrics."""
        y_true = np.array(self.ground_truths, dtype=float)
        y_pred = np.array(self.predictions, dtype=float)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "sample_count": len(y_true),
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix for classification tasks.

        Returns:
            Confusion matrix as a numpy array.

        Raises:
            ValueError: If task_type is not classification.
        """
        if self.task_type != "classification":
            raise ValueError("Confusion matrix only available for classification tasks")
        if not self.predictions:
            return np.array([])
        return confusion_matrix(self.ground_truths, self.predictions)

    def reset(self) -> None:
        """Clear all tracked predictions and ground truths."""
        self.predictions.clear()
        self.ground_truths.clear()
        self._per_class_metrics.clear()
        logger.info("Performance tracker reset")
