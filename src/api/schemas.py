"""Pydantic schemas for API request and response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for logging a prediction.

    Attributes:
        features: Dictionary of feature name to value mappings.
        model_name: Name of the model making the prediction.
        ground_truth: Optional true label for the prediction.
    """

    features: dict[str, float]
    model_name: str = "default"
    ground_truth: Any | None = None


class PredictionResponse(BaseModel):
    """Response schema for a logged prediction.

    Attributes:
        prediction: The predicted value (mocked in monitoring mode).
        confidence: Confidence score for the prediction.
        timestamp: When the prediction was recorded.
        drift_warning: Whether current drift exceeds threshold.
    """

    prediction: Any
    confidence: float
    timestamp: datetime
    drift_warning: bool = False


class DriftStatus(BaseModel):
    """Response schema for drift detection status.

    Attributes:
        drift_detected: Whether overall dataset drift was detected.
        overall_score: Aggregate drift score.
        feature_scores: Per-feature drift scores.
        drifted_features: List of features exceeding drift threshold.
        last_check: Timestamp of the last drift check.
    """

    drift_detected: bool
    overall_score: float
    feature_scores: dict[str, float]
    drifted_features: list[str]
    last_check: datetime


class PerformanceStatus(BaseModel):
    """Response schema for model performance metrics.

    Attributes:
        accuracy: Current accuracy score.
        precision: Current precision score.
        recall: Current recall score.
        f1: Current F1 score.
        sample_count: Number of predictions evaluated.
        degradation_alerts: List of active degradation alerts.
    """

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    sample_count: int = 0
    degradation_alerts: list[str] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    """Request schema for running a simulation scenario.

    Attributes:
        duration_override: Optional duration override in seconds.
        parameters_override: Optional parameter overrides.
    """

    duration_override: int | None = None
    parameters_override: dict | None = None


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint.

    Attributes:
        status: Overall service status.
        version: Application version string.
        uptime_seconds: Seconds since service start.
        components: Status of individual components.
    """

    status: str
    version: str
    uptime_seconds: float
    components: dict[str, str]
