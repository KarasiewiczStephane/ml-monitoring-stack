"""Configuration management with YAML support and Pydantic validation."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class DriftConfig(BaseSettings):
    """Configuration for drift detection parameters."""

    window_size: int = 1000
    step_size: int = 100
    threshold: float = 0.05
    stattest: str = "ks"
    features: dict[str, float] = Field(default_factory=dict)


class PerformanceConfig(BaseSettings):
    """Configuration for performance monitoring."""

    task_type: str = "classification"
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    ph_threshold: float = 50.0
    ph_delta: float = 0.005
    baseline_accuracy: float = 0.9


class DatabaseConfig(BaseSettings):
    """Configuration for database connections."""

    sqlite_path: str = "data/metrics.db"
    redis_url: str = "redis://localhost:6379"


class APIConfig(BaseSettings):
    """Configuration for the FastAPI application."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "default_model"


class MetricsConfig(BaseSettings):
    """Configuration for Prometheus metrics collection."""

    collection_interval: int = 60
    latency_buckets: list[float] = Field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    )


class Settings(BaseSettings):
    """Root application settings combining all config sections."""

    drift: DriftConfig = Field(default_factory=DriftConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reference_dataset: str = "wine"
    reference_dir: str = "data/reference"


def load_config(path: str = "configs/config.yaml") -> Settings:
    """Load application settings from a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Populated Settings instance with values from the file
        merged with defaults.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return Settings()

    with open(config_path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    logger.info("Loaded configuration from %s", path)
    return Settings(**data)
