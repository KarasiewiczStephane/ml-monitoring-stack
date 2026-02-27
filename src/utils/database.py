"""SQLite and Redis database utilities for metrics storage."""

import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsDB:
    """SQLite database for storing metrics history.

    Manages tables for drift scores, performance metrics, and predictions
    with automatic schema initialization.

    Args:
        db_path: Filesystem path for the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections.

        Yields:
            An active SQLite connection with row factory set.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create database tables if they do not already exist."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS drift_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    feature_name TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    drift_detected BOOLEAN NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    prediction TEXT,
                    confidence REAL,
                    ground_truth TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_drift_feature
                    ON drift_scores(feature_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_perf_metric
                    ON performance_metrics(metric_name, timestamp);
            """
            )
        logger.info("Database schema initialized at %s", self.db_path)

    def log_drift_score(
        self,
        feature_name: str,
        drift_score: float,
        drift_detected: bool = False,
    ) -> None:
        """Record a drift score measurement.

        Args:
            feature_name: Name of the feature being measured.
            drift_score: The computed drift score value.
            drift_detected: Whether drift was detected for this feature.
        """
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO drift_scores (feature_name, drift_score, drift_detected) "
                "VALUES (?, ?, ?)",
                (feature_name, drift_score, drift_detected),
            )

    def log_performance_metric(self, metric_name: str, metric_value: float) -> None:
        """Record a performance metric measurement.

        Args:
            metric_name: Name of the metric (e.g. 'accuracy', 'f1').
            metric_value: The metric value.
        """
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO performance_metrics (metric_name, metric_value) VALUES (?, ?)",
                (metric_name, metric_value),
            )

    def log_prediction(
        self,
        prediction: Any,
        confidence: float | None = None,
        ground_truth: Any | None = None,
    ) -> None:
        """Record a model prediction.

        Args:
            prediction: The predicted value or class.
            confidence: Optional confidence score for the prediction.
            ground_truth: Optional ground truth label.
        """
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO predictions (prediction, confidence, ground_truth) VALUES (?, ?, ?)",
                (
                    str(prediction),
                    confidence,
                    str(ground_truth) if ground_truth is not None else None,
                ),
            )

    def get_drift_scores(
        self,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve drift score history.

        Args:
            feature_name: Filter by feature name, or None for all features.
            limit: Maximum number of records to return.

        Returns:
            List of drift score records as dictionaries.
        """
        with self._connection() as conn:
            if feature_name:
                cursor = conn.execute(
                    "SELECT * FROM drift_scores WHERE feature_name = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (feature_name, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM drift_scores ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_metrics(
        self,
        metric_name: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve performance metric history.

        Args:
            metric_name: Filter by metric name, or None for all metrics.
            limit: Maximum number of records to return.

        Returns:
            List of performance metric records as dictionaries.
        """
        with self._connection() as conn:
            if metric_name:
                cursor = conn.execute(
                    "SELECT * FROM performance_metrics WHERE metric_name = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (metric_name, limit),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM performance_metrics ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_predictions(self, limit: int = 100) -> list[dict]:
        """Retrieve prediction history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of prediction records as dictionaries.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_drift_history(self, hours: int = 24) -> list[dict]:
        """Retrieve drift scores from the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            List of drift score records.
        """
        cutoff = datetime.now().isoformat()
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM drift_scores WHERE timestamp >= datetime(?, ?) "
                "ORDER BY timestamp ASC",
                (cutoff, f"-{hours} hours"),
            )
            return [dict(row) for row in cursor.fetchall()]


class RedisClient:
    """Redis client wrapper for real-time state management.

    Provides high-level methods for storing and retrieving drift scores
    and other monitoring state in Redis.

    Args:
        url: Redis connection URL (e.g. 'redis://localhost:6379').
    """

    def __init__(self, url: str) -> None:
        try:
            import redis

            self.client = redis.from_url(url, decode_responses=True)
            self._available = True
            logger.info("Redis client connected to %s", url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s), falling back to in-memory", exc)
            self._available = False
            self._memory_store: dict[str, dict[str, str]] = {}

    @property
    def available(self) -> bool:
        """Whether the Redis connection is active."""
        return self._available

    def set_drift_score(self, feature: str, score: float) -> None:
        """Store a drift score for a feature.

        Args:
            feature: Feature name.
            score: Drift score value.
        """
        if self._available:
            self.client.hset("drift_scores", feature, str(score))
        else:
            self._memory_store.setdefault("drift_scores", {})[feature] = str(score)

    def get_drift_scores(self) -> dict[str, float]:
        """Retrieve all current drift scores.

        Returns:
            Dictionary mapping feature names to drift scores.
        """
        if self._available:
            raw = self.client.hgetall("drift_scores")
        else:
            raw = self._memory_store.get("drift_scores", {})
        return {k: float(v) for k, v in raw.items()}

    def set_value(self, key: str, value: str) -> None:
        """Store a simple key-value pair.

        Args:
            key: The key name.
            value: The string value to store.
        """
        if self._available:
            self.client.set(key, value)
        else:
            self._memory_store.setdefault("_keys", {})[key] = value

    def get_value(self, key: str) -> str | None:
        """Retrieve a value by key.

        Args:
            key: The key name.

        Returns:
            The stored string value or None if not found.
        """
        if self._available:
            return self.client.get(key)
        return self._memory_store.get("_keys", {}).get(key)
