"""Tests for SQLite database operations."""

from src.utils.database import MetricsDB, RedisClient


class TestMetricsDB:
    """Tests for the MetricsDB SQLite wrapper."""

    def test_schema_creation(self, mock_db):
        """Database schema should be created on init."""
        with mock_db._connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row["name"] for row in cursor.fetchall()]

        assert "drift_scores" in tables
        assert "performance_metrics" in tables
        assert "predictions" in tables

    def test_log_and_retrieve_drift_score(self, mock_db):
        """Drift scores should be logged and retrievable."""
        mock_db.log_drift_score("feature_0", 0.15, drift_detected=True)
        mock_db.log_drift_score("feature_1", 0.02, drift_detected=False)

        all_scores = mock_db.get_drift_scores()
        assert len(all_scores) == 2

        feature_0_scores = mock_db.get_drift_scores(feature_name="feature_0")
        assert len(feature_0_scores) == 1
        assert feature_0_scores[0]["drift_score"] == 0.15
        assert feature_0_scores[0]["drift_detected"] == 1

    def test_log_and_retrieve_performance_metric(self, mock_db):
        """Performance metrics should be logged and retrievable."""
        mock_db.log_performance_metric("accuracy", 0.92)
        mock_db.log_performance_metric("f1", 0.88)

        metrics = mock_db.get_performance_metrics()
        assert len(metrics) == 2

        accuracy = mock_db.get_performance_metrics(metric_name="accuracy")
        assert len(accuracy) == 1
        assert accuracy[0]["metric_value"] == 0.92

    def test_log_and_retrieve_prediction(self, mock_db):
        """Predictions should be logged and retrievable."""
        mock_db.log_prediction("class_1", confidence=0.85, ground_truth="class_1")
        mock_db.log_prediction("class_0", confidence=0.6, ground_truth="class_1")

        predictions = mock_db.get_predictions()
        assert len(predictions) == 2
        confidences = {p["confidence"] for p in predictions}
        assert confidences == {0.85, 0.6}

    def test_limit_parameter(self, mock_db):
        """Limit parameter should restrict result count."""
        for i in range(10):
            mock_db.log_drift_score("feature_0", float(i) / 10)

        limited = mock_db.get_drift_scores(limit=5)
        assert len(limited) == 5

    def test_creates_parent_directory(self, tmp_path):
        """Database init should create parent directories as needed."""
        nested_path = tmp_path / "sub" / "dir" / "test.db"
        db = MetricsDB(str(nested_path))

        assert nested_path.exists()
        db.log_drift_score("test_feature", 0.1)
        scores = db.get_drift_scores()
        assert len(scores) == 1

    def test_prediction_with_none_ground_truth(self, mock_db):
        """Predictions without ground truth should be stored correctly."""
        mock_db.log_prediction("class_1", confidence=0.9)

        predictions = mock_db.get_predictions()
        assert len(predictions) == 1
        assert predictions[0]["ground_truth"] is None


class TestRedisClient:
    """Tests for the RedisClient wrapper (using in-memory fallback)."""

    def test_fallback_mode(self):
        """Client should fall back to in-memory store when Redis unavailable."""
        client = RedisClient("redis://localhost:99999")
        assert not client.available

    def test_set_and_get_drift_scores_fallback(self):
        """Drift scores should work via in-memory fallback."""
        client = RedisClient("redis://localhost:99999")
        client.set_drift_score("feature_0", 0.15)
        client.set_drift_score("feature_1", 0.02)

        scores = client.get_drift_scores()
        assert scores["feature_0"] == 0.15
        assert scores["feature_1"] == 0.02

    def test_set_and_get_value_fallback(self):
        """Simple key-value operations should work in fallback mode."""
        client = RedisClient("redis://localhost:99999")
        client.set_value("model_status", "healthy")

        assert client.get_value("model_status") == "healthy"
        assert client.get_value("nonexistent") is None
