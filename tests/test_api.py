"""Tests for FastAPI monitoring endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import app


@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset module-level globals before each test."""
    original = {
        "_config": app_module._config,
        "_db": app_module._db,
        "_drift_detector": app_module._drift_detector,
        "_perf_tracker": app_module._perf_tracker,
        "_degradation_detector": app_module._degradation_detector,
        "_report_generator": app_module._report_generator,
        "_scenario_runner": app_module._scenario_runner,
        "_quality_monitor": app_module._quality_monitor,
        "_reference_manager": app_module._reference_manager,
        "_drift_history": app_module._drift_history,
        "_ml_metrics": app_module._ml_metrics,
    }
    yield
    for attr, val in original.items():
        setattr(app_module, attr, val)


@pytest.fixture
def client():
    """Create a test client that skips the lifespan handler."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def initialized_client():
    """Create a test client with mocked initialized components."""
    mock_perf = MagicMock()
    mock_perf.compute_metrics.return_value = {
        "accuracy": 0.92,
        "precision_macro": 0.91,
        "recall_macro": 0.90,
        "f1_macro": 0.905,
        "sample_count": 100,
    }

    mock_degradation = MagicMock()
    mock_degradation.alerts = []

    mock_runner = MagicMock()
    mock_runner.is_running = False
    mock_runner.current_scenario = None

    mock_report = MagicMock()

    app_module._perf_tracker = mock_perf
    app_module._degradation_detector = mock_degradation
    app_module._scenario_runner = mock_runner
    app_module._report_generator = mock_report
    app_module._db = MagicMock()
    app_module._drift_detector = MagicMock()
    app_module._drift_detector.is_initialized = True

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_uninitialized(self, client):
        """Health check works even before components are initialized."""
        app_module._db = None
        app_module._drift_detector = None
        app_module._perf_tracker = None
        app_module._scenario_runner = None

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "uptime_seconds" in data
        assert data["components"]["database"] == "unavailable"

    def test_health_initialized(self, initialized_client):
        """Health check reports component status correctly."""
        resp = initialized_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["components"]["database"] == "ok"
        assert data["components"]["drift_detector"] == "ok"


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint returns Prometheus text format."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


class TestPredictEndpoint:
    """Tests for POST /monitor/predict."""

    def test_predict_uninitialized(self, client):
        """Predict returns 503 when service is not initialized."""
        app_module._perf_tracker = None
        resp = client.post(
            "/monitor/predict",
            json={"features": {"f0": 1.0, "f1": 2.0}},
        )
        assert resp.status_code == 503

    def test_predict_basic(self, initialized_client):
        """Predict returns prediction with confidence."""
        resp = initialized_client.post(
            "/monitor/predict",
            json={"features": {"f0": 1.0, "f1": 2.0}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "timestamp" in data
        assert data["drift_warning"] is False

    def test_predict_with_ground_truth(self, initialized_client):
        """Predict logs ground truth when provided."""
        resp = initialized_client.post(
            "/monitor/predict",
            json={"features": {"f0": 1.0}, "ground_truth": 1},
        )
        assert resp.status_code == 200
        app_module._perf_tracker.log_prediction.assert_called_once()

    def test_predict_drift_warning(self, initialized_client):
        """Predict reflects drift warning from history."""
        app_module._drift_history = [{"drift_detected": True}]
        resp = initialized_client.post(
            "/monitor/predict",
            json={"features": {"f0": 1.0}},
        )
        assert resp.status_code == 200
        assert resp.json()["drift_warning"] is True


class TestDriftEndpoint:
    """Tests for GET /monitor/drift."""

    def test_drift_no_history(self, client):
        """Drift status with no history returns defaults."""
        app_module._drift_history = []
        resp = client.get("/monitor/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["drift_detected"] is False
        assert data["overall_score"] == 0.0
        assert data["drifted_features"] == []

    def test_drift_with_history(self, client):
        """Drift status reflects latest history entry."""
        app_module._drift_history = [
            {"drift_detected": True, "scores": {"f0": 0.02, "f1": 0.8}, "drifted_features": ["f0"]},
        ]
        resp = client.get("/monitor/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["drift_detected"] is True
        assert len(data["drifted_features"]) == 1


class TestPerformanceEndpoint:
    """Tests for GET /monitor/performance."""

    def test_performance_uninitialized(self, client):
        """Performance returns 503 when not initialized."""
        app_module._perf_tracker = None
        resp = client.get("/monitor/performance")
        assert resp.status_code == 503

    def test_performance_metrics(self, initialized_client):
        """Performance returns computed metrics."""
        resp = initialized_client.get("/monitor/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["accuracy"] == 0.92
        assert data["sample_count"] == 100
        assert data["degradation_alerts"] == []

    def test_performance_with_alerts(self, initialized_client):
        """Performance includes degradation alerts."""
        alert = MagicMock()
        alert.message = "Accuracy dropped below threshold"
        app_module._degradation_detector.alerts = [alert]

        resp = initialized_client.get("/monitor/performance")
        assert resp.status_code == 200
        assert "Accuracy dropped" in resp.json()["degradation_alerts"][0]


class TestReportEndpoint:
    """Tests for GET /monitor/report."""

    def test_report_uninitialized(self, client):
        """Report returns 503 when not initialized."""
        app_module._report_generator = None
        app_module._perf_tracker = None
        resp = client.get("/monitor/report")
        assert resp.status_code == 503

    def test_report_generates_html(self, initialized_client, tmp_path):
        """Report endpoint returns HTML content."""
        report_path = tmp_path / "report.html"
        report_path.write_text("<!DOCTYPE html><html><body>ML Monitoring Report</body></html>")

        app_module._report_generator.generate_report.return_value = str(report_path)
        app_module._drift_history = []

        resp = initialized_client.get("/monitor/report")
        assert resp.status_code == 200
        assert "ML Monitoring Report" in resp.text


class TestSimulationEndpoint:
    """Tests for POST /simulate/{scenario}."""

    def test_invalid_scenario(self, initialized_client):
        """Unknown scenario returns 404."""
        resp = initialized_client.post("/simulate/invalid", json={})
        assert resp.status_code == 404

    def test_valid_scenario(self, initialized_client):
        """Valid scenario starts in background."""
        resp = initialized_client.post("/simulate/gradual_drift", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["scenario"] == "gradual_drift"

    def test_scenario_already_running(self, initialized_client):
        """Running scenario returns 409."""
        app_module._scenario_runner.is_running = True
        app_module._scenario_runner.current_scenario = "gradual_drift"

        resp = initialized_client.post("/simulate/sudden_drift", json={})
        assert resp.status_code == 409

    def test_simulation_uninitialized(self, client):
        """Simulation returns 503 when not initialized."""
        app_module._scenario_runner = None
        resp = client.post("/simulate/gradual_drift", json={})
        assert resp.status_code == 503

    def test_all_valid_scenarios(self, initialized_client):
        """All four scenario types are accepted."""
        for scenario in ["gradual_drift", "sudden_drift", "data_quality", "latency_spike"]:
            app_module._scenario_runner.is_running = False
            resp = initialized_client.post(f"/simulate/{scenario}", json={})
            assert resp.status_code == 200
