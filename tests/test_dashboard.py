"""Tests for the ML monitoring stack dashboard data generators."""

from src.dashboard.app import (
    generate_alert_history,
    generate_drift_data,
    generate_performance_over_time,
    generate_system_health,
)


class TestDriftData:
    """Tests for the drift detection data generator."""

    def test_returns_dict(self) -> None:
        data = generate_drift_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self) -> None:
        data = generate_drift_data()
        required = {
            "dataset_drift_detected",
            "drift_scores",
            "threshold",
            "drifted_features",
            "reference_distributions",
            "current_distributions",
        }
        assert required.issubset(data.keys())

    def test_drift_scores_are_valid(self) -> None:
        data = generate_drift_data()
        for score in data["drift_scores"].values():
            assert 0.0 <= score <= 1.0

    def test_distributions_match_features(self) -> None:
        data = generate_drift_data()
        features = set(data["drift_scores"].keys())
        assert set(data["reference_distributions"].keys()) == features
        assert set(data["current_distributions"].keys()) == features

    def test_distributions_have_samples(self) -> None:
        data = generate_drift_data()
        for feat in data["drift_scores"]:
            assert len(data["reference_distributions"][feat]) > 0
            assert len(data["current_distributions"][feat]) > 0

    def test_drifted_features_consistency(self) -> None:
        data = generate_drift_data()
        threshold = data["threshold"]
        expected = [f for f, s in data["drift_scores"].items() if s < threshold]
        assert data["drifted_features"] == expected


class TestPerformanceOverTime:
    """Tests for the performance over time generator."""

    def test_returns_list(self) -> None:
        data = generate_performance_over_time()
        assert isinstance(data, list)

    def test_has_thirty_entries(self) -> None:
        data = generate_performance_over_time()
        assert len(data) == 30

    def test_entry_keys(self) -> None:
        data = generate_performance_over_time()
        required = {"timestamp", "accuracy", "precision", "recall", "f1", "sample_count"}
        for entry in data:
            assert required.issubset(entry.keys())

    def test_scores_bounded(self) -> None:
        data = generate_performance_over_time()
        for entry in data:
            for key in ("accuracy", "precision", "recall", "f1"):
                assert 0.0 <= entry[key] <= 1.0


class TestSystemHealth:
    """Tests for the system health generator."""

    def test_returns_dict(self) -> None:
        data = generate_system_health()
        assert isinstance(data, dict)

    def test_has_gauge_metrics(self) -> None:
        data = generate_system_health()
        for key in ("cpu_percent", "memory_percent", "disk_percent", "gpu_percent"):
            assert key in data
            assert 0 <= data[key] <= 100

    def test_has_operational_metrics(self) -> None:
        data = generate_system_health()
        assert "total_predictions_24h" in data
        assert "error_rate_24h" in data
        assert "uptime_hours" in data


class TestAlertHistory:
    """Tests for the alert history generator."""

    def test_returns_list(self) -> None:
        data = generate_alert_history()
        assert isinstance(data, list)

    def test_has_entries(self) -> None:
        data = generate_alert_history()
        assert len(data) > 0

    def test_entry_keys(self) -> None:
        data = generate_alert_history()
        required = {"timestamp", "severity", "source", "message", "resolved"}
        for entry in data:
            assert required.issubset(entry.keys())

    def test_valid_severities(self) -> None:
        data = generate_alert_history()
        valid = {"critical", "warning", "info"}
        for entry in data:
            assert entry["severity"] in valid

    def test_has_unresolved_alert(self) -> None:
        data = generate_alert_history()
        assert any(not a["resolved"] for a in data)
