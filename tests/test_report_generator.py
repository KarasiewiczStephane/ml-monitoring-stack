"""Tests for the automated HTML report generator."""

import pytest

from src.reporting.report_generator import ReportGenerator


@pytest.fixture
def generator() -> ReportGenerator:
    """Create a ReportGenerator with default template directory."""
    return ReportGenerator()


@pytest.fixture
def sample_drift_history() -> list[dict]:
    """Create a sample drift history for testing."""
    return [
        {"drift_detected": False, "scores": {"f0": 0.8, "f1": 0.9}},
        {"drift_detected": False, "scores": {"f0": 0.7, "f1": 0.85}},
        {"drift_detected": True, "scores": {"f0": 0.02, "f1": 0.8}, "drifted_features": ["f0"]},
    ]


@pytest.fixture
def sample_perf_metrics() -> dict:
    """Create sample performance metrics."""
    return {
        "accuracy": 0.92,
        "precision_macro": 0.91,
        "recall_macro": 0.90,
        "f1_macro": 0.905,
        "sample_count": 1000,
    }


class TestAnalyzeDriftTrend:
    """Tests for drift trend analysis."""

    def test_increasing_trend(self, generator):
        """Increasing drift scores should be reported."""
        history = [
            {"overall_score": 0.1},
            {"overall_score": 0.3},
            {"overall_score": 0.5},
            {"overall_score": 0.7},
        ]
        result = generator.analyze_drift_trend(history)
        assert "Increasing" in result

    def test_decreasing_trend(self, generator):
        """Decreasing drift scores should be reported."""
        history = [
            {"overall_score": 0.8},
            {"overall_score": 0.5},
            {"overall_score": 0.3},
            {"overall_score": 0.1},
        ]
        result = generator.analyze_drift_trend(history)
        assert "Decreasing" in result

    def test_stable_trend(self, generator):
        """Flat drift scores should be reported as stable."""
        history = [
            {"overall_score": 0.5},
            {"overall_score": 0.5},
            {"overall_score": 0.5},
            {"overall_score": 0.5},
        ]
        result = generator.analyze_drift_trend(history)
        assert "Stable" in result

    def test_insufficient_data(self, generator):
        """Too few data points should return insufficient message."""
        history = [{"overall_score": 0.5}]
        result = generator.analyze_drift_trend(history)
        assert "Insufficient" in result

    def test_uses_scores_fallback(self, generator):
        """Should fall back to averaging feature scores when no overall_score."""
        history = [
            {"scores": {"f0": 0.1, "f1": 0.2}},
            {"scores": {"f0": 0.3, "f1": 0.4}},
            {"scores": {"f0": 0.5, "f1": 0.6}},
        ]
        result = generator.analyze_drift_trend(history)
        assert isinstance(result, str)


class TestGenerateRecommendations:
    """Tests for recommendation generation."""

    def test_no_issues(self, generator):
        """Clean status should produce positive recommendation."""
        recs = generator.generate_recommendations(
            {"drift_detected": False},
            {"accuracy": 0.95, "f1_macro": 0.93},
        )
        assert any("within expected" in r for r in recs)

    def test_drift_recommendations(self, generator):
        """Drift detection should trigger retrain recommendation."""
        recs = generator.generate_recommendations(
            {"drift_detected": True, "drifted_features": ["f0", "f1"]},
            {"accuracy": 0.95},
        )
        assert any("retrain" in r.lower() for r in recs)
        assert any("f0" in r for r in recs)

    def test_low_accuracy_recommendation(self, generator):
        """Low accuracy should trigger accuracy warning."""
        recs = generator.generate_recommendations(
            {"drift_detected": False},
            {"accuracy": 0.7},
        )
        assert any("accuracy" in r.lower() for r in recs)

    def test_low_f1_recommendation(self, generator):
        """Low F1 should trigger F1 warning."""
        recs = generator.generate_recommendations(
            {"drift_detected": False},
            {"accuracy": 0.95, "f1_macro": 0.5},
        )
        assert any("f1" in r.lower() for r in recs)


class TestGenerateReport:
    """Tests for HTML report generation."""

    def test_creates_html_file(
        self, generator, sample_drift_history, sample_perf_metrics, tmp_path
    ):
        """Report should be written as an HTML file."""
        output = str(tmp_path / "test_report.html")
        result = generator.generate_report(
            sample_drift_history,
            sample_perf_metrics,
            output_path=output,
        )

        assert result == output
        content = open(output).read()
        assert "<!DOCTYPE html>" in content
        assert "ML Monitoring Report" in content

    def test_includes_metrics(self, generator, sample_drift_history, sample_perf_metrics, tmp_path):
        """Report should include performance metrics values."""
        output = str(tmp_path / "report.html")
        generator.generate_report(sample_drift_history, sample_perf_metrics, output)

        content = open(output).read()
        assert "accuracy" in content
        assert "0.92" in content

    def test_includes_recommendations(self, generator, tmp_path):
        """Report should include recommendations."""
        drift_history = [{"drift_detected": True, "drifted_features": ["f0"]}]
        perf = {"accuracy": 0.7}
        output = str(tmp_path / "report.html")

        generator.generate_report(drift_history, perf, output)
        content = open(output).read()
        assert "Recommendations" in content
        assert "retrain" in content.lower()

    def test_creates_parent_directory(
        self, generator, sample_drift_history, sample_perf_metrics, tmp_path
    ):
        """Report should create parent directories as needed."""
        output = str(tmp_path / "sub" / "dir" / "report.html")
        result = generator.generate_report(
            sample_drift_history,
            sample_perf_metrics,
            output_path=output,
        )
        assert result == output

    def test_empty_history(self, generator, tmp_path):
        """Report should handle empty drift history gracefully."""
        output = str(tmp_path / "report.html")
        generator.generate_report([], {"accuracy": 0.9}, output)
        content = open(output).read()
        assert "<!DOCTYPE html>" in content
