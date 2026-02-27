"""Tests for the simulation engine and failure scenarios."""

import numpy as np
import pytest

from src.simulation.scenario_runner import ScenarioResult, ScenarioRunner
from src.simulation.scenarios.data_quality import DataQualityScenario
from src.simulation.scenarios.gradual_drift import GradualDriftScenario
from src.simulation.scenarios.latency_spike import LatencySpikeScenario
from src.simulation.scenarios.sudden_drift import SuddenDriftScenario


@pytest.fixture
def runner(tmp_path) -> ScenarioRunner:
    """Create a ScenarioRunner with the project's scenario configs."""
    return ScenarioRunner(scenarios_dir="configs/scenarios")


class TestScenarioRunner:
    """Tests for the ScenarioRunner orchestrator."""

    def test_load_scenario_config(self, runner):
        """Config should be loaded from YAML."""
        config = runner.load_scenario_config("gradual_drift")
        assert config["name"] == "gradual_drift"
        assert "parameters" in config

    def test_load_missing_config_raises(self, runner):
        """Missing config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            runner.load_scenario_config("nonexistent")

    def test_list_scenarios(self, runner):
        """Available scenarios should include config files."""
        scenarios = runner.list_scenarios()
        assert "gradual_drift" in scenarios
        assert "sudden_drift" in scenarios
        assert "data_quality" in scenarios
        assert "latency_spike" in scenarios

    def test_unregistered_scenario_raises(self, runner):
        """Running an unregistered scenario should raise ValueError."""
        with pytest.raises(ValueError, match="No implementation registered"):
            runner.run_scenario("gradual_drift")

    def test_run_registered_scenario(self, runner):
        """A registered scenario should execute and return a result."""
        runner.register_scenario("gradual_drift", GradualDriftScenario)
        result = runner.run_scenario("gradual_drift")

        assert isinstance(result, ScenarioResult)
        assert result.status == "completed"
        assert result.events_generated > 0

    def test_is_running_property(self, runner):
        """is_running should be False when no scenario is active."""
        assert not runner.is_running
        assert runner.current_scenario is None


class TestGradualDriftScenario:
    """Tests for the gradual drift scenario."""

    def test_generates_increasing_drift(self):
        """Each step should produce data with increasing mean shift."""
        config = {
            "name": "gradual_drift",
            "duration_seconds": 10,
            "parameters": {
                "features": ["f0", "f1"],
                "shift_magnitude": 0.5,
                "steps": 5,
                "batch_size": 200,
            },
        }
        scenario = GradualDriftScenario(config)
        batches = []
        result = scenario.run(callback=lambda df: batches.append(df))

        assert result.status == "completed"
        assert len(batches) == 5

        # Mean should increase with each batch
        means = [batch["f0"].mean() for batch in batches]
        for i in range(1, len(means)):
            assert means[i] > means[i - 1]

    def test_batch_size(self):
        """Each batch should have the configured number of rows."""
        config = {
            "name": "test",
            "duration_seconds": 5,
            "parameters": {
                "features": ["f0"],
                "shift_magnitude": 0.1,
                "steps": 3,
                "batch_size": 50,
            },
        }
        scenario = GradualDriftScenario(config)
        batches = []
        scenario.run(callback=lambda df: batches.append(df))

        for batch in batches:
            assert len(batch) == 50


class TestSuddenDriftScenario:
    """Tests for the sudden drift scenario."""

    def test_applies_shift_after_change_point(self):
        """Shift should be applied only after the change point."""
        config = {
            "name": "sudden_drift",
            "duration_seconds": 10,
            "parameters": {
                "features": ["f0"],
                "shift_magnitude": 5.0,
                "change_point_fraction": 0.5,
                "steps": 10,
                "batch_size": 500,
            },
        }
        scenario = SuddenDriftScenario(config)
        batches = []
        scenario.run(callback=lambda df: batches.append(df))

        # First half should have mean near 0
        pre_mean = np.mean([b["f0"].mean() for b in batches[:5]])
        # Second half should have mean near 5.0
        post_mean = np.mean([b["f0"].mean() for b in batches[5:]])

        assert abs(pre_mean) < 1.0
        assert post_mean > 3.0


class TestDataQualityScenario:
    """Tests for the data quality degradation scenario."""

    def test_increasing_missing_values(self):
        """Missing rate should increase over time."""
        config = {
            "name": "data_quality",
            "duration_seconds": 10,
            "parameters": {
                "features": ["f0", "f1"],
                "missing_rate_start": 0.0,
                "missing_rate_end": 0.5,
                "outlier_multiplier": 10.0,
                "steps": 5,
                "batch_size": 200,
            },
        }
        scenario = DataQualityScenario(config)
        batches = []
        scenario.run(callback=lambda df: batches.append(df))

        # First batch should have minimal missing
        first_missing = batches[0].isnull().sum().sum()
        # Last batch should have more missing
        last_missing = batches[-1].isnull().sum().sum()
        assert last_missing > first_missing


class TestLatencySpikeScenario:
    """Tests for the latency spike scenario."""

    def test_generates_latency_events(self):
        """Scenario should produce latency measurements with spikes."""
        config = {
            "name": "latency_spike",
            "duration_seconds": 10,
            "parameters": {
                "base_latency_ms": 10,
                "spike_latency_ms": 500,
                "spike_frequency": 0.3,
                "steps": 5,
                "batch_size": 20,
            },
        }
        scenario = LatencySpikeScenario(config)
        events = []
        scenario.run(callback=lambda e: events.append(e))

        assert len(events) == 100  # 5 steps * 20 batch_size
        spikes = [e for e in events if e["is_spike"]]
        non_spikes = [e for e in events if not e["is_spike"]]

        # Spikes should have higher latency
        if spikes and non_spikes:
            avg_spike = np.mean([e["latency_ms"] for e in spikes])
            avg_normal = np.mean([e["latency_ms"] for e in non_spikes])
            assert avg_spike > avg_normal

    def test_result_details(self):
        """Result should include spike statistics."""
        config = {
            "name": "latency_spike",
            "duration_seconds": 5,
            "parameters": {
                "base_latency_ms": 10,
                "spike_latency_ms": 500,
                "spike_frequency": 0.2,
                "steps": 3,
                "batch_size": 10,
            },
        }
        scenario = LatencySpikeScenario(config)
        result = scenario.run()

        assert result.status == "completed"
        assert "total_spikes" in result.details
        assert result.events_generated == 30
