"""Tests for configuration loading and validation."""

import yaml

from src.utils.config import DriftConfig, Settings, load_config


class TestLoadConfig:
    """Tests for YAML configuration loading."""

    def test_load_from_valid_yaml(self, tmp_path):
        """Config should be populated from a valid YAML file."""
        config_data = {
            "drift": {"window_size": 500, "threshold": 0.1},
            "database": {"sqlite_path": "custom/path.db"},
            "api": {"port": 9000},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        settings = load_config(str(config_file))

        assert settings.drift.window_size == 500
        assert settings.drift.threshold == 0.1
        assert settings.database.sqlite_path == "custom/path.db"
        assert settings.api.port == 9000

    def test_default_values_when_missing(self, tmp_path):
        """Missing keys should fall back to defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{}")

        settings = load_config(str(config_file))

        assert settings.drift.window_size == 1000
        assert settings.drift.threshold == 0.05
        assert settings.database.sqlite_path == "data/metrics.db"

    def test_returns_defaults_when_file_missing(self, tmp_path):
        """A missing file should return default Settings."""
        settings = load_config(str(tmp_path / "nonexistent.yaml"))

        assert isinstance(settings, Settings)
        assert settings.drift.window_size == 1000

    def test_partial_override(self, tmp_path):
        """Partial config should override only specified values."""
        config_data = {"drift": {"threshold": 0.2}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        settings = load_config(str(config_file))

        assert settings.drift.threshold == 0.2
        assert settings.drift.window_size == 1000  # default preserved

    def test_performance_config_defaults(self):
        """Performance config defaults should be sensible."""
        settings = Settings()

        assert settings.performance.task_type == "classification"
        assert settings.performance.cusum_threshold == 5.0
        assert settings.performance.baseline_accuracy == 0.9


class TestDriftConfig:
    """Tests for DriftConfig validation."""

    def test_custom_feature_thresholds(self):
        """Feature-level thresholds should be configurable."""
        config = DriftConfig(features={"feature_0": 0.1, "feature_1": 0.02})

        assert config.features["feature_0"] == 0.1
        assert config.features["feature_1"] == 0.02

    def test_default_stattest(self):
        """Default statistical test should be KS."""
        config = DriftConfig()
        assert config.stattest == "ks"
