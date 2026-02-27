"""Tests for reference dataset management."""

import json

import numpy as np
import pandas as pd
import pytest

from src.drift.reference_manager import ReferenceManager


@pytest.fixture
def ref_manager(tmp_path) -> ReferenceManager:
    """Provide a ReferenceManager with a temporary directory."""
    return ReferenceManager(reference_dir=str(tmp_path / "reference"))


class TestLoadSklearn:
    """Tests for loading sklearn datasets."""

    def test_load_wine(self, ref_manager):
        """Wine dataset should load with expected columns."""
        df = ref_manager.load_sklearn_dataset("wine")
        assert "target" in df.columns
        assert len(df) > 100

    def test_load_breast_cancer(self, ref_manager):
        """Breast cancer dataset should load with expected columns."""
        df = ref_manager.load_sklearn_dataset("breast_cancer")
        assert "target" in df.columns
        assert len(df) > 100

    def test_unknown_dataset_raises(self, ref_manager):
        """An unrecognized name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            ref_manager.load_sklearn_dataset("nonexistent")


class TestComputeStatistics:
    """Tests for statistics computation."""

    def test_statistics_keys(self, ref_manager):
        """Each column should have mean, std, min, max, quantiles."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        stats = ref_manager.compute_statistics(df)

        assert "a" in stats
        assert "b" in stats
        for key in ("mean", "std", "min", "max", "quantiles"):
            assert key in stats["a"]

    def test_statistics_accuracy(self, ref_manager):
        """Computed statistics should match pandas describe values."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        df = pd.DataFrame({"x": data})
        stats = ref_manager.compute_statistics(df)

        assert abs(stats["x"]["mean"] - df["x"].mean()) < 1e-10
        assert abs(stats["x"]["std"] - df["x"].std()) < 1e-10

    def test_empty_numeric_skipped(self, ref_manager):
        """Columns with all NaN should be skipped."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
        stats = ref_manager.compute_statistics(df)
        assert "a" in stats
        assert "b" not in stats


class TestSaveLoadRoundtrip:
    """Tests for reference persistence."""

    def test_save_and_load_preserves_data(self, ref_manager):
        """A save/load cycle should preserve the DataFrame contents."""
        original = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        ref_manager.save_reference(original, "test_ref")

        loaded_df, loaded_stats = ref_manager.load_reference("test_ref")
        pd.testing.assert_frame_equal(loaded_df, original)
        assert "x" in loaded_stats
        assert "y" in loaded_stats

    def test_has_reference(self, ref_manager):
        """has_reference should correctly detect presence."""
        assert not ref_manager.has_reference("test_ref")

        df = pd.DataFrame({"x": [1.0, 2.0]})
        ref_manager.save_reference(df, "test_ref")
        assert ref_manager.has_reference("test_ref")

    def test_load_missing_raises(self, ref_manager):
        """Loading a nonexistent reference should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ref_manager.load_reference("nonexistent")


class TestFeatureThresholds:
    """Tests for per-feature threshold management."""

    def test_default_thresholds(self, ref_manager):
        """All features should get the default threshold."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "target": [0]})
        ref_manager.save_reference(df, "test_ref")

        thresholds = ref_manager.get_feature_thresholds("test_ref", default=0.1)
        assert thresholds["a"] == 0.1
        assert thresholds["b"] == 0.1
        assert "target" not in thresholds

    def test_custom_thresholds_file(self, ref_manager):
        """Custom thresholds from a JSON file should take precedence."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "target": [0]})
        ref_manager.save_reference(df, "test_ref")

        custom = {"a": 0.01, "b": 0.2}
        threshold_path = ref_manager.reference_dir / "test_ref_thresholds.json"
        with open(threshold_path, "w") as f:
            json.dump(custom, f)

        thresholds = ref_manager.get_feature_thresholds("test_ref")
        assert thresholds == custom


class TestInitializeDefault:
    """Tests for default reference initialization."""

    def test_initialize_creates_reference(self, ref_manager):
        """initialize_default_reference should create reference files."""
        ref_manager.initialize_default_reference("wine")
        assert ref_manager.has_reference("wine")

    def test_initialize_is_idempotent(self, ref_manager):
        """Calling initialize twice should not overwrite existing reference."""
        ref_manager.initialize_default_reference("wine")
        _, stats1 = ref_manager.load_reference("wine")

        ref_manager.initialize_default_reference("wine")
        _, stats2 = ref_manager.load_reference("wine")

        assert stats1 == stats2
