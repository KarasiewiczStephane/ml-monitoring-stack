"""Tests for data quality monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.drift.data_quality import DataQualityMonitor
from src.drift.reference_manager import ReferenceManager


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """Create a reference dataset with known distributions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "feature_0": rng.standard_normal(200),
            "feature_1": rng.uniform(0, 10, 200),
            "feature_2": rng.standard_normal(200),
        }
    )


@pytest.fixture
def reference_stats(reference_df) -> dict:
    """Compute reference statistics."""
    mgr = ReferenceManager(reference_dir="/tmp/test_ref_unused")
    return mgr.compute_statistics(reference_df)


@pytest.fixture
def monitor(reference_stats) -> DataQualityMonitor:
    """Create a DataQualityMonitor with reference stats."""
    return DataQualityMonitor(reference_stats=reference_stats)


class TestCheckMissingValues:
    """Tests for missing value detection."""

    def test_no_missing_values(self, monitor, reference_df):
        """Clean data should report zero missing values."""
        result = monitor.check_missing_values(reference_df)
        assert result["total_missing_rate"] == 0.0
        assert result["columns_with_missing"] == []

    def test_detects_injected_nans(self, monitor, reference_df):
        """Injected NaN values should be detected."""
        df = reference_df.copy()
        df.loc[:9, "feature_0"] = np.nan  # 10 missing values

        result = monitor.check_missing_values(df)
        assert result["missing_rates"]["feature_0"] == pytest.approx(10 / 200)
        assert "feature_0" in result["columns_with_missing"]

    def test_all_missing(self, monitor):
        """A column of all NaN should report 100% missing."""
        df = pd.DataFrame({"feature_0": [np.nan] * 10})
        result = monitor.check_missing_values(df)
        assert result["missing_rates"]["feature_0"] == 1.0


class TestCheckOutOfRange:
    """Tests for out-of-range value detection."""

    def test_in_range_data(self, monitor, reference_df):
        """Data within reference bounds should report no violations."""
        result = monitor.check_out_of_range(reference_df)
        assert len(result) == 0

    def test_detects_below_min(self, monitor, reference_df, reference_stats):
        """Values below reference minimum should be flagged."""
        df = reference_df.copy()
        min_val = reference_stats["feature_1"]["min"]
        df.loc[0, "feature_1"] = min_val - 10

        result = monitor.check_out_of_range(df)
        assert "feature_1" in result
        assert result["feature_1"]["below_min"] >= 1

    def test_detects_above_max(self, monitor, reference_df, reference_stats):
        """Values above reference maximum should be flagged."""
        df = reference_df.copy()
        max_val = reference_stats["feature_1"]["max"]
        df.loc[0, "feature_1"] = max_val + 10

        result = monitor.check_out_of_range(df)
        assert "feature_1" in result
        assert result["feature_1"]["above_max"] >= 1


class TestCheckNewCategories:
    """Tests for new category detection."""

    def test_no_new_categories(self, monitor):
        """Identical categories should report nothing."""
        ref = pd.DataFrame({"cat": ["a", "b", "c"]})
        cur = pd.DataFrame({"cat": ["a", "b"]})
        result = monitor.check_new_categories(cur, ref)
        assert len(result) == 0

    def test_detects_new_categories(self, monitor):
        """New categorical values should be detected."""
        ref = pd.DataFrame({"cat": ["a", "b"]})
        cur = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
        result = monitor.check_new_categories(cur, ref)
        assert "cat" in result
        assert set(result["cat"]) == {"c", "d"}


class TestCheckSchema:
    """Tests for schema validation."""

    def test_matching_schema(self, monitor, reference_df):
        """Identical schemas should be valid."""
        result = monitor.check_schema(reference_df, reference_df)
        assert result["schema_valid"]
        assert result["missing_columns"] == []
        assert result["extra_columns"] == []

    def test_missing_column(self, monitor, reference_df):
        """Missing columns should be detected."""
        df = reference_df.drop(columns=["feature_0"])
        result = monitor.check_schema(df, reference_df)
        assert not result["schema_valid"]
        assert "feature_0" in result["missing_columns"]

    def test_extra_column(self, monitor, reference_df):
        """Extra columns should be detected."""
        df = reference_df.copy()
        df["new_col"] = 1.0
        result = monitor.check_schema(df, reference_df)
        assert not result["schema_valid"]
        assert "new_col" in result["extra_columns"]


class TestComputeQualityScore:
    """Tests for overall quality score computation."""

    def test_perfect_quality(self, monitor, reference_df):
        """Clean data should score 1.0."""
        score = monitor.compute_quality_score(reference_df, reference_df)
        assert score == pytest.approx(1.0)

    def test_low_quality_with_missing(self, monitor, reference_df):
        """Heavy missing values should lower the score."""
        df = reference_df.copy()
        df.loc[:, "feature_0"] = np.nan  # All values missing
        score = monitor.compute_quality_score(df, reference_df)
        assert score < 1.0

    def test_score_clamped_to_zero(self, monitor):
        """Score should never go below 0."""
        df = pd.DataFrame({"a": [np.nan] * 100, "b": [np.nan] * 100})
        ref = pd.DataFrame({"a": [1.0] * 100, "b": [2.0] * 100})
        score = monitor.compute_quality_score(df, ref)
        assert score >= 0.0


class TestRunFullCheck:
    """Tests for the combined quality report."""

    def test_full_check_keys(self, monitor, reference_df):
        """Full check should include all sub-check results."""
        result = monitor.run_full_check(reference_df, reference_df)
        assert "missing_values" in result
        assert "out_of_range" in result
        assert "new_categories" in result
        assert "schema" in result
        assert "quality_score" in result

    def test_full_check_with_issues(self, monitor, reference_df, reference_stats):
        """Full check should detect multiple quality issues."""
        df = reference_df.copy()
        df.loc[:4, "feature_0"] = np.nan
        df.loc[0, "feature_1"] = reference_stats["feature_1"]["max"] + 100

        result = monitor.run_full_check(df, reference_df)
        assert result["quality_score"] < 1.0
        assert len(result["missing_values"]["columns_with_missing"]) > 0
        assert len(result["out_of_range"]) > 0
