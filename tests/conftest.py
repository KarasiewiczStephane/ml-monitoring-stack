"""Shared test fixtures for the ML monitoring stack."""

import numpy as np
import pandas as pd
import pytest

from src.utils.database import MetricsDB


@pytest.fixture
def sample_reference_data() -> pd.DataFrame:
    """Generate a reproducible reference dataset with 10 numeric features."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({f"feature_{i}": rng.standard_normal(1000) for i in range(10)})


@pytest.fixture
def sample_drifted_data(sample_reference_data: pd.DataFrame) -> pd.DataFrame:
    """Create a copy of reference data with drift injected into feature_0."""
    drifted = sample_reference_data.copy()
    drifted["feature_0"] += 2.0
    return drifted


@pytest.fixture
def mock_db(tmp_path) -> MetricsDB:
    """Provide a temporary SQLite database for testing."""
    return MetricsDB(str(tmp_path / "test_metrics.db"))
