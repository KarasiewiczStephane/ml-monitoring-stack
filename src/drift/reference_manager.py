"""Reference dataset management for baseline data statistics."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine

logger = logging.getLogger(__name__)

_SKLEARN_LOADERS = {
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


class ReferenceManager:
    """Manage reference datasets and their statistical profiles.

    Stores baseline data and precomputed statistics for comparison
    during drift detection.

    Args:
        reference_dir: Directory path for persisting reference data.
    """

    def __init__(self, reference_dir: str = "data/reference") -> None:
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def load_sklearn_dataset(self, name: str = "wine") -> pd.DataFrame:
        """Load an sklearn toy dataset as a reference DataFrame.

        Args:
            name: Dataset name, one of 'wine' or 'breast_cancer'.

        Returns:
            DataFrame with features and a 'target' column.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        if name not in _SKLEARN_LOADERS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_SKLEARN_LOADERS)}")

        loader = _SKLEARN_LOADERS[name]
        data = loader(as_frame=True)
        df = data.frame.copy()
        df["target"] = data.target
        logger.info("Loaded sklearn dataset '%s' with shape %s", name, df.shape)
        return df

    def compute_statistics(self, df: pd.DataFrame) -> dict:
        """Compute a statistical profile for each numeric column.

        Args:
            df: Input DataFrame to profile.

        Returns:
            Dictionary mapping column names to their statistics
            (mean, std, min, max, quantiles).
        """
        stats: dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            quantiles = series.quantile([0.25, 0.5, 0.75])
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "quantiles": {str(k): float(v) for k, v in quantiles.items()},
            }
        return stats

    def save_reference(self, df: pd.DataFrame, name: str) -> None:
        """Persist a reference dataset and its statistics to disk.

        Args:
            df: Reference DataFrame to save.
            name: Identifier used for filenames.
        """
        data_path = self.reference_dir / f"{name}_data.parquet"
        stats_path = self.reference_dir / f"{name}_stats.json"

        df.to_parquet(data_path, index=False)
        stats = self.compute_statistics(df)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Saved reference '%s': data=%s, stats=%s", name, data_path, stats_path)

    def load_reference(self, name: str) -> tuple[pd.DataFrame, dict]:
        """Load a previously saved reference dataset and its statistics.

        Args:
            name: Identifier matching a saved reference.

        Returns:
            Tuple of (DataFrame, statistics dict).

        Raises:
            FileNotFoundError: If the reference data or stats file is missing.
        """
        data_path = self.reference_dir / f"{name}_data.parquet"
        stats_path = self.reference_dir / f"{name}_stats.json"

        if not data_path.exists():
            raise FileNotFoundError(f"Reference data not found: {data_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"Reference stats not found: {stats_path}")

        df = pd.read_parquet(data_path)
        with open(stats_path) as f:
            stats = json.load(f)

        logger.info("Loaded reference '%s' with shape %s", name, df.shape)
        return df, stats

    def get_feature_thresholds(
        self,
        name: str,
        default: float = 0.05,
    ) -> dict[str, float]:
        """Get per-feature drift detection thresholds.

        Loads custom thresholds from a JSON file if available,
        otherwise returns the default threshold for every feature.

        Args:
            name: Reference dataset identifier.
            default: Default threshold value for each feature.

        Returns:
            Dictionary mapping feature names to threshold values.
        """
        threshold_path = self.reference_dir / f"{name}_thresholds.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                return json.load(f)

        df, _ = self.load_reference(name)
        return {col: default for col in df.columns if col != "target"}

    def has_reference(self, name: str) -> bool:
        """Check whether a reference dataset exists on disk.

        Args:
            name: Reference dataset identifier.

        Returns:
            True if both data and stats files exist.
        """
        data_path = self.reference_dir / f"{name}_data.parquet"
        stats_path = self.reference_dir / f"{name}_stats.json"
        return data_path.exists() and stats_path.exists()

    def initialize_default_reference(self, name: str = "wine") -> None:
        """Load an sklearn dataset and save it as the default reference.

        Convenience method for first-time setup.

        Args:
            name: sklearn dataset name to use as baseline.
        """
        if self.has_reference(name):
            logger.info("Reference '%s' already exists, skipping initialization", name)
            return
        df = self.load_sklearn_dataset(name)
        self.save_reference(df, name)
        logger.info("Initialized default reference dataset '%s'", name)
