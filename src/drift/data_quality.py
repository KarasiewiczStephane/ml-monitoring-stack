"""Data quality monitoring for missing values, out-of-range, and schema issues."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor data quality against a reference statistical profile.

    Detects missing values, out-of-range values, and new categories
    that were not present in the reference dataset.

    Args:
        reference_stats: Statistical profile from ReferenceManager.compute_statistics().
    """

    def __init__(self, reference_stats: dict) -> None:
        self.reference_stats = reference_stats

    def check_missing_values(self, df: pd.DataFrame) -> dict:
        """Detect missing value rates per column.

        Args:
            df: Current data to check.

        Returns:
            Dictionary with missing rates, total rate, and affected columns.
        """
        missing = df.isnull().sum() / max(len(df), 1)
        total_cells = max(df.size, 1)
        return {
            "missing_rates": {k: float(v) for k, v in missing.items()},
            "total_missing_rate": float(df.isnull().sum().sum() / total_cells),
            "columns_with_missing": list(missing[missing > 0].index),
        }

    def check_out_of_range(self, df: pd.DataFrame) -> dict:
        """Detect values outside the reference min/max range.

        Args:
            df: Current data to check.

        Returns:
            Dictionary mapping column names to violation counts.
        """
        out_of_range: dict = {}
        for col, stats in self.reference_stats.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if len(series) == 0:
                continue
            below = int((series < stats["min"]).sum())
            above = int((series > stats["max"]).sum())
            if below > 0 or above > 0:
                out_of_range[col] = {
                    "below_min": below,
                    "above_max": above,
                    "total_violations": below + above,
                }
        return out_of_range

    def check_new_categories(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> dict:
        """Detect categorical values not present in the reference data.

        Args:
            df: Current data to check.
            reference_df: Original reference dataset.

        Returns:
            Dictionary mapping column names to lists of new categories.
        """
        new_categories: dict = {}
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if col not in reference_df.columns:
                continue
            ref_cats = set(reference_df[col].dropna().unique())
            curr_cats = set(df[col].dropna().unique())
            new = curr_cats - ref_cats
            if new:
                new_categories[col] = sorted(str(c) for c in new)
        return new_categories

    def check_schema(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> dict:
        """Validate the schema against the reference dataset.

        Args:
            df: Current data to check.
            reference_df: Original reference dataset.

        Returns:
            Dictionary with missing and extra columns.
        """
        ref_cols = set(reference_df.columns)
        cur_cols = set(df.columns)
        return {
            "missing_columns": sorted(ref_cols - cur_cols),
            "extra_columns": sorted(cur_cols - ref_cols),
            "schema_valid": ref_cols == cur_cols,
        }

    def compute_quality_score(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> float:
        """Compute an overall data quality score between 0 and 1.

        The score is penalized for missing values, out-of-range values,
        and new categories.

        Args:
            df: Current data to check.
            reference_df: Original reference dataset.

        Returns:
            Quality score where 1.0 is perfect quality.
        """
        missing = self.check_missing_values(df)
        out_of_range = self.check_out_of_range(df)
        new_cats = self.check_new_categories(df, reference_df)

        score = 1.0
        score -= missing["total_missing_rate"] * 0.5

        num_cols = max(len(df.columns), 1)
        score -= len(out_of_range) / num_cols * 0.3

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        num_cat_cols = max(len(cat_cols), 1)
        score -= len(new_cats) / num_cat_cols * 0.2

        return max(0.0, min(1.0, score))

    def run_full_check(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> dict:
        """Run all data quality checks and return a combined report.

        Args:
            df: Current data to check.
            reference_df: Original reference dataset.

        Returns:
            Dictionary with results from all quality checks.
        """
        result = {
            "missing_values": self.check_missing_values(df),
            "out_of_range": self.check_out_of_range(df),
            "new_categories": self.check_new_categories(df, reference_df),
            "schema": self.check_schema(df, reference_df),
            "quality_score": self.compute_quality_score(df, reference_df),
        }
        logger.info("Data quality check complete: score=%.3f", result["quality_score"])
        return result
