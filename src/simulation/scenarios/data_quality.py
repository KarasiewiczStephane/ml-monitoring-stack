"""Data quality degradation scenario: increasing missing values and outliers."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.scenario_runner import ScenarioResult

logger = logging.getLogger(__name__)


class DataQualityScenario:
    """Simulate data quality degradation over time.

    Progressively injects missing values and outliers into the
    generated data batches.

    Args:
        config: Scenario configuration dict loaded from YAML.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.params = config["parameters"]

    def run(
        self,
        callback: Callable[[Any], None] | None = None,
    ) -> ScenarioResult:
        """Execute the data quality degradation scenario.

        Args:
            callback: Optional function called with each degraded DataFrame.

        Returns:
            ScenarioResult with execution details.
        """
        steps = self.params["steps"]
        batch_size = self.params.get("batch_size", 100)
        features = self.params["features"]
        missing_start = self.params.get("missing_rate_start", 0.0)
        missing_end = self.params.get("missing_rate_end", 0.5)
        outlier_mult = self.params.get("outlier_multiplier", 10.0)
        events = 0

        rng = np.random.default_rng()

        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            missing_rate = missing_start + (missing_end - missing_start) * fraction
            data = rng.standard_normal((batch_size, len(features)))
            df = pd.DataFrame(data, columns=features)

            # Inject missing values
            mask = rng.random(df.shape) < missing_rate
            df[mask] = np.nan

            # Inject outliers in remaining non-NaN values
            outlier_mask = rng.random(df.shape) < (fraction * 0.1)
            df[outlier_mask & ~mask] *= outlier_mult

            if callback:
                callback(df)
            events += 1

        return ScenarioResult(
            name=self.config["name"],
            status="completed",
            start_time="",
            events_generated=events,
            details={
                "final_missing_rate": missing_end,
                "outlier_multiplier": outlier_mult,
            },
        )
