"""Gradual drift scenario: slowly shift feature distributions over time."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.scenario_runner import ScenarioResult

logger = logging.getLogger(__name__)


class GradualDriftScenario:
    """Simulate gradual data drift by incrementally shifting feature means.

    Each step increases the mean shift for the configured features,
    producing data that gradually deviates from the reference distribution.

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
        """Execute the gradual drift scenario.

        Args:
            callback: Optional function called with each drifted DataFrame.

        Returns:
            ScenarioResult with execution details.
        """
        steps = self.params["steps"]
        shift = self.params["shift_magnitude"]
        batch_size = self.params.get("batch_size", 100)
        features = self.params["features"]
        events = 0

        rng = np.random.default_rng()

        for step in range(steps):
            drift_amount = shift * (step + 1)
            data = self._generate_drifted_batch(rng, features, drift_amount, batch_size)
            if callback:
                callback(data)
            events += 1

        return ScenarioResult(
            name=self.config["name"],
            status="completed",
            start_time="",
            events_generated=events,
            details={"final_drift_amount": shift * steps, "steps": steps},
        )

    def _generate_drifted_batch(
        self,
        rng: np.random.Generator,
        features: list[str],
        drift_amount: float,
        batch_size: int,
    ) -> pd.DataFrame:
        """Generate a batch of data with the specified mean shift.

        Args:
            rng: NumPy random generator.
            features: List of feature column names.
            drift_amount: Amount to shift the mean.
            batch_size: Number of samples per batch.

        Returns:
            DataFrame with drifted feature values.
        """
        data = rng.standard_normal((batch_size, len(features)))
        data += drift_amount
        return pd.DataFrame(data, columns=features)
