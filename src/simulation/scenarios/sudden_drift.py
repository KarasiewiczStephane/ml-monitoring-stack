"""Sudden drift scenario: abrupt concept drift at a configured point."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.scenario_runner import ScenarioResult

logger = logging.getLogger(__name__)


class SuddenDriftScenario:
    """Simulate sudden concept drift with an abrupt distribution change.

    Data follows the reference distribution until a configured change
    point, after which a large shift is applied to feature means.

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
        """Execute the sudden drift scenario.

        Args:
            callback: Optional function called with each data batch.

        Returns:
            ScenarioResult with execution details.
        """
        steps = self.params["steps"]
        shift = self.params["shift_magnitude"]
        change_point = int(steps * self.params.get("change_point_fraction", 0.5))
        batch_size = self.params.get("batch_size", 100)
        features = self.params["features"]
        events = 0

        rng = np.random.default_rng()

        for step in range(steps):
            drift = shift if step >= change_point else 0.0
            data = rng.standard_normal((batch_size, len(features)))
            if drift > 0:
                data += drift
            df = pd.DataFrame(data, columns=features)
            if callback:
                callback(df)
            events += 1

        return ScenarioResult(
            name=self.config["name"],
            status="completed",
            start_time="",
            events_generated=events,
            details={"change_point_step": change_point, "shift_magnitude": shift},
        )
