"""Latency spike scenario: simulated infrastructure latency issues."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from src.simulation.scenario_runner import ScenarioResult

logger = logging.getLogger(__name__)


class LatencySpikeScenario:
    """Simulate infrastructure latency spikes.

    Generates timing data with occasional spikes that exceed
    normal processing times.

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
        """Execute the latency spike scenario.

        Args:
            callback: Optional function called with each latency measurement
                dict containing 'latency_ms' and 'is_spike' keys.

        Returns:
            ScenarioResult with execution details.
        """
        steps = self.params["steps"]
        base_latency = self.params.get("base_latency_ms", 10)
        spike_latency = self.params.get("spike_latency_ms", 500)
        spike_freq = self.params.get("spike_frequency", 0.2)
        batch_size = self.params.get("batch_size", 50)
        events = 0
        total_spikes = 0

        rng = np.random.default_rng()

        for _step in range(steps):
            for _ in range(batch_size):
                is_spike = rng.random() < spike_freq
                latency = spike_latency if is_spike else base_latency
                latency += rng.standard_normal() * (base_latency * 0.1)
                latency = max(1.0, latency)

                if is_spike:
                    total_spikes += 1

                if callback:
                    callback({"latency_ms": latency, "is_spike": is_spike})
                events += 1

        return ScenarioResult(
            name=self.config["name"],
            status="completed",
            start_time="",
            events_generated=events,
            details={
                "total_spikes": total_spikes,
                "spike_frequency": spike_freq,
                "base_latency_ms": base_latency,
                "spike_latency_ms": spike_latency,
            },
        )
