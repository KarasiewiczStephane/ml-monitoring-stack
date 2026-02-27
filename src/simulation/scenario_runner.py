"""Scenario runner for executing configurable failure simulations."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result from a completed simulation scenario.

    Attributes:
        name: Scenario name.
        status: Final status (running, completed, failed).
        start_time: ISO start timestamp.
        end_time: ISO end timestamp or None if still running.
        events_generated: Number of data batches produced.
        details: Additional result details.
    """

    name: str
    status: str
    start_time: str
    end_time: str | None = None
    events_generated: int = 0
    details: dict = field(default_factory=dict)


class ScenarioRunner:
    """Load and execute failure simulation scenarios from YAML configs.

    Maintains a registry of scenario implementations and coordinates
    their execution.

    Args:
        scenarios_dir: Directory containing scenario YAML configs.
    """

    def __init__(self, scenarios_dir: str = "configs/scenarios") -> None:
        self.scenarios_dir = Path(scenarios_dir)
        self._running_scenario: str | None = None
        self._scenario_classes: dict[str, type] = {}

    def register_scenario(self, name: str, scenario_class: type) -> None:
        """Register a scenario implementation.

        Args:
            name: Scenario name matching the YAML config filename.
            scenario_class: Class implementing the scenario logic.
        """
        self._scenario_classes[name] = scenario_class
        logger.info("Registered scenario: %s", name)

    def load_scenario_config(self, name: str) -> dict:
        """Load a scenario configuration from YAML.

        Args:
            name: Scenario name (without .yaml extension).

        Returns:
            Parsed YAML configuration dictionary.

        Raises:
            FileNotFoundError: If the scenario config file is missing.
        """
        config_path = self.scenarios_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Scenario config not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    def list_scenarios(self) -> list[str]:
        """List all available scenario names.

        Returns:
            List of scenario names from registered classes and config files.
        """
        registered = set(self._scenario_classes.keys())
        on_disk = {p.stem for p in self.scenarios_dir.glob("*.yaml")}
        return sorted(registered | on_disk)

    def run_scenario(
        self,
        name: str,
        data_callback: Callable[[Any], None] | None = None,
    ) -> ScenarioResult:
        """Execute a scenario synchronously.

        Args:
            name: Scenario name to run.
            data_callback: Optional callback invoked with each generated
                data batch.

        Returns:
            ScenarioResult with execution details.

        Raises:
            RuntimeError: If another scenario is already running.
            ValueError: If the scenario name is not registered.
        """
        if self._running_scenario:
            raise RuntimeError(f"Scenario '{self._running_scenario}' already running")

        config = self.load_scenario_config(name)
        scenario_class = self._scenario_classes.get(name)
        if not scenario_class:
            raise ValueError(f"No implementation registered for scenario: {name}")

        self._running_scenario = name
        start_time = datetime.now(tz=UTC).isoformat()

        try:
            scenario = scenario_class(config)
            result = scenario.run(data_callback)
            result.start_time = start_time
            result.end_time = datetime.now(tz=UTC).isoformat()
            logger.info("Scenario '%s' completed: %d events", name, result.events_generated)
            return result
        except Exception as exc:
            logger.exception("Scenario '%s' failed", name)
            return ScenarioResult(
                name=name,
                status="failed",
                start_time=start_time,
                end_time=datetime.now(tz=UTC).isoformat(),
                details={"error": str(exc)},
            )
        finally:
            self._running_scenario = None

    @property
    def is_running(self) -> bool:
        """Whether a scenario is currently executing."""
        return self._running_scenario is not None

    @property
    def current_scenario(self) -> str | None:
        """Name of the currently running scenario, if any."""
        return self._running_scenario
