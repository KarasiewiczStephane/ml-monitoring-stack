"""FastAPI application with monitoring, drift, performance, and simulation endpoints."""

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CollectorRegistry, generate_latest

from src.api.schemas import (
    DriftStatus,
    HealthResponse,
    PerformanceStatus,
    PredictionRequest,
    PredictionResponse,
    SimulationRequest,
)
from src.drift.data_quality import DataQualityMonitor
from src.drift.detector import DriftDetector
from src.drift.reference_manager import ReferenceManager
from src.metrics.prometheus_metrics import MLMetrics
from src.performance.degradation_detector import DegradationDetector
from src.performance.tracker import PerformanceTracker
from src.reporting.report_generator import ReportGenerator
from src.simulation.scenario_runner import ScenarioRunner
from src.simulation.scenarios.data_quality import DataQualityScenario
from src.simulation.scenarios.gradual_drift import GradualDriftScenario
from src.simulation.scenarios.latency_spike import LatencySpikeScenario
from src.simulation.scenarios.sudden_drift import SuddenDriftScenario
from src.utils.config import Settings, load_config
from src.utils.database import MetricsDB

logger = logging.getLogger(__name__)

START_TIME = time.time()
VERSION = "1.0.0"

_registry = CollectorRegistry()
_ml_metrics = MLMetrics(registry=_registry)
_config: Settings | None = None
_db: MetricsDB | None = None
_drift_detector: DriftDetector | None = None
_perf_tracker: PerformanceTracker | None = None
_degradation_detector: DegradationDetector | None = None
_report_generator: ReportGenerator | None = None
_scenario_runner: ScenarioRunner | None = None
_quality_monitor: DataQualityMonitor | None = None
_reference_manager: ReferenceManager | None = None
_drift_history: list[dict] = []

VALID_SCENARIOS = ["gradual_drift", "sudden_drift", "data_quality", "latency_spike"]


def _initialize_components() -> None:
    """Initialize all monitoring components from configuration."""
    global _config, _db, _drift_detector, _perf_tracker
    global _degradation_detector, _report_generator, _scenario_runner
    global _quality_monitor, _reference_manager

    _config = load_config()
    _db = MetricsDB(_config.database.sqlite_path)
    _reference_manager = ReferenceManager(_config.reference_dir)

    _reference_manager.initialize_default_reference(_config.reference_dataset)

    _drift_detector = DriftDetector(
        config=_config.drift,
        db=_db,
        reference_manager=_reference_manager,
    )
    _drift_detector.initialize(_config.reference_dataset)

    ref_stats = _drift_detector.get_reference_stats() or {}
    _quality_monitor = DataQualityMonitor(reference_stats=ref_stats)

    _perf_tracker = PerformanceTracker(
        task_type=_config.performance.task_type,
        db=_db,
    )

    _degradation_detector = DegradationDetector(
        baseline_accuracy=_config.performance.baseline_accuracy,
        config={
            "cusum_threshold": _config.performance.cusum_threshold,
            "cusum_drift": _config.performance.cusum_drift,
            "ph_threshold": _config.performance.ph_threshold,
            "ph_delta": _config.performance.ph_delta,
        },
    )

    _report_generator = ReportGenerator()

    _scenario_runner = ScenarioRunner(scenarios_dir="configs/scenarios")
    _scenario_runner.register_scenario("gradual_drift", GradualDriftScenario)
    _scenario_runner.register_scenario("sudden_drift", SuddenDriftScenario)
    _scenario_runner.register_scenario("data_quality", DataQualityScenario)
    _scenario_runner.register_scenario("latency_spike", LatencySpikeScenario)

    logger.info("All monitoring components initialized")


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    _initialize_components()
    yield


app = FastAPI(title="ML Monitoring API", version=VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return the health status of the monitoring service."""
    return HealthResponse(
        status="healthy",
        version=VERSION,
        uptime_seconds=time.time() - START_TIME,
        components={
            "database": "ok" if _db else "unavailable",
            "drift_detector": "ok"
            if _drift_detector and _drift_detector.is_initialized
            else "unavailable",
            "performance_tracker": "ok" if _perf_tracker else "unavailable",
            "simulation_engine": "ok" if _scenario_runner else "unavailable",
        },
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(_registry),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/monitor/predict", response_model=PredictionResponse)
async def log_prediction(request: PredictionRequest) -> PredictionResponse:
    """Log a prediction for monitoring.

    Accepts feature values and optionally ground truth, records the
    prediction in the database, and returns a mock prediction response
    with drift warning status.
    """
    if _perf_tracker is None or _ml_metrics is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    with _ml_metrics.track_prediction_time(request.model_name):
        prediction = 0
        confidence = 0.85

        _ml_metrics.record_prediction(str(prediction), request.model_name)

        if request.ground_truth is not None:
            _perf_tracker.log_prediction(prediction, request.ground_truth, confidence)

        drift_warning = False
        if _drift_history:
            last = _drift_history[-1]
            drift_warning = last.get("drift_detected", False)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        timestamp=datetime.now(tz=UTC),
        drift_warning=drift_warning,
    )


@app.get("/monitor/drift", response_model=DriftStatus)
async def get_drift_status() -> DriftStatus:
    """Return the current drift detection status."""
    if not _drift_history:
        return DriftStatus(
            drift_detected=False,
            overall_score=0.0,
            feature_scores={},
            drifted_features=[],
            last_check=datetime.now(tz=UTC),
        )

    last = _drift_history[-1]
    scores = last.get("scores", {})
    overall = float(sum(scores.values()) / max(len(scores), 1)) if scores else 0.0

    return DriftStatus(
        drift_detected=last.get("drift_detected", False),
        overall_score=overall,
        feature_scores=scores,
        drifted_features=last.get("drifted_features", []),
        last_check=datetime.now(tz=UTC),
    )


@app.get("/monitor/performance", response_model=PerformanceStatus)
async def get_performance() -> PerformanceStatus:
    """Return current model performance metrics."""
    if _perf_tracker is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    perf = _perf_tracker.compute_metrics()
    alerts = []
    if _degradation_detector:
        alerts = [a.message for a in _degradation_detector.alerts[-10:]]

    return PerformanceStatus(
        accuracy=perf.get("accuracy"),
        precision=perf.get("precision_macro"),
        recall=perf.get("recall_macro"),
        f1=perf.get("f1_macro"),
        sample_count=perf.get("sample_count", 0),
        degradation_alerts=alerts,
    )


@app.get("/monitor/report", response_class=HTMLResponse)
async def generate_report() -> HTMLResponse:
    """Generate and return an HTML monitoring report."""
    if _report_generator is None or _perf_tracker is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    perf = _perf_tracker.compute_metrics()
    path = _report_generator.generate_report(
        drift_history=_drift_history,
        performance_metrics=perf,
        output_path="reports/latest_report.html",
    )

    with open(path) as f:
        html = f.read()
    return HTMLResponse(content=html)


def _run_scenario_task(scenario: str) -> None:
    """Background task to execute a simulation scenario."""
    if _scenario_runner is None:
        return

    def on_data(data: object) -> None:
        logger.debug("Scenario '%s' produced data batch", scenario)

    result = _scenario_runner.run_scenario(scenario, data_callback=on_data)
    logger.info("Scenario '%s' completed: %s", scenario, result.status)


@app.post("/simulate/{scenario}")
async def run_simulation(
    scenario: str,
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Trigger a failure simulation scenario.

    The scenario runs in the background. Valid scenarios are:
    gradual_drift, sudden_drift, data_quality, latency_spike.
    """
    if scenario not in VALID_SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario}'. Valid: {VALID_SCENARIOS}",
        )
    if _scenario_runner is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if _scenario_runner.is_running:
        raise HTTPException(
            status_code=409,
            detail=f"Scenario '{_scenario_runner.current_scenario}' already running",
        )

    background_tasks.add_task(_run_scenario_task, scenario)
    return {"status": "started", "scenario": scenario}
