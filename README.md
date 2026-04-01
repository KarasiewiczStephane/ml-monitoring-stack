# ML Monitoring & Observability Stack

> Production ML monitoring — drift detection, performance tracking, Prometheus metrics, Grafana dashboards, and automated alerting.

## Features

- **Data Drift Detection** -- Evidently AI integration with KS test, PSI, and Wasserstein distance. Sliding window streaming detection for real-time monitoring.
- **Performance Monitoring** -- Online accuracy, precision, recall, F1 tracking with CUSUM and Page-Hinkley degradation detection.
- **Data Quality Monitoring** -- Missing value detection, out-of-range checks, schema validation, and quality scoring.
- **Prometheus Metrics** -- Custom ML metrics (prediction latency, drift scores, accuracy gauges) with configurable alert rules.
- **Grafana Dashboards** -- Pre-built dashboards for model health, data drift, performance trends, latency, and data quality.
- **Streamlit Dashboard** -- Interactive monitoring UI with drift visualization, performance charts, system health gauges, and alert history.
- **Simulation Engine** -- Test monitoring under failure scenarios: gradual drift, sudden drift, data quality degradation, latency spikes.
- **Automated Reports** -- HTML reports with drift trend analysis, performance forecasts, and actionable recommendations.

## Architecture

```
                         ┌──────────────────────────────────────────────┐
┌─────────────┐          │              FastAPI API (:8000)              │
│   Client     │────────▶│                                              │
│  Requests    │          │  /predict  /drift  /performance  /simulate   │
└─────────────┘          └────┬──────────┬──────────┬───────────────────┘
                              │          │          │
                         ┌────▼───┐ ┌────▼───┐ ┌────▼────────┐
                         │ SQLite │ │ Redis  │ │ Prometheus   │
                         │  (DB)  │ │(State) │ │  (:9090)     │
                         └────────┘ └────────┘ └──────┬───────┘
                                                      │
┌──────────────────┐                           ┌──────▼───────┐
│ Streamlit UI     │                           │   Grafana     │
│   (:8501)        │                           │   (:3000)     │
└──────────────────┘                           └──────────────┘
```

## Quick Start

### Local Development

```bash
git clone git@github.com:KarasiewiczStephane/ml-monitoring-stack.git
cd ml-monitoring-stack

# 1. Install dependencies
make install

# 2. Create the data directory (SQLite DB and reference data are generated at runtime)
mkdir -p data/reference

# 3. Start the FastAPI monitoring API (localhost:8000)
make run

# 4. In a separate terminal, launch the Streamlit dashboard (localhost:8501)
make dashboard
```

The API auto-initializes on startup: it creates the SQLite database at `data/metrics.db` and loads a default reference dataset (wine) into `data/reference/`. No manual data preparation is needed.

### Docker (Full Stack)

```bash
docker compose up -d
```

This starts the API, Prometheus, Grafana, and Redis together. Services:

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Streamlit Dashboard** (local only): http://localhost:8501

The Streamlit dashboard runs outside Docker. Launch it separately with `make dashboard`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with component status |
| GET | `/metrics` | Prometheus metrics export |
| POST | `/monitor/predict` | Log a prediction for monitoring |
| GET | `/monitor/drift` | Current drift detection status |
| GET | `/monitor/performance` | Model performance metrics |
| GET | `/monitor/report` | Generate HTML monitoring report |
| POST | `/simulate/{scenario}` | Trigger a failure simulation |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Log a prediction
curl -X POST http://localhost:8000/monitor/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"f0": 1.5, "f1": 2.3, "f2": 0.8}, "ground_truth": 1}'

# Check drift status
curl http://localhost:8000/monitor/drift

# Check performance
curl http://localhost:8000/monitor/performance

# Generate report
curl http://localhost:8000/monitor/report -o report.html

# Run a simulation
curl -X POST http://localhost:8000/simulate/gradual_drift \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Streamlit Dashboard

The Streamlit dashboard (`src/dashboard/app.py`) provides an interactive monitoring UI using synthetic demo data. It visualizes:

- **System summary** -- Prediction counts, error rate, P50/P99 latency, uptime.
- **Data drift detection** -- Per-feature drift scores with threshold overlay, reference vs. current distribution histograms.
- **Model performance** -- Accuracy, precision, recall, and F1 trends over time.
- **System health gauges** -- CPU, memory, disk, and GPU utilization.
- **Alert history** -- Filterable table of critical, warning, and info alerts.

Launch with:

```bash
make dashboard
```

## Simulation Scenarios

| Scenario | Description |
|----------|-------------|
| `gradual_drift` | Progressive feature distribution shift over configurable steps |
| `sudden_drift` | Abrupt distribution change at a configurable trigger point |
| `data_quality` | Increasing missing values and outlier injection |
| `latency_spike` | Random prediction latency spikes |

Scenario configurations live in `configs/scenarios/`.

## Configuration

Application configuration is in `configs/config.yaml`. Key settings:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `database` | `sqlite_path` | `data/metrics.db` | SQLite database path |
| `database` | `redis_url` | `redis://localhost:6379` | Redis connection URL |
| `api` | `port` | `8000` | API server port |
| `drift` | `threshold` | `0.05` | Drift detection p-value threshold |
| `drift` | `stattest` | `ks` | Statistical test (KS test) |
| `performance` | `baseline_accuracy` | `0.9` | Baseline accuracy for degradation detection |
| `reference_dataset` | -- | `wine` | Default reference dataset |

Environment variable overrides in Docker:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DATABASE_PATH` | `data/metrics.db` | SQLite database path |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | Grafana admin password |

## Development

```bash
make install    # Install dependencies
make test       # Run tests with coverage
make lint       # Lint and format with ruff
make run        # Start the API server (localhost:8000)
make dashboard  # Start the Streamlit dashboard (localhost:8501)
make clean      # Remove __pycache__ and .pyc files
```

## Project Structure

```
ml-monitoring-stack/
├── src/
│   ├── api/              # FastAPI endpoints and Pydantic schemas
│   ├── dashboard/        # Streamlit monitoring dashboard
│   ├── drift/            # Drift detection (Evidently, streaming, data quality)
│   ├── metrics/          # Prometheus metrics and collectors
│   ├── performance/      # Performance tracking and degradation detection
│   ├── reporting/        # HTML report generation with Jinja2 templates
│   ├── simulation/       # Failure scenario engine and scenario definitions
│   └── utils/            # Configuration, database, logging
├── tests/                # Unit tests (pytest)
├── configs/              # Application and scenario YAML configs
├── data/                 # Runtime data (SQLite DB, reference datasets)
├── docs/                 # Architecture documentation
├── grafana/              # Grafana provisioning (datasources, dashboards)
├── notebooks/            # Exploratory analysis
├── prometheus/           # Prometheus config and alert rules
├── docker-compose.yml    # Full stack orchestration
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Troubleshooting

**Containers not starting**: Check logs with `make docker-logs`. Ensure ports 8000, 9090, 3000, and 6379 are available.

**Redis connection errors**: The API falls back to in-memory storage when Redis is unavailable. Check Redis health with `docker compose exec redis redis-cli ping`.

**Grafana dashboards empty**: Verify Prometheus is scraping the API: visit http://localhost:9090/targets. The API must be healthy before metrics appear.

**Tests failing locally**: Ensure all dependencies are installed with `make install`. Tests run without Docker or external services.

**Dashboard not loading**: Verify `streamlit` and `plotly` are installed (`make install`). The dashboard runs independently of the API and uses synthetic demo data.


## Author

**Stéphane Karasiewicz** — [skarazdata.com](https://skarazdata.com) | [LinkedIn](https://www.linkedin.com/in/stephane-karasiewicz/)

## License

MIT
