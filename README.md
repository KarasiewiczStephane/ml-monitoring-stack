# ML Monitoring & Observability Stack

Production-grade ML model monitoring system with drift detection, performance tracking, alerting, and observability dashboards.

## Features

- **Data Drift Detection** — Evidently AI integration with KS test, PSI, and Wasserstein distance. Sliding window streaming detection for real-time monitoring.
- **Performance Monitoring** — Online accuracy, precision, recall, F1 tracking with CUSUM and Page-Hinkley degradation detection.
- **Data Quality Monitoring** — Missing value detection, out-of-range checks, schema validation, and quality scoring.
- **Prometheus Metrics** — Custom ML metrics (prediction latency, drift scores, accuracy gauges) with configurable alert rules.
- **Grafana Dashboards** — Pre-built dashboards for model health, data drift, performance trends, latency, and data quality.
- **Simulation Engine** — Test monitoring under failure scenarios: gradual drift, sudden drift, data quality degradation, latency spikes.
- **Automated Reports** — HTML reports with drift trend analysis, performance forecasts, and actionable recommendations.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Client     │────▶│              FastAPI API (:8000)              │
│  Requests    │     │                                              │
└─────────────┘     │  /predict  /drift  /performance  /simulate   │
                    └────┬──────────┬──────────┬───────────────────┘
                         │          │          │
                    ┌────▼───┐ ┌────▼───┐ ┌────▼────────┐
                    │ SQLite │ │ Redis  │ │ Prometheus   │
                    │  (DB)  │ │(State) │ │  (:9090)     │
                    └────────┘ └────────┘ └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │   Grafana     │
                                          │   (:3000)     │
                                          └──────────────┘
```

## Quick Start

```bash
git clone git@github.com:KarasiewiczStephane/ml-monitoring-stack.git
cd ml-monitoring-stack

# Start the full stack
docker compose up -d
```

Services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

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

## Simulation Scenarios

| Scenario | Description |
|----------|-------------|
| `gradual_drift` | Progressive feature distribution shift over configurable steps |
| `sudden_drift` | Abrupt distribution change at a configurable trigger point |
| `data_quality` | Increasing missing values and outlier injection |
| `latency_spike` | Random prediction latency spikes |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DATABASE_PATH` | `data/metrics.db` | SQLite database path |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | Grafana admin password |

## Development

```bash
# Install dependencies
make install

# Run tests
make test

# Lint and format
make lint

# Run locally (without Docker)
make run
```

## Project Structure

```
ml-monitoring-stack/
├── src/
│   ├── api/              # FastAPI endpoints and schemas
│   ├── drift/            # Drift detection (Evidently, streaming, data quality)
│   ├── metrics/          # Prometheus metrics and collectors
│   ├── performance/      # Performance tracking and degradation detection
│   ├── reporting/        # HTML report generation with Jinja2 templates
│   ├── simulation/       # Failure scenario engine
│   └── utils/            # Configuration, database, logging
├── tests/                # Unit tests (pytest)
├── configs/              # Application and scenario YAML configs
├── prometheus/           # Prometheus config and alert rules
├── grafana/              # Grafana provisioning (datasources, dashboards)
├── .github/workflows/    # CI pipeline
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

## License

MIT
