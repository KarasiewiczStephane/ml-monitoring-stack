"""Streamlit dashboard for the ML monitoring stack.

Visualizes data drift detection, model performance metrics, system health
gauges, and alert history using demo or simulated data.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Monitoring Stack Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def generate_drift_data() -> dict:
    """Generate synthetic drift detection results with distribution comparisons."""
    rng = np.random.default_rng(42)
    features = [f"feature_{i}" for i in range(6)]
    drift_scores = {}
    for feat in features:
        drift_scores[feat] = round(rng.uniform(0.01, 0.95), 4)

    reference_distributions = {}
    current_distributions = {}
    for feat in features:
        ref = rng.standard_normal(500)
        shift = rng.uniform(-0.5, 2.0) if drift_scores[feat] < 0.3 else rng.uniform(-0.1, 0.1)
        cur = rng.standard_normal(500) + shift
        reference_distributions[feat] = ref.tolist()
        current_distributions[feat] = cur.tolist()

    return {
        "dataset_drift_detected": True,
        "drift_scores": drift_scores,
        "threshold": 0.05,
        "drifted_features": [f for f, s in drift_scores.items() if s < 0.05],
        "reference_distributions": reference_distributions,
        "current_distributions": current_distributions,
    }


@st.cache_data
def generate_performance_over_time() -> list[dict]:
    """Generate synthetic model performance metrics over time."""
    base_time = datetime(2025, 2, 1, 0, 0, 0, tzinfo=UTC)
    entries = []
    accuracy = 0.94
    rng = np.random.default_rng(123)

    for i in range(30):
        degradation = max(0, (i - 15) * 0.003) if i > 15 else 0
        noise = rng.normal(0, 0.005)
        accuracy_val = max(0.80, min(0.98, accuracy - degradation + noise))

        entries.append(
            {
                "timestamp": (base_time + timedelta(days=i)).isoformat(),
                "accuracy": round(accuracy_val, 4),
                "precision": round(accuracy_val - 0.01, 4),
                "recall": round(accuracy_val - 0.005, 4),
                "f1": round(accuracy_val - 0.008, 4),
                "sample_count": int(rng.integers(800, 1200)),
            }
        )
    return entries


@st.cache_data
def generate_system_health() -> dict:
    """Generate synthetic system health metrics."""
    return {
        "cpu_percent": 42.5,
        "memory_percent": 67.3,
        "disk_percent": 55.8,
        "gpu_percent": 78.2,
        "inference_latency_p50_ms": 12.4,
        "inference_latency_p99_ms": 45.8,
        "active_models": 3,
        "total_predictions_24h": 28450,
        "error_rate_24h": 0.0012,
        "uptime_hours": 720.5,
    }


@st.cache_data
def generate_alert_history() -> list[dict]:
    """Generate synthetic alert history entries."""
    base_time = datetime(2025, 3, 1, 0, 0, 0, tzinfo=UTC)
    alerts = [
        {
            "timestamp": (base_time - timedelta(hours=2)).isoformat(),
            "severity": "warning",
            "source": "drift_detector",
            "message": "Feature feature_0 drift score below threshold (p=0.023)",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=5)).isoformat(),
            "severity": "critical",
            "source": "performance_tracker",
            "message": "Model accuracy dropped below 0.90 threshold",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=8)).isoformat(),
            "severity": "info",
            "source": "system_monitor",
            "message": "Scheduled reference data refresh completed",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=12)).isoformat(),
            "severity": "warning",
            "source": "system_monitor",
            "message": "Memory usage exceeded 80% threshold",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=18)).isoformat(),
            "severity": "critical",
            "source": "drift_detector",
            "message": "Dataset-level drift detected across 3 features",
            "resolved": False,
        },
        {
            "timestamp": (base_time - timedelta(hours=24)).isoformat(),
            "severity": "warning",
            "source": "performance_tracker",
            "message": "Prediction latency p99 exceeded 50ms threshold",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=36)).isoformat(),
            "severity": "info",
            "source": "drift_detector",
            "message": "New reference dataset registered: wine_v3",
            "resolved": True,
        },
        {
            "timestamp": (base_time - timedelta(hours=48)).isoformat(),
            "severity": "warning",
            "source": "system_monitor",
            "message": "GPU utilization sustained above 90% for 30 minutes",
            "resolved": True,
        },
    ]
    return alerts


def render_header() -> None:
    """Render the dashboard header with high-level system stats."""
    st.title("ML Monitoring Stack Dashboard")
    st.caption("Data drift detection, model performance tracking, system health, and alerts")


def render_system_summary(health: dict) -> None:
    """Render top-level system summary metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Predictions (24h)", f"{health['total_predictions_24h']:,}")
    col2.metric("Error Rate", f"{health['error_rate_24h']:.4%}")
    col3.metric("P50 Latency", f"{health['inference_latency_p50_ms']:.1f}ms")
    col4.metric("P99 Latency", f"{health['inference_latency_p99_ms']:.1f}ms")
    col5.metric("Uptime", f"{health['uptime_hours']:.0f}h")


def render_drift_panels(drift_data: dict) -> None:
    """Render data drift detection panels with distribution comparison charts."""
    st.subheader("Data Drift Detection")

    drift_status = "DETECTED" if drift_data["dataset_drift_detected"] else "NOT DETECTED"
    drift_color = "red" if drift_data["dataset_drift_detected"] else "green"
    st.markdown(f"Dataset drift: :{drift_color}[**{drift_status}**]")

    scores = drift_data["drift_scores"]
    threshold = drift_data["threshold"]

    fig = go.Figure()
    colors = ["#F44336" if s < threshold else "#4CAF50" for s in scores.values()]
    fig.add_trace(
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker_color=colors,
            text=[f"{s:.4f}" for s in scores.values()],
            textposition="auto",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold})",
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="P-Value (Drift Score)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_feature = st.selectbox(
        "Compare distributions for feature:",
        list(scores.keys()),
    )

    if selected_feature:
        ref_data = drift_data["reference_distributions"][selected_feature]
        cur_data = drift_data["current_distributions"][selected_feature]

        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=ref_data,
                name="Reference",
                opacity=0.6,
                marker_color="#2196F3",
                nbinsx=40,
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=cur_data,
                name="Current",
                opacity=0.6,
                marker_color="#FF9800",
                nbinsx=40,
            )
        )
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title="Value",
            yaxis_title="Count",
            height=350,
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            title=f"Distribution Comparison: {selected_feature}",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


def render_performance_metrics(perf_data: list[dict]) -> None:
    """Render model performance metrics over time."""
    st.subheader("Model Performance Over Time")

    df = pd.DataFrame(perf_data)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

    metric_choice = st.multiselect(
        "Select metrics to display:",
        ["accuracy", "precision", "recall", "f1"],
        default=["accuracy", "f1"],
    )

    fig = go.Figure()
    colors = {"accuracy": "#2196F3", "precision": "#4CAF50", "recall": "#FF9800", "f1": "#9C27B0"}

    for metric in metric_choice:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[metric],
                mode="lines+markers",
                name=metric.title(),
                line={"color": colors.get(metric, "#999"), "width": 2},
                marker={"size": 4},
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis={"range": [0.80, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_system_health_gauges(health: dict) -> None:
    """Render system health gauges for CPU, memory, disk, and GPU."""
    st.subheader("System Health")

    gauges = [
        ("CPU", health["cpu_percent"]),
        ("Memory", health["memory_percent"]),
        ("Disk", health["disk_percent"]),
        ("GPU", health["gpu_percent"]),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, gauges):
        with col:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={"text": label},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2196F3"},
                        "steps": [
                            {"range": [0, 60], "color": "#E8F5E9"},
                            {"range": [60, 80], "color": "#FFF3E0"},
                            {"range": [80, 100], "color": "#FFEBEE"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 2},
                            "thickness": 0.8,
                            "value": 85,
                        },
                    },
                    number={"suffix": "%"},
                )
            )
            fig.update_layout(
                height=200,
                margin={"l": 20, "r": 20, "t": 40, "b": 10},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_alert_history(alerts: list[dict]) -> None:
    """Render alert history table."""
    st.subheader("Alert History")

    severity_filter = st.multiselect(
        "Filter by severity:",
        ["critical", "warning", "info"],
        default=["critical", "warning", "info"],
    )

    filtered = [a for a in alerts if a["severity"] in severity_filter]
    df = pd.DataFrame(filtered)

    if df.empty:
        st.info("No alerts match the selected filters.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    df["resolved"] = df["resolved"].map({True: "Yes", False: "No"})

    st.dataframe(
        df[["timestamp", "severity", "source", "message", "resolved"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    health = generate_system_health()
    render_system_summary(health)
    st.markdown("---")

    st.sidebar.markdown("### Dashboard Controls")
    show_drift = st.sidebar.checkbox("Show drift detection", value=True)
    show_performance = st.sidebar.checkbox("Show performance metrics", value=True)
    show_health = st.sidebar.checkbox("Show system health", value=True)
    show_alerts = st.sidebar.checkbox("Show alert history", value=True)

    if show_drift:
        drift_data = generate_drift_data()
        render_drift_panels(drift_data)
        st.markdown("---")

    if show_performance:
        perf_data = generate_performance_over_time()
        render_performance_metrics(perf_data)
        st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        if show_health:
            render_system_health_gauges(health)

    with col_right:
        if show_alerts:
            alerts = generate_alert_history()
            render_alert_history(alerts)


if __name__ == "__main__":
    main()
