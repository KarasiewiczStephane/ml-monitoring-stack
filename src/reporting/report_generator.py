"""Automated HTML report generator with trend analysis and recommendations."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = str(Path(__file__).parent / "templates")


class ReportGenerator:
    """Generate HTML monitoring reports with drift analysis and recommendations.

    Combines performance metrics, drift history, and trend analysis
    into a comprehensive HTML report using Jinja2 templates.

    Args:
        template_dir: Directory containing Jinja2 templates.
    """

    def __init__(self, template_dir: str | None = None) -> None:
        template_path = template_dir or _TEMPLATE_DIR
        self.env = Environment(loader=FileSystemLoader(template_path))
        self.template = self.env.get_template("report.html")

    def analyze_drift_trend(self, drift_history: list[dict]) -> str:
        """Analyze the trajectory of drift scores over time.

        Args:
            drift_history: List of drift result dicts, each containing
                an 'overall_score' key.

        Returns:
            Human-readable trend description.
        """
        if len(drift_history) < 3:
            return "Insufficient data for trend analysis"

        scores = []
        for entry in drift_history:
            score = entry.get("overall_score")
            if score is None:
                scores_dict = entry.get("scores", {})
                if scores_dict:
                    score = float(np.mean(list(scores_dict.values())))
                else:
                    score = 0.0
            scores.append(float(score))

        coeffs = np.polyfit(range(len(scores)), scores, 1)
        slope = coeffs[0]

        if slope > 0.01:
            return "Increasing drift - monitor closely"
        if slope < -0.01:
            return "Decreasing drift - stabilizing"
        return "Stable"

    def generate_recommendations(
        self,
        drift_status: dict,
        perf_status: dict,
    ) -> list[str]:
        """Generate actionable recommendations based on current status.

        Args:
            drift_status: Current drift detection results.
            perf_status: Current performance metrics.

        Returns:
            List of recommendation strings.
        """
        recs: list[str] = []

        if drift_status.get("drift_detected"):
            recs.append("Consider retraining the model with recent data")
            drifted = drift_status.get("drifted_features", [])
            if drifted:
                feature_list = ", ".join(drifted[:5])
                recs.append(f"Investigate drifted features: {feature_list}")

        accuracy = perf_status.get("accuracy", 1.0)
        if accuracy < 0.85:
            recs.append("Model accuracy below threshold - retraining recommended")

        if perf_status.get("f1_macro", 1.0) < 0.8:
            recs.append("F1 score is low - check class-level performance")

        if not recs:
            recs.append("Model performing within expected parameters")

        return recs

    def _compute_health_score(
        self,
        drift_history: list[dict],
        performance_metrics: dict,
    ) -> float:
        """Compute an overall health score between 0 and 1.

        Args:
            drift_history: Recent drift detection results.
            performance_metrics: Current performance metrics.

        Returns:
            Health score where 1.0 is fully healthy.
        """
        score = 1.0

        # Penalize for recent drift detections
        recent = drift_history[-5:] if drift_history else []
        drift_count = sum(1 for d in recent if d.get("drift_detected"))
        score -= drift_count * 0.1

        # Penalize for low accuracy
        accuracy = performance_metrics.get("accuracy", 1.0)
        if accuracy < 0.9:
            score -= (0.9 - accuracy) * 2

        return max(0.0, min(1.0, score))

    def _compute_perf_change(self, performance_metrics: dict) -> float:
        """Compute performance change as a percentage.

        Args:
            performance_metrics: Current performance metrics containing
                'accuracy' or similar.

        Returns:
            Percentage change from a baseline of 0.9.
        """
        accuracy = performance_metrics.get("accuracy", 0.9)
        baseline = 0.9
        if baseline == 0:
            return 0.0
        return ((accuracy - baseline) / baseline) * 100

    def generate_report(
        self,
        drift_history: list[dict],
        performance_metrics: dict,
        output_path: str = "reports/report.html",
    ) -> str:
        """Generate a complete HTML monitoring report.

        Args:
            drift_history: Historical drift detection results.
            performance_metrics: Current model performance metrics.
            output_path: Where to write the HTML report file.

        Returns:
            Path to the generated report file.
        """
        now = datetime.now()
        context = {
            "report_date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "start_date": (now - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end_date": now.strftime("%Y-%m-%d"),
            "health_score": self._compute_health_score(drift_history, performance_metrics),
            "drift_detected": any(d.get("drift_detected") for d in drift_history[-5:]),
            "performance_change": self._compute_perf_change(performance_metrics),
            "performance_metrics": {
                k: v for k, v in performance_metrics.items() if isinstance(v, int | float)
            },
            "drift_trend": self.analyze_drift_trend(drift_history),
            "performance_forecast": "Stable"
            if performance_metrics.get("accuracy", 0.9) > 0.85
            else "Declining",
            "recommendations": self.generate_recommendations(
                drift_history[-1] if drift_history else {},
                performance_metrics,
            ),
        }

        html = self.template.render(**context)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)

        logger.info("Report generated at %s", output_path)
        return str(output)
