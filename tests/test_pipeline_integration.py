from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.epoch.run_pipeline import run_pipeline


class TestPipelineIntegration(unittest.TestCase):
    def test_pipeline_generates_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "outputs"
            run_pipeline(
                runtime_config={
                    "input_csv": Path("data/Students Social Media Addiction.csv"),
                    "output_dir": output_dir,
                    "processed_dir": output_dir / "processed",
                    "tables_dir": output_dir / "tables",
                    "figures_dir": output_dir / "figures",
                    "models_dir": output_dir / "models",
                    "reports_dir": output_dir / "reports",
                    "cluster_k": 4,
                    "bootstrap_iters": 200,
                    "random_seed": 42,
                }
            )

            must_exist = [
                output_dir / "processed" / "analysis_base.csv",
                output_dir / "processed" / "model_matrix.csv",
                output_dir / "processed" / "analysis_enriched.csv",
                output_dir / "tables" / "silhouette_by_k.csv",
                output_dir / "tables" / "evaluation_rmse.csv",
                output_dir / "tables" / "evaluation_group_mae.csv",
                output_dir / "reports" / "policy_recommendations.md",
                output_dir / "reports" / "pipeline_summary.json",
            ]
            for artifact in must_exist:
                self.assertTrue(artifact.exists(), msg=f"Missing artifact: {artifact}")

            report_text = (output_dir / "reports" / "policy_recommendations.md").read_text(encoding="utf-8")
            self.assertEqual(report_text.count("## Recommendation"), 3)

            summary = json.loads((output_dir / "reports" / "pipeline_summary.json").read_text(encoding="utf-8"))
            self.assertIn("rq_results", summary)
            self.assertIn("evaluation", summary)


if __name__ == "__main__":
    unittest.main()
