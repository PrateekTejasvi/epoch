from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.epoch.clustering import run_clustering
from src.epoch.harm import compute_harm_index
from src.epoch.preprocessing import run_preprocessing


class TestHarmAndClustering(unittest.TestCase):
    def test_harm_weights_and_cluster_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = run_preprocessing(
                input_csv=Path("data/Students Social Media Addiction.csv"),
                out_dir=Path(tmp_dir),
            )
            base_df = pd.read_csv(outputs["analysis_base"])
            with_harm, weights = compute_harm_index(base_df)

            weight_values = [
                weights["w1_mental_health_harm"],
                weights["w2_sleep_harm"],
                weights["w3_conflict_harm"],
                weights["w4_addiction_harm"],
            ]
            self.assertAlmostEqual(sum(weight_values), 1.0, places=6)
            self.assertTrue(all(value > 0 for value in weight_values))
            self.assertIn("Harm_Index", with_harm.columns)

            first = run_clustering(with_harm, k=4, random_seed=42)["data"]
            second = run_clustering(with_harm, k=4, random_seed=42)["data"]
            merged = first[["Student_ID", "Persona"]].merge(
                second[["Student_ID", "Persona"]],
                on="Student_ID",
                suffixes=("_a", "_b"),
            )
            self.assertTrue((merged["Persona_a"] == merged["Persona_b"]).all())


if __name__ == "__main__":
    unittest.main()
