from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.epoch.preprocessing import run_preprocessing


class TestPreprocessing(unittest.TestCase):
    def test_run_preprocessing_outputs_expected_files_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            outputs = run_preprocessing(
                input_csv=Path("data/Students Social Media Addiction.csv"),
                out_dir=output_dir,
            )

            analysis_path = Path(outputs["analysis_base"])
            model_matrix_path = Path(outputs["model_matrix"])
            self.assertTrue(analysis_path.exists())
            self.assertTrue(model_matrix_path.exists())

            analysis_df = pd.read_csv(analysis_path)
            self.assertIn("Country_Grouped", analysis_df.columns)
            self.assertIn("Academic_Harm_Binary", analysis_df.columns)
            self.assertEqual(int(analysis_df.isna().sum().sum()), 0)
            self.assertGreaterEqual(len(analysis_df), 700)

            model_df = pd.read_csv(model_matrix_path)
            self.assertIn("Student_ID", model_df.columns)
            self.assertIn("Addicted_Score", model_df.columns)
            self.assertEqual(len(model_df), len(analysis_df))


if __name__ == "__main__":
    unittest.main()
