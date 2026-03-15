from __future__ import annotations

import unittest

from src.epoch_ui.model_service import (
    assign_persona,
    compute_academic_impact_pct,
    compute_correlations,
    compute_harm_scores,
    estimate_selected_platform_harm,
    load_dashboard_artifacts,
    platform_comparison,
    recommendation_from_risk,
    slider_bounds,
)


class TestUIModelService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = load_dashboard_artifacts("outputs")
        bounds = slider_bounds(cls.artifacts)
        cls.profile = {
            "Age": int(bounds["Age"][2]),
            "Gender": "Female",
            "Academic_Level": "Undergraduate",
            "Avg_Daily_Usage_Hours": float(bounds["Avg_Daily_Usage_Hours"][2]),
            "Sleep_Hours_Per_Night": float(bounds["Sleep_Hours_Per_Night"][2]),
            "Mental_Health_Score": float(bounds["Mental_Health_Score"][2]),
            "Conflicts_Over_Social_Media": float(bounds["Conflicts_Over_Social_Media"][2]),
            "Addicted_Score": float(bounds["Addicted_Score"][2]),
        }

    def test_harm_scores_are_bounded(self) -> None:
        scores = compute_harm_scores(self.profile, self.artifacts)
        self.assertIn("harm_index_10", scores)
        self.assertIn("harm_index_base_10", scores)
        self.assertGreaterEqual(scores["harm_index_10"], 0.0)
        self.assertLessEqual(scores["harm_index_10"], 10.0)
        self.assertGreaterEqual(scores["sleep_risk_pct"], 0.0)
        self.assertLessEqual(scores["sleep_risk_pct"], 100.0)
        self.assertAlmostEqual(scores["harm_index"], scores["harm_index_base"], places=8)

    def test_persona_and_academic_impact(self) -> None:
        persona = assign_persona(self.profile, self.artifacts)
        self.assertTrue(persona.startswith("The "))
        impact = compute_academic_impact_pct(self.profile, persona, self.artifacts)
        self.assertGreaterEqual(impact, 0.0)
        self.assertLessEqual(impact, 100.0)

    def test_platform_comparison_and_correlations(self) -> None:
        scores = compute_harm_scores(self.profile, self.artifacts)
        comparison = platform_comparison("Instagram", scores["harm_index_base"], self.artifacts)
        self.assertIn("Most_Used_Platform", comparison.columns)
        self.assertIn("Profile_Adjusted_Harm_10", comparison.columns)
        self.assertTrue((comparison["Profile_Adjusted_Harm_10"] >= 0.0).all())
        self.assertTrue((comparison["Profile_Adjusted_Harm_10"] <= 10.0).all())

        e1 = estimate_selected_platform_harm(scores["harm_index_base"], "TikTok", self.artifacts)
        e2 = estimate_selected_platform_harm(scores["harm_index_base"], "YouTube", self.artifacts)
        self.assertNotEqual(round(e1["harm_index_10"], 4), round(e2["harm_index_10"], 4))

        c1 = platform_comparison("TikTok", scores["harm_index_base"], self.artifacts)
        c2 = platform_comparison("YouTube", scores["harm_index_base"], self.artifacts)
        diff = (c1["Profile_Adjusted_Harm_10"] - c2["Profile_Adjusted_Harm_10"]).abs().max()
        self.assertLess(diff, 1e-9)

        pearson, spearman = compute_correlations(self.artifacts)
        self.assertAlmostEqual(float(pearson.loc["Harm_Index", "Harm_Index"]), 1.0, places=6)
        self.assertAlmostEqual(float(spearman.loc["Harm_Index", "Harm_Index"]), 1.0, places=6)

    def test_recommendation_labels(self) -> None:
        low = recommendation_from_risk(10.0)[0]
        moderate = recommendation_from_risk(45.0)[0]
        high = recommendation_from_risk(85.0)[0]
        self.assertEqual(low, "Low risk")
        self.assertEqual(moderate, "Moderate risk")
        self.assertEqual(high, "High risk")


if __name__ == "__main__":
    unittest.main()
