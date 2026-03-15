from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _top_platform_line(platform_means: pd.DataFrame) -> str:
    if platform_means.empty:
        return "Platform signal was inconclusive in this run."
    top = platform_means.sort_values("Harm_Index", ascending=False).iloc[0]
    return (
        f"{top['Most_Used_Platform']} shows the highest average Harm_Index "
        f"({top['Harm_Index']:.3f}) after preprocessing."
    )


def _persona_line(persona_distribution: pd.DataFrame) -> str:
    if persona_distribution.empty:
        return "Persona split was unavailable."
    top = persona_distribution.sort_values("Count", ascending=False).iloc[0]
    return f"The most common profile is {top['Persona']} ({int(top['Count'])} students; {top['Share']:.1%})."


def build_policy_report(results: dict[str, Any], report_path: str | Path) -> str:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    platform_means = results.get("rq1_platform_means_df", pd.DataFrame())
    persona_distribution = results.get("persona_distribution_df", pd.DataFrame())
    rq3_indirect = float(results.get("rq3_primary_indirect_effect", 0.0))
    rmse_linear = float(results.get("linear_rmse", float("nan")))
    rmse_xgb = float(results.get("xgb_rmse", float("nan")))
    best_model = results.get("best_model", "unknown")

    report = f"""# Campus Policy Recommendations (Data-Backed)

## Context
This report summarizes findings from a reproducible observational pipeline on student digital harm.
Interpretation is associative and predictive, not causal proof.

## Recommendation 1: Platform-Targeted Risk Outreach
Prioritize counseling and digital wellness outreach by platform-specific harm profile rather than usage-hours alone.
Evidence: {_top_platform_line(platform_means)}

## Recommendation 2: Sleep-First Academic Support Protocol
Introduce sleep-protection interventions (quiet hours campaigns, sleep hygiene nudges, advisor check-ins) as the first response for heavy social-media users.
Evidence: the estimated mediated (indirect) effect of usage through sleep on academic harm is {rq3_indirect:.4f}; this indicates sleep is a measurable pathway, not only a side symptom.

## Recommendation 3: Persona-Segmented Intervention Tracks
Deploy separate intervention tracks for high-conflict users, hidden-addiction users, and high-usage but resilient users instead of one broad program.
Evidence: {_persona_line(persona_distribution)}

## Validation Snapshot
- Harm-index RMSE (Linear): {rmse_linear:.4f}
- Harm-index RMSE (Tree Model): {rmse_xgb:.4f}
- Best predictive model for this run: {best_model}
"""

    path.write_text(report, encoding="utf-8")
    return str(path)
