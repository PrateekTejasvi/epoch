from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERSONA_DESCRIPTIONS = {
    "The Doomscroller": "High usage, low sleep, poor mental health, and elevated overall harm.",
    "The Social Warrior": "Moderate usage but high conflict; relationships are more likely strained by social media.",
    "The Quiet Addict": "Lower visible usage with persistently high addiction and hidden risk patterns.",
    "The Balanced User": "High or moderate usage with stronger sleep and mental-health resilience.",
}


@dataclass(frozen=True)
class DashboardArtifacts:
    analysis_df: pd.DataFrame
    harm_weights: dict[str, float]
    platform_harm_df: pd.DataFrame
    persona_centroids_df: pd.DataFrame
    rq3_mediation_df: pd.DataFrame
    rq4_gender_slopes_df: pd.DataFrame
    eval_rmse_df: pd.DataFrame
    eval_group_mae_df: pd.DataFrame


def _resolve_outputs_dir(outputs_dir: str | Path) -> Path:
    candidate = Path(outputs_dir)
    if candidate.is_absolute():
        return candidate

    project_root = Path(__file__).resolve().parents[2]
    root_candidate = project_root / candidate
    cwd_candidate = Path.cwd() / candidate

    if root_candidate.exists():
        return root_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return root_candidate


def _assert_required_artifacts(base: Path) -> None:
    required = [
        base / "processed" / "analysis_enriched.csv",
        base / "tables" / "harm_weights.csv",
        base / "tables" / "persona_centroids.csv",
        base / "tables" / "rq1_platform_mean_harm.csv",
        base / "tables" / "rq3_sleep_mediation.csv",
        base / "tables" / "rq4_gender_slopes.csv",
        base / "tables" / "evaluation_rmse.csv",
        base / "tables" / "evaluation_group_mae.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required pipeline outputs are missing.\n"
            f"Expected base directory: {base}\n"
            "Missing files:\n"
            + "\n".join(missing)
            + "\nRun: python3 -m epoch.run_pipeline"
        )


def load_dashboard_artifacts(outputs_dir: str | Path = "outputs") -> DashboardArtifacts:
    base = _resolve_outputs_dir(outputs_dir)
    _assert_required_artifacts(base)
    tables = base / "tables"
    processed = base / "processed"

    analysis_df = pd.read_csv(processed / "analysis_enriched.csv")
    harm_weights_df = pd.read_csv(tables / "harm_weights.csv")
    harm_weights = dict(zip(harm_weights_df["component"], harm_weights_df["value"]))

    return DashboardArtifacts(
        analysis_df=analysis_df,
        harm_weights=harm_weights,
        platform_harm_df=pd.read_csv(tables / "rq1_platform_mean_harm.csv"),
        persona_centroids_df=pd.read_csv(tables / "persona_centroids.csv"),
        rq3_mediation_df=pd.read_csv(tables / "rq3_sleep_mediation.csv"),
        rq4_gender_slopes_df=pd.read_csv(tables / "rq4_gender_slopes.csv"),
        eval_rmse_df=pd.read_csv(tables / "evaluation_rmse.csv"),
        eval_group_mae_df=pd.read_csv(tables / "evaluation_group_mae.csv"),
    )


def _component_values(profile: dict[str, float]) -> dict[str, float]:
    return {
        "Mental_Health_Harm_Component": 10.0 - float(profile["Mental_Health_Score"]),
        "Sleep_Harm_Component": 8.0 - float(profile["Sleep_Hours_Per_Night"]),
        "Conflict_Harm_Component": float(profile["Conflicts_Over_Social_Media"]),
        "Addiction_Harm_Component": float(profile["Addicted_Score"]),
    }


def _to_percentile(value: float, distribution: pd.Series) -> float:
    min_v = float(distribution.min())
    max_v = float(distribution.max())
    if max_v - min_v < 1e-9:
        return 50.0
    return float(np.clip((value - min_v) / (max_v - min_v) * 100.0, 0.0, 100.0))


def _platform_harm_delta(selected_platform: str | None, artifacts: DashboardArtifacts) -> float:
    if not selected_platform:
        return 0.0
    platform_df = artifacts.platform_harm_df
    match = platform_df[platform_df["Most_Used_Platform"] == selected_platform]
    if match.empty:
        return 0.0
    global_mean = float(artifacts.analysis_df["Harm_Index"].mean())
    platform_mean = float(match.iloc[0]["Harm_Index"])
    return platform_mean - global_mean


def compute_harm_scores(
    profile: dict[str, float],
    artifacts: DashboardArtifacts,
) -> dict[str, float]:
    components = _component_values(profile)
    df = artifacts.analysis_df

    def zscore(value: float, column: str) -> float:
        col = df[column]
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        return 0.0 if std < 1e-9 else (value - mean) / std

    z_mental = zscore(components["Mental_Health_Harm_Component"], "Mental_Health_Harm_Component")
    z_sleep = zscore(components["Sleep_Harm_Component"], "Sleep_Harm_Component")
    z_conflict = zscore(components["Conflict_Harm_Component"], "Conflict_Harm_Component")
    z_addiction = zscore(components["Addiction_Harm_Component"], "Addiction_Harm_Component")

    harm_index_base = (
        artifacts.harm_weights["w1_mental_health_harm"] * z_mental
        + artifacts.harm_weights["w2_sleep_harm"] * z_sleep
        + artifacts.harm_weights["w3_conflict_harm"] * z_conflict
        + artifacts.harm_weights["w4_addiction_harm"] * z_addiction
    )

    harm_distribution = df["Harm_Index"]
    harm_100 = _to_percentile(harm_index_base, harm_distribution)
    harm_10 = harm_100 / 10.0

    sleep_risk = _to_percentile(components["Sleep_Harm_Component"], df["Sleep_Harm_Component"])
    mental_risk = _to_percentile(components["Mental_Health_Harm_Component"], df["Mental_Health_Harm_Component"])
    conflict_risk = _to_percentile(components["Conflict_Harm_Component"], df["Conflict_Harm_Component"])

    return {
        "harm_index_base": float(harm_index_base),
        "harm_index_base_100": float(harm_100),
        "harm_index_base_10": float(harm_10),
        # Backward-compatible aliases that now represent base harm only.
        "harm_index": float(harm_index_base),
        "harm_index_100": float(harm_100),
        "harm_index_10": float(harm_10),
        "sleep_risk_pct": float(sleep_risk),
        "mental_risk_pct": float(mental_risk),
        "conflict_risk_pct": float(conflict_risk),
        **components,
    }


def assign_persona(profile: dict[str, float], artifacts: DashboardArtifacts) -> str:
    features = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
        "Addicted_Score",
    ]
    point = np.array([float(profile[column]) for column in features], dtype=float)
    centroids = artifacts.persona_centroids_df[features].to_numpy(dtype=float)
    distances = np.linalg.norm(centroids - point.reshape(1, -1), axis=1)
    best_idx = int(np.argmin(distances))
    return str(artifacts.persona_centroids_df.iloc[best_idx]["Persona"])


def compute_academic_impact_pct(
    profile: dict[str, Any],
    persona: str,
    artifacts: DashboardArtifacts,
) -> float:
    df = artifacts.analysis_df
    subset = df[
        (df["Gender"] == profile["Gender"])
        & (df["Academic_Level"] == profile["Academic_Level"])
        & (df["Persona"] == persona)
    ]
    if len(subset) < 20:
        subset = df[(df["Gender"] == profile["Gender"]) & (df["Academic_Level"] == profile["Academic_Level"])]
    if len(subset) < 20:
        subset = df[df["Persona"] == persona]
    if len(subset) == 0:
        subset = df
    return float(subset["Academic_Harm_Binary"].mean() * 100.0)


def platform_comparison(
    selected_platform: str,
    base_harm_index: float,
    artifacts: DashboardArtifacts,
) -> pd.DataFrame:
    platform_df = artifacts.platform_harm_df.copy()
    global_mean = float(artifacts.analysis_df["Harm_Index"].mean())
    min_harm = float(artifacts.analysis_df["Harm_Index"].min())
    max_harm = float(artifacts.analysis_df["Harm_Index"].max())

    platform_df["Profile_Adjusted_Harm_Index"] = base_harm_index + (platform_df["Harm_Index"] - global_mean)
    if max_harm - min_harm < 1e-9:
        platform_df["Profile_Adjusted_Harm_10"] = 5.0
    else:
        platform_df["Profile_Adjusted_Harm_10"] = (
            (platform_df["Profile_Adjusted_Harm_Index"] - min_harm) / (max_harm - min_harm) * 10.0
        )
    platform_df["Platform_Delta"] = platform_df["Harm_Index"] - global_mean
    platform_df["Profile_Adjusted_Harm_10"] = platform_df["Profile_Adjusted_Harm_10"].clip(0.0, 10.0)
    platform_df["Selected"] = platform_df["Most_Used_Platform"] == selected_platform
    return platform_df.sort_values("Profile_Adjusted_Harm_10", ascending=False).reset_index(drop=True)


def estimate_selected_platform_harm(
    base_harm_index: float,
    selected_platform: str,
    artifacts: DashboardArtifacts,
) -> dict[str, float]:
    delta = _platform_harm_delta(selected_platform, artifacts)
    harm_index = base_harm_index + delta
    distribution = artifacts.analysis_df["Harm_Index"]
    harm_100 = _to_percentile(harm_index, distribution)
    return {
        "platform_delta": float(delta),
        "harm_index": float(harm_index),
        "harm_index_100": float(harm_100),
        "harm_index_10": float(harm_100 / 10.0),
    }


def compute_correlations(artifacts: DashboardArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
        "Addicted_Score",
        "Academic_Harm_Binary",
        "Harm_Index",
    ]
    data = artifacts.analysis_df[cols].copy()
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    return pearson, spearman


def recommendation_from_risk(composite_risk_pct: float) -> tuple[str, str]:
    if composite_risk_pct < 33:
        return (
            "Low risk",
            "Maintain current habits, keep sleep stable, and do weekly self-checks.",
        )
    if composite_risk_pct < 66:
        return (
            "Moderate risk",
            "Worth monitoring. Consider a digital wellness check-in, especially around exam periods.",
        )
    return (
        "High risk",
        "Recommend early support: sleep plan, social media limits, and advisor/counselor follow-up.",
    )


def slider_bounds(artifacts: DashboardArtifacts) -> dict[str, tuple[float, float, float]]:
    df = artifacts.analysis_df
    bounds: dict[str, tuple[float, float, float]] = {}
    columns = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Sleep_Hours_Per_Night",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media",
        "Addicted_Score",
    ]
    for column in columns:
        bounds[column] = (
            float(df[column].min()),
            float(df[column].max()),
            float(df[column].median()),
        )
    return bounds
