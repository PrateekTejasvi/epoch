from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _coef_table(model: Any, model_name: str) -> pd.DataFrame:
    table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    table.insert(0, "model_name", model_name)
    return table


def _fdr_pairwise_from_model(model: Any, factor: str, outcome: str) -> pd.DataFrame:
    try:
        pairwise = model.t_test_pairwise(factor, method="fdr_bh")
        frame = pairwise.result_frame.reset_index().rename(columns={"index": "comparison"})
        frame.insert(0, "outcome", outcome)
        return frame
    except Exception:
        return pd.DataFrame()


def _fallback_tukey(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    tukey = pairwise_tukeyhsd(endog=df[outcome].astype(float), groups=df["Most_Used_Platform"], alpha=0.05)
    summary = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    summary.insert(0, "outcome", outcome)
    summary.insert(1, "test_type", "tukey_hsd_unadjusted_covariates")
    return summary


def _fit_rq1_models(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> dict[str, Any]:
    outcomes = [
        "Harm_Index",
        "Sleep_Harm_Component",
        "Mental_Health_Harm_Component",
        "Conflicts_Over_Social_Media",
    ]
    formulas = {
        outcome: (
            f"{outcome} ~ C(Most_Used_Platform) + Avg_Daily_Usage_Hours + Age + "
            "C(Gender) + C(Academic_Level) + C(Country_Grouped)"
        )
        for outcome in outcomes
    }

    model_coefs: list[pd.DataFrame] = []
    platform_effects: list[pd.DataFrame] = []
    pairwise_rows: list[pd.DataFrame] = []

    for outcome in outcomes:
        model = smf.ols(formulas[outcome], data=df).fit()
        coef_df = _coef_table(model, model_name=f"RQ1_{outcome}")
        model_coefs.append(coef_df)

        platform_df = coef_df[coef_df["term"].str.contains(r"C\(Most_Used_Platform\)", regex=True)].copy()
        platform_df.insert(1, "outcome", outcome)
        platform_effects.append(platform_df)

        pairwise_df = _fdr_pairwise_from_model(model, "C(Most_Used_Platform)", outcome)
        if pairwise_df.empty:
            pairwise_df = _fallback_tukey(df, outcome)
        pairwise_rows.append(pairwise_df)

    rq1_coef_path = tables_dir / "rq1_model_coefficients.csv"
    rq1_platform_path = tables_dir / "rq1_platform_effects.csv"
    rq1_pairwise_path = tables_dir / "rq1_platform_pairwise_fdr.csv"
    pd.concat(model_coefs, ignore_index=True).to_csv(rq1_coef_path, index=False)
    pd.concat(platform_effects, ignore_index=True).to_csv(rq1_platform_path, index=False)
    pd.concat(pairwise_rows, ignore_index=True).to_csv(rq1_pairwise_path, index=False)

    platform_means = (
        df.groupby("Most_Used_Platform", as_index=False)["Harm_Index"].mean().sort_values("Harm_Index", ascending=False)
    )
    platform_means_path = tables_dir / "rq1_platform_mean_harm.csv"
    platform_means.to_csv(platform_means_path, index=False)

    rq1_figure_path = figures_dir / "rq1_platform_harm_fingerprint.png"
    plt.figure(figsize=(10, 5))
    sns.barplot(data=platform_means, x="Most_Used_Platform", y="Harm_Index", color="#1273de")
    plt.xticks(rotation=35, ha="right")
    plt.title("RQ1: Harm Fingerprint by Most Used Platform")
    plt.tight_layout()
    plt.savefig(rq1_figure_path, dpi=140)
    plt.close()

    return {
        "rq1_model_coefficients": str(rq1_coef_path),
        "rq1_platform_effects": str(rq1_platform_path),
        "rq1_pairwise_tests": str(rq1_pairwise_path),
        "rq1_platform_means": str(platform_means_path),
        "rq1_platform_figure": str(rq1_figure_path),
        "platform_means_df": platform_means,
    }


def _fit_rq2_model(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> dict[str, Any]:
    formula = (
        "Harm_Index ~ Avg_Daily_Usage_Hours * C(Academic_Level) + "
        "Age + C(Gender) + C(Country_Grouped)"
    )
    model = smf.ols(formula, data=df).fit()
    coef_df = _coef_table(model, model_name="RQ2_Academic_Level_Interaction")
    rq2_coef_path = tables_dir / "rq2_model_coefficients.csv"
    coef_df.to_csv(rq2_coef_path, index=False)

    interaction_df = coef_df[coef_df["term"].str.contains("Avg_Daily_Usage_Hours", regex=False)].copy()
    interaction_path = tables_dir / "rq2_interaction_terms.csv"
    interaction_df.to_csv(interaction_path, index=False)

    hours = np.linspace(df["Avg_Daily_Usage_Hours"].min(), df["Avg_Daily_Usage_Hours"].max(), 60)
    reference_gender = df["Gender"].mode().iloc[0]
    reference_country = df["Country_Grouped"].mode().iloc[0]
    reference_age = float(df["Age"].mean())
    levels = sorted(df["Academic_Level"].dropna().unique().tolist())

    pred_rows: list[pd.DataFrame] = []
    for level in levels:
        grid = pd.DataFrame(
            {
                "Avg_Daily_Usage_Hours": hours,
                "Academic_Level": level,
                "Age": reference_age,
                "Gender": reference_gender,
                "Country_Grouped": reference_country,
            }
        )
        grid["Predicted_Harm_Index"] = model.predict(grid)
        pred_rows.append(grid)
    prediction_df = pd.concat(pred_rows, ignore_index=True)
    prediction_path = tables_dir / "rq2_predicted_lines.csv"
    prediction_df.to_csv(prediction_path, index=False)

    rq2_figure_path = figures_dir / "rq2_usage_academic_interaction.png"
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=prediction_df,
        x="Avg_Daily_Usage_Hours",
        y="Predicted_Harm_Index",
        hue="Academic_Level",
        linewidth=2.2,
    )
    plt.title("RQ2: Academic-Level Interaction at Equal Usage Hours")
    plt.tight_layout()
    plt.savefig(rq2_figure_path, dpi=140)
    plt.close()

    return {
        "rq2_model_coefficients": str(rq2_coef_path),
        "rq2_interaction_terms": str(interaction_path),
        "rq2_predictions": str(prediction_path),
        "rq2_figure": str(rq2_figure_path),
    }


def _fit_linear_coef(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def _mediation_bootstrap(
    df: pd.DataFrame,
    outcome_col: str,
    boot_iters: int = 2000,
    seed: int = 42,
) -> tuple[dict[str, float], np.ndarray]:
    covariates = pd.get_dummies(
        df[["Age", "Gender", "Academic_Level", "Country_Grouped"]],
        drop_first=True,
    ).astype(float)
    X = df["Avg_Daily_Usage_Hours"].to_numpy(dtype=float)
    M = df["Sleep_Hours_Per_Night"].to_numpy(dtype=float)
    Y = df[outcome_col].to_numpy(dtype=float)
    C = covariates.to_numpy(dtype=float)

    n = len(df)
    intercept = np.ones((n, 1), dtype=float)
    x_col = X.reshape(-1, 1)
    m_col = M.reshape(-1, 1)

    design_m = np.column_stack([intercept, x_col, C])
    coef_m = _fit_linear_coef(design_m, M)
    a_coef = float(coef_m[1])

    design_y = np.column_stack([intercept, x_col, m_col, C])
    coef_y = _fit_linear_coef(design_y, Y)
    c_prime = float(coef_y[1])
    b_coef = float(coef_y[2])

    design_total = np.column_stack([intercept, x_col, C])
    coef_total = _fit_linear_coef(design_total, Y)
    c_total = float(coef_total[1])

    indirect = a_coef * b_coef
    rng = np.random.default_rng(seed)
    boot_indirect = np.zeros(boot_iters, dtype=float)

    for i in range(boot_iters):
        idx = rng.integers(0, n, size=n)
        m_coef_boot = _fit_linear_coef(design_m[idx], M[idx])
        y_coef_boot = _fit_linear_coef(design_y[idx], Y[idx])
        boot_indirect[i] = float(m_coef_boot[1] * y_coef_boot[2])

    ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    p_two_tailed = 2 * min(np.mean(boot_indirect <= 0), np.mean(boot_indirect >= 0))

    result = {
        "a_effect_usage_to_sleep": a_coef,
        "b_effect_sleep_to_outcome": b_coef,
        "direct_effect_c_prime": c_prime,
        "total_effect_c": c_total,
        "indirect_effect_ab": indirect,
        "indirect_ci_lower_95": float(ci_low),
        "indirect_ci_upper_95": float(ci_high),
        "indirect_pvalue_approx": float(p_two_tailed),
    }
    return result, boot_indirect


def _fit_rq3_mediation(
    df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    boot_iters: int,
    seed: int,
) -> dict[str, Any]:
    primary, primary_boot = _mediation_bootstrap(
        df=df,
        outcome_col="Academic_Harm_Binary",
        boot_iters=boot_iters,
        seed=seed,
    )
    sensitivity, sensitivity_boot = _mediation_bootstrap(
        df=df,
        outcome_col="Harm_Index",
        boot_iters=boot_iters,
        seed=seed + 1,
    )

    primary_row = {"analysis": "primary_academic_harm_binary", **primary}
    sensitivity_row = {"analysis": "sensitivity_harm_index", **sensitivity}
    mediation_df = pd.DataFrame([primary_row, sensitivity_row])
    mediation_path = tables_dir / "rq3_sleep_mediation.csv"
    mediation_df.to_csv(mediation_path, index=False)

    distribution_df = pd.DataFrame(
        {
            "bootstrap_indirect_effect": np.concatenate([primary_boot, sensitivity_boot]),
            "analysis": ["primary_academic_harm_binary"] * len(primary_boot)
            + ["sensitivity_harm_index"] * len(sensitivity_boot),
        }
    )
    distribution_path = tables_dir / "rq3_bootstrap_distribution.csv"
    distribution_df.to_csv(distribution_path, index=False)

    rq3_figure_path = figures_dir / "rq3_mediation_bootstrap.png"
    plt.figure(figsize=(9, 5))
    for label, subset in distribution_df.groupby("analysis"):
        plt.hist(
            subset["bootstrap_indirect_effect"],
            bins=40,
            alpha=0.45,
            density=True,
            label=label,
            edgecolor="white",
        )
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.legend()
    plt.title("RQ3: Sleep Mediation Bootstrap Distribution")
    plt.tight_layout()
    plt.savefig(rq3_figure_path, dpi=140)
    plt.close()

    return {
        "rq3_mediation_table": str(mediation_path),
        "rq3_bootstrap_distribution_table": str(distribution_path),
        "rq3_figure": str(rq3_figure_path),
        "rq3_primary_indirect_effect": primary["indirect_effect_ab"],
    }


def _extract_gender_slopes(model: Any, genders: list[str]) -> pd.DataFrame:
    params = model.params
    cov = model.cov_params()
    base_slope = float(params.get("Addicted_Score", np.nan))

    rows = []
    base_gender = genders[0]
    base_var = float(cov.loc["Addicted_Score", "Addicted_Score"])

    rows.append(
        {
            "Gender": base_gender,
            "Addicted_Score_Slope": base_slope,
            "Std_Error": np.sqrt(max(base_var, 0.0)),
        }
    )

    for gender in genders[1:]:
        interaction_term = f"Addicted_Score:C(Gender)[T.{gender}]"
        interaction = float(params.get(interaction_term, 0.0))
        slope = base_slope + interaction

        if interaction_term in cov.index:
            var = (
                cov.loc["Addicted_Score", "Addicted_Score"]
                + cov.loc[interaction_term, interaction_term]
                + 2.0 * cov.loc["Addicted_Score", interaction_term]
            )
        else:
            var = base_var
        rows.append(
            {
                "Gender": gender,
                "Addicted_Score_Slope": slope,
                "Std_Error": float(np.sqrt(max(var, 0.0))),
            }
        )

    slopes_df = pd.DataFrame(rows)
    slopes_df["CI_Lower_95"] = slopes_df["Addicted_Score_Slope"] - 1.96 * slopes_df["Std_Error"]
    slopes_df["CI_Upper_95"] = slopes_df["Addicted_Score_Slope"] + 1.96 * slopes_df["Std_Error"]
    return slopes_df


def _fit_rq4_gender(df: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> dict[str, Any]:
    formula = (
        "Harm_Index ~ Addicted_Score * C(Gender) + Avg_Daily_Usage_Hours + "
        "Age + C(Academic_Level) + C(Country_Grouped)"
    )
    model = smf.ols(formula, data=df).fit()

    coef_df = _coef_table(model, model_name="RQ4_Gender_Asymmetry")
    rq4_coef_path = tables_dir / "rq4_model_coefficients.csv"
    coef_df.to_csv(rq4_coef_path, index=False)

    genders = sorted(df["Gender"].dropna().unique().tolist())
    slopes_df = _extract_gender_slopes(model, genders=genders)
    slopes_path = tables_dir / "rq4_gender_slopes.csv"
    slopes_df.to_csv(slopes_path, index=False)

    addiction_grid = np.linspace(df["Addicted_Score"].min(), df["Addicted_Score"].max(), 60)
    reference_usage = float(df["Avg_Daily_Usage_Hours"].mean())
    reference_age = float(df["Age"].mean())
    reference_level = df["Academic_Level"].mode().iloc[0]
    reference_country = df["Country_Grouped"].mode().iloc[0]

    pred_frames: list[pd.DataFrame] = []
    for gender in genders:
        grid = pd.DataFrame(
            {
                "Addicted_Score": addiction_grid,
                "Gender": gender,
                "Avg_Daily_Usage_Hours": reference_usage,
                "Age": reference_age,
                "Academic_Level": reference_level,
                "Country_Grouped": reference_country,
            }
        )
        grid["Predicted_Harm_Index"] = model.predict(grid)
        pred_frames.append(grid)
    prediction_df = pd.concat(pred_frames, ignore_index=True)
    pred_path = tables_dir / "rq4_gender_predictions.csv"
    prediction_df.to_csv(pred_path, index=False)

    quantiles = df["Addicted_Score"].quantile([0.25, 0.5, 0.75]).tolist()
    contrast_rows = []
    for q in quantiles:
        for gender in genders:
            point = pd.DataFrame(
                {
                    "Addicted_Score": [q],
                    "Gender": [gender],
                    "Avg_Daily_Usage_Hours": [reference_usage],
                    "Age": [reference_age],
                    "Academic_Level": [reference_level],
                    "Country_Grouped": [reference_country],
                }
            )
            contrast_rows.append(
                {
                    "Addicted_Score_Point": q,
                    "Gender": gender,
                    "Predicted_Harm_Index": float(model.predict(point).iloc[0]),
                }
            )
    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_path = tables_dir / "rq4_gender_contrasts.csv"
    contrasts_df.to_csv(contrasts_path, index=False)

    rq4_figure_path = figures_dir / "rq4_addiction_by_gender.png"
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=prediction_df, x="Addicted_Score", y="Predicted_Harm_Index", hue="Gender", linewidth=2.2)
    plt.title("RQ4: Gender Asymmetry at Equal Addiction Score")
    plt.tight_layout()
    plt.savefig(rq4_figure_path, dpi=140)
    plt.close()

    return {
        "rq4_model_coefficients": str(rq4_coef_path),
        "rq4_gender_slopes": str(slopes_path),
        "rq4_gender_predictions": str(pred_path),
        "rq4_gender_contrasts": str(contrasts_path),
        "rq4_figure": str(rq4_figure_path),
    }


def run_rq_analysis(
    df: pd.DataFrame,
    tables_dir: str | Path,
    figures_dir: str | Path,
    bootstrap_iters: int = 2000,
    random_seed: int = 42,
) -> dict[str, Any]:
    tables_path = Path(tables_dir)
    figures_path = Path(figures_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    rq1 = _fit_rq1_models(df=df, tables_dir=tables_path, figures_dir=figures_path)
    rq2 = _fit_rq2_model(df=df, tables_dir=tables_path, figures_dir=figures_path)
    rq3 = _fit_rq3_mediation(
        df=df,
        tables_dir=tables_path,
        figures_dir=figures_path,
        boot_iters=bootstrap_iters,
        seed=random_seed,
    )
    rq4 = _fit_rq4_gender(df=df, tables_dir=tables_path, figures_dir=figures_path)

    return {
        "rq1": rq1,
        "rq2": rq2,
        "rq3": rq3,
        "rq4": rq4,
    }
