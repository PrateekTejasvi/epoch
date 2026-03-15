from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.epoch.run_pipeline import run_pipeline  # noqa: E402
from src.epoch_ui.model_service import (  # noqa: E402
    PERSONA_DESCRIPTIONS,
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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(1200px 600px at 10% 10%, #2c2c2c 0%, #1c1c1c 45%, #141414 100%);
            color: #f0f0f0;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 14px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .persona-chip {
            display: inline-block;
            padding: 8px 14px;
            border: 2px solid #ffad33;
            color: #ffca6a;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
        }
        .small-muted {
            color: #c7c7c7;
            font-size: 0.95rem;
        }
        .recommend-card {
            border-left: 4px solid #ffad33;
            background: rgba(255, 173, 51, 0.08);
            border-radius: 8px;
            padding: 12px;
            margin-top: 6px;
        }
        .platform-card {
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 10px;
            text-align: center;
            min-height: 120px;
            background: rgba(255,255,255,0.03);
        }
        .platform-selected {
            border: 2px solid #8f7bff;
            background: rgba(143,123,255,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_artifacts() -> object:
    return load_dashboard_artifacts(ROOT_DIR / "outputs")


def _platform_card(platform: str, score: float, selected: bool) -> str:
    cls = "platform-card platform-selected" if selected else "platform-card"
    return (
        f"<div class='{cls}'>"
        f"<div style='font-weight:700;font-size:1.15rem'>{platform}</div>"
        f"<div style='font-size:1.9rem;color:#ff7043;font-weight:700'>{score:.1f}</div>"
        f"<div class='small-muted'>harm score</div>"
        f"</div>"
    )


def main() -> None:
    st.set_page_config(page_title="Student Harm Live Dashboard", page_icon="📊", layout="wide")
    _inject_styles()

    try:
        artifacts = _load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python3 -m epoch.run_pipeline")
        if st.button("Generate missing outputs now"):
            with st.spinner("Running pipeline to generate outputs..."):
                run_pipeline()
            st.success("Outputs generated. Reloading dashboard...")
            st.rerun()
        return

    bounds = slider_bounds(artifacts)
    pearson_corr, spearman_corr = compute_correlations(artifacts)

    st.title("Scroll Until It Hurts: Live Student Harm Diagnosis")

    student_col, diag_col = st.columns([1.02, 1.0], gap="large")

    with student_col:
        st.subheader("Student Profile")
        gender = st.radio("Gender", options=["Male", "Female"], horizontal=True, index=0)
        level = st.radio("Academic level", options=["High School", "Undergraduate", "Graduate"], horizontal=True, index=1)
        selected_platform = st.selectbox(
            "Platform in use",
            options=artifacts.platform_harm_df["Most_Used_Platform"].tolist(),
            index=1 if len(artifacts.platform_harm_df) > 1 else 0,
        )
        age = st.slider("Age", int(bounds["Age"][0]), int(bounds["Age"][1]), int(bounds["Age"][2]), step=1)
        usage = st.slider(
            "Daily usage (hours)",
            float(bounds["Avg_Daily_Usage_Hours"][0]),
            float(bounds["Avg_Daily_Usage_Hours"][1]),
            float(bounds["Avg_Daily_Usage_Hours"][2]),
            step=0.1,
        )
        sleep = st.slider(
            "Sleep (hours/night)",
            float(bounds["Sleep_Hours_Per_Night"][0]),
            float(bounds["Sleep_Hours_Per_Night"][1]),
            float(bounds["Sleep_Hours_Per_Night"][2]),
            step=0.1,
        )
        mental = st.slider(
            "Mental health score",
            int(bounds["Mental_Health_Score"][0]),
            int(bounds["Mental_Health_Score"][1]),
            int(bounds["Mental_Health_Score"][2]),
            step=1,
        )
        conflicts = st.slider(
            "Conflicts (count)",
            int(bounds["Conflicts_Over_Social_Media"][0]),
            int(bounds["Conflicts_Over_Social_Media"][1]),
            int(bounds["Conflicts_Over_Social_Media"][2]),
            step=1,
        )
        addiction = st.slider(
            "Addiction score",
            int(bounds["Addicted_Score"][0]),
            int(bounds["Addicted_Score"][1]),
            int(bounds["Addicted_Score"][2]),
            step=1,
        )

    profile = {
        "Age": age,
        "Gender": gender,
        "Academic_Level": level,
        "Avg_Daily_Usage_Hours": usage,
        "Sleep_Hours_Per_Night": sleep,
        "Mental_Health_Score": mental,
        "Conflicts_Over_Social_Media": conflicts,
        "Addicted_Score": addiction,
    }
    scores = compute_harm_scores(profile, artifacts)
    platform_estimate = estimate_selected_platform_harm(
        base_harm_index=scores["harm_index_base"],
        selected_platform=selected_platform,
        artifacts=artifacts,
    )
    persona = assign_persona(profile, artifacts)
    academic_impact = compute_academic_impact_pct(profile, persona, artifacts)
    composite_risk = float(
        np.mean(
            [
                scores["sleep_risk_pct"],
                scores["mental_risk_pct"],
                scores["conflict_risk_pct"],
                academic_impact,
                scores["harm_index_base_100"],
            ]
        )
    )
    risk_label, risk_text = recommendation_from_risk(composite_risk)
    platform_df = platform_comparison(selected_platform, scores["harm_index_base"], artifacts)

    top_metrics = st.columns(4)
    top_metrics[0].metric("Base Harm Index", f"{scores['harm_index_base_10']:.1f}/10")
    top_metrics[1].metric("Sleep risk", f"{scores['sleep_risk_pct']:.0f}%")
    top_metrics[2].metric("Mental health risk", f"{scores['mental_risk_pct']:.0f}%")
    top_metrics[3].metric("Conflict risk", f"{scores['conflict_risk_pct']:.0f}%")

    with diag_col:
        st.subheader("Live Diagnosis")
        st.markdown(f"<span class='persona-chip'>{persona}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{PERSONA_DESCRIPTIONS[persona]}</div>", unsafe_allow_html=True)
        st.write("")
        st.write(f"Sleep deprivation risk: **{scores['sleep_risk_pct']:.0f}%**")
        st.progress(int(round(scores["sleep_risk_pct"])))
        st.write(f"Mental health deterioration: **{scores['mental_risk_pct']:.0f}%**")
        st.progress(int(round(scores["mental_risk_pct"])))
        st.write(f"Relationship conflict: **{scores['conflict_risk_pct']:.0f}%**")
        st.progress(int(round(scores["conflict_risk_pct"])))
        st.write(f"Behavior-only overall risk: **{scores['harm_index_base_100']:.0f}%**")
        st.progress(int(round(scores["harm_index_base_100"])))
        st.write(f"If this same student used **{selected_platform}**: **{platform_estimate['harm_index_10']:.1f}/10**")
        st.progress(int(round(platform_estimate["harm_index_100"])))
        direction = "higher" if platform_estimate["platform_delta"] >= 0 else "lower"
        st.caption(
            f"Platform effect: **{abs(platform_estimate['platform_delta']):.2f}** harm-index points "
            f"{direction} than the all-platform average."
        )
        st.write(f"Academic impact: **{academic_impact:.0f}%**")
        st.progress(int(round(academic_impact)))
        st.markdown(
            f"<div class='recommend-card'><b>{risk_label}.</b> {risk_text}</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    left_bottom, right_bottom = st.columns([1.02, 1.0], gap="large")

    with left_bottom:
        st.subheader("Platform in Use")
        top_cards = platform_df.head(5)
        selected_rows = platform_df[platform_df["Selected"]]
        if (
            not selected_rows.empty
            and selected_rows.iloc[0]["Most_Used_Platform"] not in top_cards["Most_Used_Platform"].tolist()
        ):
            top_cards = pd.concat([top_cards.iloc[:4], selected_rows.iloc[[0]]], ignore_index=True)
            top_cards = top_cards.drop_duplicates(subset=["Most_Used_Platform"], keep="first")
        top_cards = top_cards.head(6)
        for start in (0, 3):
            cols = st.columns(3)
            chunk = top_cards.iloc[start : start + 3]
            for i, (_, row) in enumerate(chunk.iterrows()):
                cols[i].markdown(
                    _platform_card(
                        platform=row["Most_Used_Platform"],
                        score=float(row["Profile_Adjusted_Harm_10"]),
                        selected=bool(row["Selected"]),
                    ),
                    unsafe_allow_html=True,
                )

    with right_bottom:
        st.subheader("Platform Harm Comparison")
        st.caption("Counterfactual view: same student behavior, different platform.")
        chart_df = platform_df.head(8).sort_values("Profile_Adjusted_Harm_10", ascending=True)
        fig = px.bar(
            chart_df,
            x="Profile_Adjusted_Harm_10",
            y="Most_Used_Platform",
            orientation="h",
            color="Most_Used_Platform",
            text=chart_df["Profile_Adjusted_Harm_10"].round(1),
            height=360,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Adjusted Harm (0-10)",
            yaxis_title="",
            showlegend=False,
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_row = platform_df[platform_df["Most_Used_Platform"] == selected_platform].iloc[0]
        lowest_row = platform_df.iloc[-1]
        diff = float(selected_row["Profile_Adjusted_Harm_10"] - lowest_row["Profile_Adjusted_Harm_10"])
        st.caption(
            f"For this profile, **{selected_platform}** is estimated at "
            f"**{diff:.2f}** harm points above **{lowest_row['Most_Used_Platform']}**."
        )

    st.write("")
    with st.expander("Formula, model terms, and coefficients"):
        st.markdown(
            """
            **Harm components**
            - `Mental_Health_Harm_Component = 10 - Mental_Health_Score`
            - `Sleep_Harm_Component = 8 - Sleep_Hours_Per_Night`
            - `Conflict_Harm_Component = Conflicts_Over_Social_Media`
            - `Addiction_Harm_Component = Addicted_Score`

            **Composite harm index**
            - `Harm_Index = w1*z(mental_harm) + w2*z(sleep_harm) + w3*z(conflict_harm) + w4*z(addiction_harm)`
            - Weights are learned via PCA loadings.

            **Counterfactual platform comparison (correct logic)**
            - `Base_Harm = f(student_behavior)`
            - `Platform_Delta(p) = mean_harm(p) - global_mean_harm`
            - `Counterfactual_Harm(p) = Base_Harm + Platform_Delta(p)`
            - Platform selection does **not** change `Base_Harm`; it only selects which counterfactual row to highlight.
            """
        )
        st.dataframe(
            artifacts.eval_rmse_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Evaluation RMSE from the same trained pipeline artifacts.")
        st.dataframe(
            artifacts.rq4_gender_slopes_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("RQ4 gender asymmetry slopes from the regression stage.")
        st.dataframe(
            artifacts.rq3_mediation_df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption("RQ3 mediation estimates (direct, indirect, and CI).")

    with st.expander("Correlations used for interpretation (Pearson and Spearman)"):
        corr_col1, corr_col2 = st.columns(2)
        with corr_col1:
            st.markdown("**Pearson correlation matrix**")
            pearson_fig = px.imshow(
                pearson_corr.round(2),
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            pearson_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", height=430)
            st.plotly_chart(pearson_fig, use_container_width=True)
        with corr_col2:
            st.markdown("**Spearman correlation matrix**")
            spearman_fig = px.imshow(
                spearman_corr.round(2),
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            spearman_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", height=430)
            st.plotly_chart(spearman_fig, use_container_width=True)

        st.markdown("**Subgroup fairness (MAE by Gender x Academic Level)**")
        st.dataframe(artifacts.eval_group_mae_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
