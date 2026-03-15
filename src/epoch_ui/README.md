# Student Harm Live Dashboard (Streamlit)

## Run
```bash
streamlit run src/epoch_ui/app.py
```

If that fails with `ModuleNotFoundError: streamlit`, run with the interpreter that has the package installed:
```bash
python3 -m streamlit run src/epoch_ui/app.py
```

Or use the helper launcher:
```bash
./scripts/run_ui.sh
```

If `outputs/` artifacts are missing, the app now shows the missing files and provides a button to run:
```bash
python3 -m epoch.run_pipeline
```

## Data Sources
- `outputs/processed/analysis_enriched.csv`
- `outputs/tables/harm_weights.csv`
- `outputs/tables/persona_centroids.csv`
- `outputs/tables/rq1_platform_mean_harm.csv`
- `outputs/tables/rq3_sleep_mediation.csv`
- `outputs/tables/rq4_gender_slopes.csv`
- `outputs/tables/evaluation_rmse.csv`
- `outputs/tables/evaluation_group_mae.csv`

## Core Math
- `mental_harm = 10 - Mental_Health_Score`
- `sleep_harm = 8 - Sleep_Hours_Per_Night`
- `conflict_harm = Conflicts_Over_Social_Media`
- `addiction_harm = Addicted_Score`
- `Harm_Index = w1*z(mental_harm) + w2*z(sleep_harm) + w3*z(conflict_harm) + w4*z(addiction_harm)`
- `Base_Harm = f(student_behavior)` (platform-independent)
- `Counterfactual_Harm(platform) = Base_Harm + Platform_Delta(platform)`

## UI Sections
- KPI row: Harm index + sleep/mental/conflict risk percentages.
- Student profile controls: gender, level, platform, and behavioral sliders.
- Live diagnosis: persona assignment and risk bars.
- Platform cards and platform harm comparison chart.
- Explainability expanders: formulas, model tables, Pearson/Spearman correlation matrices, subgroup MAE.
