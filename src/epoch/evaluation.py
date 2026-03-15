from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from config import MODEL_TEST_SIZE, RANDOM_SEED


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    base = df.copy()
    y = base["Harm_Index"].astype(float)

    X = base.drop(
        columns=[
            "Harm_Index",
            "Harm_Index_100",
            "Persona",
            "Cluster_ID",
            "Mental_Health_Harm_Component",
            "Sleep_Harm_Component",
            "Conflict_Harm_Component",
            "Addiction_Harm_Component",
            "Mental_Health_Score",
            "Sleep_Hours_Per_Night",
            "Conflicts_Over_Social_Media",
            "Addicted_Score",
            "Student_ID",
        ],
        errors="ignore",
    )
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    return X, y


def _fit_xgb_or_fallback(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> tuple[Any, str]:
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        return model, "xgboost"
    except Exception:
        fallback = GradientBoostingRegressor(random_state=seed)
        fallback.fit(X_train, y_train)
        return fallback, "gradient_boosting_fallback"


def _compute_group_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genders: pd.Series,
    academic_levels: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "Gender": genders.values,
            "Academic_Level": academic_levels.values,
        }
    )
    frame["Group"] = frame["Gender"].astype(str) + " | " + frame["Academic_Level"].astype(str)
    rows = []
    for group_name, group_df in frame.groupby("Group", sort=True):
        rows.append(
            {
                "Group": group_name,
                "Count": int(len(group_df)),
                "MAE": float(mean_absolute_error(group_df["y_true"], group_df["y_pred"])),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)


def run_model_eval(
    df: pd.DataFrame,
    tables_dir: str | Path,
    models_dir: str | Path,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    tables_path = Path(tables_dir)
    models_path = Path(models_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    X, y = _prepare_features(df)
    groups = df.loc[X.index, ["Gender", "Academic_Level"]].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=MODEL_TEST_SIZE,
        random_state=random_seed,
    )
    group_test = groups.loc[X_test.index]

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_rmse = float(np.sqrt(mean_squared_error(y_test, linear_pred)))

    xgb_model, xgb_model_name = _fit_xgb_or_fallback(X_train, y_train, seed=random_seed)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))

    if xgb_rmse <= linear_rmse:
        best_pred = xgb_pred
        best_model_name = xgb_model_name
    else:
        best_pred = linear_pred
        best_model_name = "linear_regression"

    group_mae = _compute_group_mae(
        y_true=y_test.to_numpy(),
        y_pred=np.asarray(best_pred, dtype=float),
        genders=group_test["Gender"],
        academic_levels=group_test["Academic_Level"],
    )

    rmse_table = pd.DataFrame(
        [
            {"model": "linear_regression", "rmse": linear_rmse},
            {"model": xgb_model_name, "rmse": xgb_rmse},
        ]
    )
    rmse_path = tables_path / "evaluation_rmse.csv"
    group_mae_path = tables_path / "evaluation_group_mae.csv"
    rmse_table.to_csv(rmse_path, index=False)
    group_mae.to_csv(group_mae_path, index=False)

    linear_model_path = models_path / "linear_regression_harm_index.pkl"
    xgb_model_path = models_path / "xgb_harm_index.pkl"
    with linear_model_path.open("wb") as file:
        pickle.dump(linear_model, file)
    with xgb_model_path.open("wb") as file:
        pickle.dump(xgb_model, file)

    return {
        "rmse_table": str(rmse_path),
        "group_mae_table": str(group_mae_path),
        "linear_rmse": linear_rmse,
        "xgb_rmse": xgb_rmse,
        "xgb_model_name": xgb_model_name,
        "best_model": best_model_name,
        "group_mae_df": group_mae,
        "model_paths": {
            "linear": str(linear_model_path),
            "xgb": str(xgb_model_path),
        },
    }
