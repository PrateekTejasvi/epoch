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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import HARM_FEATURES


HARM_COMPONENT_COLUMNS = [
    "Mental_Health_Harm_Component",
    "Sleep_Harm_Component",
    "Conflict_Harm_Component",
    "Addiction_Harm_Component",
]


def compute_harm_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    required = HARM_FEATURES
    missing = [feature for feature in required if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for harm index: {missing}")

    result = df.copy()
    result["Mental_Health_Harm_Component"] = 10 - result["Mental_Health_Score"]
    result["Sleep_Harm_Component"] = 8 - result["Sleep_Hours_Per_Night"]
    result["Conflict_Harm_Component"] = result["Conflicts_Over_Social_Media"]
    result["Addiction_Harm_Component"] = result["Addicted_Score"]

    component_matrix = result[HARM_COMPONENT_COLUMNS].astype(float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(component_matrix)

    pca = PCA(n_components=1, random_state=0)
    pca.fit(scaled)
    raw_loadings = np.abs(pca.components_[0])
    weights = raw_loadings / raw_loadings.sum()

    harm_index = np.dot(scaled, weights)
    result["Harm_Index"] = harm_index
    min_harm, max_harm = result["Harm_Index"].min(), result["Harm_Index"].max()
    if max_harm - min_harm > 1e-9:
        result["Harm_Index_100"] = (result["Harm_Index"] - min_harm) / (max_harm - min_harm) * 100.0
    else:
        result["Harm_Index_100"] = 50.0

    weight_dict = {
        "w1_mental_health_harm": float(weights[0]),
        "w2_sleep_harm": float(weights[1]),
        "w3_conflict_harm": float(weights[2]),
        "w4_addiction_harm": float(weights[3]),
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }
    return result, weight_dict


def save_harm_outputs(
    df: pd.DataFrame,
    weights: dict[str, float],
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, str]:
    tables_path = Path(tables_dir)
    figures_path = Path(figures_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    weights_table_path = tables_path / "harm_weights.csv"
    pd.DataFrame(
        [{"component": key, "value": value} for key, value in weights.items()]
    ).to_csv(weights_table_path, index=False)

    distribution_path = figures_path / "harm_index_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.hist(df["Harm_Index"], bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
    plt.title("Harm Index Distribution")
    plt.xlabel("Harm Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=140)
    plt.close()

    return {
        "harm_weights_table": str(weights_table_path),
        "harm_distribution_figure": str(distribution_path),
    }
