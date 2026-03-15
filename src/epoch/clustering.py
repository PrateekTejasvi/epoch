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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_FEATURES, SILHOUETTE_K_RANGE


PERSONAS = [
    "The Doomscroller",
    "The Social Warrior",
    "The Quiet Addict",
    "The Balanced User",
]


def _persona_targets() -> dict[str, np.ndarray]:
    return {
        "The Doomscroller": np.array([1.2, 1.2, 1.2, 0.5, 1.0]),
        "The Social Warrior": np.array([0.3, 0.4, 0.4, 1.6, 0.2]),
        "The Quiet Addict": np.array([-1.2, 0.5, 0.6, 0.2, 1.4]),
        "The Balanced User": np.array([1.0, -1.3, -1.3, -0.8, -0.6]),
    }


def evaluate_silhouette(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    k_values: range = SILHOUETTE_K_RANGE,
    random_seed: int = 42,
) -> pd.DataFrame:
    features = feature_cols or CLUSTER_FEATURES
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].astype(float))

    rows: list[dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": float(k), "silhouette_score": float(score)})
    return pd.DataFrame(rows)


def _cluster_to_harm_space(centroids_scaled: np.ndarray) -> np.ndarray:
    usage = centroids_scaled[:, 0]
    sleep_harm = -centroids_scaled[:, 1]
    mental_harm = -centroids_scaled[:, 2]
    conflict_harm = centroids_scaled[:, 3]
    addiction_harm = centroids_scaled[:, 4]
    return np.column_stack([usage, sleep_harm, mental_harm, conflict_harm, addiction_harm])


def _assign_persona_labels(centroids_scaled: np.ndarray) -> dict[int, str]:
    target_dict = _persona_targets()
    target_matrix = np.vstack([target_dict[persona] for persona in PERSONAS])
    cluster_harm = _cluster_to_harm_space(centroids_scaled)
    cost = cdist(cluster_harm, target_matrix, metric="euclidean")
    row_idx, col_idx = linear_sum_assignment(cost)
    return {int(cluster): PERSONAS[int(persona)] for cluster, persona in zip(row_idx, col_idx)}


def run_clustering(
    df: pd.DataFrame,
    k: int = 4,
    random_seed: int = 42,
) -> dict[str, Any]:
    missing = [column for column in CLUSTER_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")

    result = df.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(result[CLUSTER_FEATURES].astype(float))

    silhouette_df = evaluate_silhouette(result, feature_cols=CLUSTER_FEATURES, random_seed=random_seed)
    model = KMeans(n_clusters=k, random_state=random_seed, n_init=30)
    raw_labels = model.fit_predict(X)
    cluster_to_persona = _assign_persona_labels(model.cluster_centers_)

    result["Cluster_ID"] = raw_labels.astype(int)
    result["Persona"] = result["Cluster_ID"].map(cluster_to_persona)

    centroids_original = scaler.inverse_transform(model.cluster_centers_)
    centroids_table = pd.DataFrame(centroids_original, columns=CLUSTER_FEATURES)
    centroids_table.insert(0, "Cluster_ID", np.arange(k))
    centroids_table["Persona"] = centroids_table["Cluster_ID"].map(cluster_to_persona)

    persona_distribution = (
        result["Persona"].value_counts(normalize=False).rename_axis("Persona").reset_index(name="Count")
    )
    persona_distribution["Share"] = persona_distribution["Count"] / persona_distribution["Count"].sum()

    return {
        "data": result,
        "kmeans_model": model,
        "scaler": scaler,
        "silhouette_table": silhouette_df,
        "centroids_table": centroids_table,
        "persona_distribution": persona_distribution,
        "cluster_to_persona": cluster_to_persona,
    }


def save_clustering_outputs(
    clustering_results: dict[str, Any],
    tables_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, str]:
    tables_path = Path(tables_dir)
    figures_path = Path(figures_dir)
    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    silhouette_path = tables_path / "silhouette_by_k.csv"
    centroids_path = tables_path / "persona_centroids.csv"
    persona_dist_path = tables_path / "persona_distribution.csv"
    cluster_map_path = tables_path / "cluster_persona_mapping.csv"

    clustering_results["silhouette_table"].to_csv(silhouette_path, index=False)
    clustering_results["centroids_table"].to_csv(centroids_path, index=False)
    clustering_results["persona_distribution"].to_csv(persona_dist_path, index=False)
    pd.DataFrame(
        [
            {"Cluster_ID": cluster_id, "Persona": persona}
            for cluster_id, persona in clustering_results["cluster_to_persona"].items()
        ]
    ).to_csv(cluster_map_path, index=False)

    silhouette_fig_path = figures_path / "silhouette_curve.png"
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=clustering_results["silhouette_table"], x="k", y="silhouette_score", marker="o")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(silhouette_fig_path, dpi=140)
    plt.close()

    scatter_fig_path = figures_path / "persona_cluster_scatter.png"
    enriched = clustering_results["data"]
    pca = PCA(n_components=2, random_state=42)
    X = StandardScaler().fit_transform(enriched[CLUSTER_FEATURES].astype(float))
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Persona": enriched["Persona"]})

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Persona", alpha=0.8)
    plt.title("Persona Clusters (PCA Projection)")
    plt.tight_layout()
    plt.savefig(scatter_fig_path, dpi=140)
    plt.close()

    return {
        "silhouette_table": str(silhouette_path),
        "centroids_table": str(centroids_path),
        "persona_distribution_table": str(persona_dist_path),
        "cluster_persona_mapping_table": str(cluster_map_path),
        "silhouette_figure": str(silhouette_fig_path),
        "persona_scatter_figure": str(scatter_fig_path),
    }
