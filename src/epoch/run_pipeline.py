from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    BOOTSTRAP_ITERS,
    CLUSTER_K,
    FIGURES_DIR,
    INPUT_CSV,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TABLES_DIR,
)
from .clustering import run_clustering, save_clustering_outputs
from .evaluation import run_model_eval
from .harm import compute_harm_index, save_harm_outputs
from .preprocessing import run_preprocessing
from .report import build_policy_report
from .research import run_rq_analysis
from .utils import ensure_dirs, write_json


def _resolve_runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {
        "input_csv": Path(INPUT_CSV),
        "output_dir": Path(OUTPUT_DIR),
        "processed_dir": Path(PROCESSED_DIR),
        "tables_dir": Path(TABLES_DIR),
        "figures_dir": Path(FIGURES_DIR),
        "models_dir": Path(MODELS_DIR),
        "reports_dir": Path(REPORTS_DIR),
        "cluster_k": int(CLUSTER_K),
        "bootstrap_iters": int(BOOTSTRAP_ITERS),
        "random_seed": int(RANDOM_SEED),
    }
    if overrides:
        for key, value in overrides.items():
            cfg[key] = Path(value) if key.endswith("_dir") or key == "input_csv" else value
    return cfg


def run_pipeline(runtime_config: dict[str, Any] | None = None) -> None:
    cfg = _resolve_runtime_config(runtime_config)
    ensure_dirs(
        cfg["output_dir"],
        cfg["processed_dir"],
        cfg["tables_dir"],
        cfg["figures_dir"],
        cfg["models_dir"],
        cfg["reports_dir"],
    )

    preprocess_paths = run_preprocessing(cfg["input_csv"], cfg["processed_dir"])
    analysis_df = pd.read_csv(preprocess_paths["analysis_base"])

    analysis_df, weights = compute_harm_index(analysis_df)
    harm_outputs = save_harm_outputs(
        df=analysis_df,
        weights=weights,
        tables_dir=cfg["tables_dir"],
        figures_dir=cfg["figures_dir"],
    )

    clustering_results = run_clustering(
        df=analysis_df,
        k=int(cfg["cluster_k"]),
        random_seed=int(cfg["random_seed"]),
    )
    analysis_df = clustering_results["data"]
    clustering_outputs = save_clustering_outputs(
        clustering_results=clustering_results,
        tables_dir=cfg["tables_dir"],
        figures_dir=cfg["figures_dir"],
    )

    enriched_path = cfg["processed_dir"] / "analysis_enriched.csv"
    analysis_df.to_csv(enriched_path, index=False)

    rq_results = run_rq_analysis(
        df=analysis_df,
        tables_dir=cfg["tables_dir"],
        figures_dir=cfg["figures_dir"],
        bootstrap_iters=int(cfg["bootstrap_iters"]),
        random_seed=int(cfg["random_seed"]),
    )
    evaluation_results = run_model_eval(
        df=analysis_df,
        tables_dir=cfg["tables_dir"],
        models_dir=cfg["models_dir"],
        random_seed=int(cfg["random_seed"]),
    )

    report_path = build_policy_report(
        results={
            "rq1_platform_means_df": rq_results["rq1"]["platform_means_df"],
            "persona_distribution_df": clustering_results["persona_distribution"],
            "rq3_primary_indirect_effect": rq_results["rq3"]["rq3_primary_indirect_effect"],
            "linear_rmse": evaluation_results["linear_rmse"],
            "xgb_rmse": evaluation_results["xgb_rmse"],
            "best_model": evaluation_results["best_model"],
        },
        report_path=cfg["reports_dir"] / "policy_recommendations.md",
    )

    summary = {
        "config": {key: str(value) for key, value in cfg.items()},
        "preprocessing": preprocess_paths,
        "harm": {"weights": weights, **harm_outputs},
        "clustering": {
            **clustering_outputs,
            "best_silhouette_k": int(
                clustering_results["silhouette_table"]
                .sort_values("silhouette_score", ascending=False)
                .iloc[0]["k"]
            ),
        },
        "rq_results": {
            "rq1": {k: v for k, v in rq_results["rq1"].items() if not k.endswith("_df")},
            "rq2": rq_results["rq2"],
            "rq3": rq_results["rq3"],
            "rq4": rq_results["rq4"],
        },
        "evaluation": {
            "rmse_table": evaluation_results["rmse_table"],
            "group_mae_table": evaluation_results["group_mae_table"],
            "linear_rmse": evaluation_results["linear_rmse"],
            "xgb_rmse": evaluation_results["xgb_rmse"],
            "best_model": evaluation_results["best_model"],
            "xgb_model_name": evaluation_results["xgb_model_name"],
            "model_paths": evaluation_results["model_paths"],
        },
        "outputs": {
            "analysis_enriched": str(enriched_path),
            "policy_report": report_path,
        },
    }
    write_json(summary, cfg["reports_dir"] / "pipeline_summary.json")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the student digital harm research pipeline.")
    parser.add_argument("--input-csv", type=str, default=str(INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--cluster-k", type=int, default=CLUSTER_K)
    parser.add_argument("--bootstrap-iters", type=int, default=BOOTSTRAP_ITERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    run_pipeline(
        runtime_config={
            "input_csv": args.input_csv,
            "output_dir": output_dir,
            "processed_dir": output_dir / "processed",
            "tables_dir": output_dir / "tables",
            "figures_dir": output_dir / "figures",
            "models_dir": output_dir / "models",
            "reports_dir": output_dir / "reports",
            "cluster_k": args.cluster_k,
            "bootstrap_iters": args.bootstrap_iters,
            "random_seed": args.seed,
        }
    )


if __name__ == "__main__":
    main()
