from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import COUNTRY_MIN_COUNT, REQUIRED_COLUMNS, TARGET
from .utils import ensure_dirs


def load_raw_dataset(input_csv: str | Path) -> pd.DataFrame:
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].fillna(result[column].mean())
        else:
            mode = result[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            result[column] = result[column].fillna(fill_value)
    return result


def _group_rare_countries(df: pd.DataFrame, min_count: int = COUNTRY_MIN_COUNT) -> pd.DataFrame:
    result = df.copy()
    country_counts = result["Country"].value_counts()
    frequent_countries = set(country_counts[country_counts >= min_count].index.tolist())
    result["Country_Grouped"] = result["Country"].apply(
        lambda value: value if value in frequent_countries else "Other"
    )
    return result


def create_model_matrix(clean_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = clean_df.drop(columns=["Student_ID"])
    target = feature_df[TARGET].copy()
    feature_df = feature_df.drop(columns=[TARGET])

    encoded = pd.get_dummies(feature_df, drop_first=True)
    numeric_cols = encoded.select_dtypes(include=[np.number, bool]).columns.tolist()
    encoded[numeric_cols] = encoded[numeric_cols].astype(float)

    scaler = StandardScaler()
    encoded[numeric_cols] = scaler.fit_transform(encoded[numeric_cols])

    model_matrix = encoded.copy()
    model_matrix.insert(0, "Student_ID", clean_df["Student_ID"].values)
    model_matrix[TARGET] = target.values
    return model_matrix


def run_preprocessing(input_csv: str | Path, out_dir: str | Path) -> dict[str, str]:
    validate_path = Path(input_csv)
    output_dir = Path(out_dir)
    ensure_dirs(output_dir)

    df = load_raw_dataset(validate_path)
    validate_required_columns(df)

    # Keep row-level alignment stable by cleaning before any extraction.
    clean_df = df.drop_duplicates().reset_index(drop=True)
    clean_df = _impute_missing(clean_df)
    clean_df = _group_rare_countries(clean_df)
    clean_df["Academic_Harm_Binary"] = (
        clean_df["Affects_Academic_Performance"].astype(str).str.lower() == "yes"
    ).astype(int)

    analysis_path = output_dir / "analysis_base.csv"
    model_matrix_path = output_dir / "model_matrix.csv"

    clean_df.to_csv(analysis_path, index=False)
    model_matrix = create_model_matrix(clean_df)
    model_matrix.to_csv(model_matrix_path, index=False)

    metadata_path = output_dir / "preprocessing_metadata.json"
    metadata: dict[str, Any] = {
        "input_csv": str(validate_path),
        "rows_raw": int(len(df)),
        "rows_clean": int(len(clean_df)),
        "columns_clean": clean_df.columns.tolist(),
        "analysis_base": str(analysis_path),
        "model_matrix": str(model_matrix_path),
    }
    pd.Series(metadata).to_json(metadata_path, indent=2)

    return {
        "analysis_base": str(analysis_path),
        "model_matrix": str(model_matrix_path),
        "metadata": str(metadata_path),
    }
