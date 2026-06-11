from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.viability.config import ViabilityConfig


def run_output_dir(config: ViabilityConfig) -> Path:
    return Path(config.run.output_dir) / "runs" / config.run.name


def write_config_resolved(config: ViabilityConfig, directory: str | Path) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    return config.dump_resolved_config(output_dir / "config_resolved.yaml")


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    prefer_parquet: bool = True,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefer_parquet:
        try:
            df.to_parquet(output_path, index=False)
            return output_path
        except Exception as exc:
            warnings.warn(
                f"Could not write parquet to {output_path}: {exc}; falling back to CSV",
                stacklevel=2,
            )

    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def write_evaluation_batch(
    df: pd.DataFrame,
    directory: str | Path,
    batch_index: int,
    *,
    prefer_parquet: bool = True,
) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".parquet" if prefer_parquet else ".csv"
    batch_path = output_dir / f"evaluations_batch_{batch_index:04d}{suffix}"
    return write_table(df, batch_path, prefer_parquet=prefer_parquet)
