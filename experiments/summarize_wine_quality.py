"""Agregación de los experimentos híbridos de Wine Quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

INPUT_PATTERN = (
    "wine_quality_hybrid_split_42_seed_*.json"
)

EXPECTED_SEEDS = {
    42,
    123,
    456,
    789,
    2026,
}

GLOBAL_METRICS = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)

CLASS_NAMES = (
    "quality_5",
    "quality_6",
    "quality_7",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga el resumen JSON de una ejecución."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        return json.load(input_file)


def extract_run_data(
    summary: dict[str, Any],
) -> dict[str, float | int | bool]:
    """Extrae los resultados relevantes de una ejecución."""
    seeds = summary["random_seeds"]
    training_result = summary["training_result"]
    execution_times = summary["execution_times_seconds"]
    test_results = summary["test_results"]

    global_metrics = test_results["metrics"]
    per_class_metrics = test_results["per_class_metrics"]

    confusion = np.asarray(
        test_results["confusion_matrix"],
        dtype=np.int64,
    )

    if confusion.shape != (3, 3):
        raise ValueError(
            "La matriz de confusión de Wine debe tener forma (3, 3)."
        )

    row: dict[str, float | int | bool] = {
        "split_seed": int(
            seeds["dataset_split"]
        ),
        "training_seed": int(
            seeds["model_and_training"]
        ),
        "epochs_completed": int(
            training_result["epochs_completed"]
        ),
        "best_epoch": int(
            training_result["best_epoch"]
        ),
        "best_validation_loss": float(
            training_result["best_validation_loss"]
        ),
        "stopped_early": bool(
            training_result["stopped_early"]
        ),
        "training_time_seconds": float(
            execution_times["training"]
        ),
        "evaluation_time_seconds": float(
            execution_times["evaluation"]
        ),
        "accuracy": float(
            global_metrics["accuracy"]
        ),
        "precision_macro": float(
            global_metrics["precision_macro"]
        ),
        "recall_macro": float(
            global_metrics["recall_macro"]
        ),
        "f1_macro": float(
            global_metrics["f1_macro"]
        ),
    }

    for class_name in CLASS_NAMES:
        class_metrics = per_class_metrics[class_name]

        row[f"{class_name}_precision"] = float(
            class_metrics["precision"]
        )

        row[f"{class_name}_recall"] = float(
            class_metrics["recall"]
        )

        row[f"{class_name}_f1"] = float(
            class_metrics["f1"]
        )

        row[f"{class_name}_support"] = int(
            class_metrics["support"]
        )

    for true_index in range(3):
        for predicted_index in range(3):
            row[
                f"confusion_{true_index}_{predicted_index}"
            ] = int(
                confusion[
                    true_index,
                    predicted_index,
                ]
            )

    return row


def summarize_columns(
    runs_table: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Calcula media, desviación típica, mínimo y máximo."""
    rows: list[dict[str, float | str]] = []

    for column in columns:
        values = runs_table[column].astype(float)

        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "standard_deviation": float(
                    values.std(ddof=1)
                ),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def build_confusion_matrices(
    runs_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula las matrices de confusión sumada y media."""
    summed_matrix = np.zeros(
        shape=(3, 3),
        dtype=np.int64,
    )

    mean_matrix = np.zeros(
        shape=(3, 3),
        dtype=np.float64,
    )

    for true_index in range(3):
        for predicted_index in range(3):
            column_name = (
                f"confusion_{true_index}_{predicted_index}"
            )

            summed_matrix[
                true_index,
                predicted_index,
            ] = int(
                runs_table[column_name].sum()
            )

            mean_matrix[
                true_index,
                predicted_index,
            ] = float(
                runs_table[column_name].mean()
            )

    return summed_matrix, mean_matrix


def main() -> None:
    """Genera el resumen agregado de Wine Quality."""
    input_paths = sorted(
        METRICS_DIRECTORY.glob(INPUT_PATTERN)
    )

    if len(input_paths) != len(EXPECTED_SEEDS):
        raise RuntimeError(
            "Se esperaban exactamente cinco ejecuciones, "
            f"pero se encontraron {len(input_paths)}."
        )

    summaries = [
        load_json(path)
        for path in input_paths
    ]

    run_rows = [
        extract_run_data(summary)
        for summary in summaries
    ]

    runs_table = (
        pd.DataFrame(run_rows)
        .sort_values("training_seed")
        .reset_index(drop=True)
    )

    detected_seeds = set(
        runs_table["training_seed"].astype(int)
    )

    if detected_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Las semillas encontradas no coinciden con las esperadas. "
            f"Encontradas: {sorted(detected_seeds)}."
        )

    if runs_table["split_seed"].nunique() != 1:
        raise RuntimeError(
            "Las ejecuciones no utilizan la misma semilla de partición."
        )

    metric_columns = [
        *GLOBAL_METRICS,
        "best_validation_loss",
        "best_epoch",
        "epochs_completed",
        "training_time_seconds",
    ]

    for class_name in CLASS_NAMES:
        metric_columns.extend(
            [
                f"{class_name}_precision",
                f"{class_name}_recall",
                f"{class_name}_f1",
            ]
        )

    summary_table = summarize_columns(
        runs_table,
        metric_columns,
    )

    summed_confusion, mean_confusion = (
        build_confusion_matrices(
            runs_table
        )
    )

    runs_output_path = (
        METRICS_DIRECTORY
        / "wine_quality_hybrid_runs.csv"
    )

    summary_output_path = (
        METRICS_DIRECTORY
        / "wine_quality_hybrid_summary.csv"
    )

    json_output_path = (
        METRICS_DIRECTORY
        / "wine_quality_hybrid_summary.json"
    )

    runs_table.to_csv(
        runs_output_path,
        index=False,
    )

    summary_table.to_csv(
        summary_output_path,
        index=False,
    )

    metric_summary = {
        str(row["metric"]): {
            "mean": float(row["mean"]),
            "standard_deviation": float(
                row["standard_deviation"]
            ),
            "minimum": float(row["minimum"]),
            "maximum": float(row["maximum"]),
        }
        for _, row in summary_table.iterrows()
    }

    aggregate_summary = {
        "dataset": "Wine Quality",
        "model": "hybrid quantum-classical classifier",
        "number_of_runs": int(
            len(runs_table)
        ),
        "split_seed": int(
            runs_table["split_seed"].iloc[0]
        ),
        "training_seeds": [
            int(seed)
            for seed in runs_table[
                "training_seed"
            ].tolist()
        ],
        "classes": {
            "0": 5,
            "1": 6,
            "2": 7,
        },
        "metrics": metric_summary,
        "summed_confusion_matrix": (
            summed_confusion.tolist()
        ),
        "mean_confusion_matrix": (
            mean_confusion.tolist()
        ),
    }

    with json_output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            aggregate_summary,
            output_file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print("WINE QUALITY — INDIVIDUAL RUNS")
    print("=" * 90)

    display_columns = [
        "training_seed",
        "best_epoch",
        "best_validation_loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "training_time_seconds",
    ]

    print(
        runs_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 90)
    print("AGGREGATED GLOBAL RESULTS")
    print("=" * 90)

    global_summary = summary_table[
        summary_table["metric"].isin(
            [
                *GLOBAL_METRICS,
                "best_validation_loss",
                "best_epoch",
                "training_time_seconds",
            ]
        )
    ]

    print(
        global_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n" + "=" * 90)
    print("AGGREGATED PER-CLASS RESULTS")
    print("=" * 90)

    per_class_summary = summary_table[
        summary_table["metric"].str.startswith(
            CLASS_NAMES
        )
    ]

    print(
        per_class_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSummed confusion matrix:")
    print(summed_confusion)

    print("\nMean confusion matrix:")
    print(
        np.round(
            mean_confusion,
            decimals=2,
        )
    )

    print("\nGenerated files:")
    print(f"  {runs_output_path}")
    print(f"  {summary_output_path}")
    print(f"  {json_output_path}")


if __name__ == "__main__":
    main()