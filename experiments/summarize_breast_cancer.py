"""Agregación de los experimentos híbridos de Breast Cancer."""

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
    "breast_cancer_hybrid_split_42_seed_*.json"
)

EXPECTED_NUMBER_OF_RUNS = 5

METRICS_TO_SUMMARIZE = (
    "accuracy",
    "precision_malignant",
    "recall_malignant",
    "f1_malignant",
    "specificity_benign",
    "best_validation_loss",
    "best_epoch",
    "training_time_seconds",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga un archivo JSON experimental."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        return json.load(input_file)


def extract_run_data(
    summary: dict[str, Any],
) -> dict[str, float | int | bool]:
    """Extrae los valores relevantes de una ejecución."""
    metrics = summary["test_results"]["metrics"]
    training_result = summary["training_result"]
    execution_times = summary["execution_times_seconds"]
    seeds = summary["random_seeds"]

    confusion_matrix = np.asarray(
        summary["test_results"]["confusion_matrix"],
        dtype=np.int64,
    )

    if confusion_matrix.shape != (2, 2):
        raise ValueError(
            "La matriz de confusión debe tener forma (2, 2)."
        )

    true_negatives = int(confusion_matrix[0, 0])
    false_positives = int(confusion_matrix[0, 1])
    false_negatives = int(confusion_matrix[1, 0])
    true_positives = int(confusion_matrix[1, 1])

    return {
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
        "accuracy": float(
            metrics["accuracy"]
        ),
        "precision_malignant": float(
            metrics["precision_malignant"]
        ),
        "recall_malignant": float(
            metrics["recall_malignant"]
        ),
        "f1_malignant": float(
            metrics["f1_malignant"]
        ),
        "specificity_benign": float(
            metrics["specificity_benign"]
        ),
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }


def create_summary_table(
    runs_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula estadísticos sobre las cinco ejecuciones."""
    rows: list[dict[str, float | str]] = []

    for metric_name in METRICS_TO_SUMMARIZE:
        values = runs_table[
            metric_name
        ].astype(float)

        rows.append(
            {
                "metric": metric_name,
                "mean": float(values.mean()),
                "standard_deviation": float(
                    values.std(ddof=1)
                ),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Genera los resúmenes de Breast Cancer."""
    input_paths = sorted(
        METRICS_DIRECTORY.glob(INPUT_PATTERN)
    )

    if len(input_paths) != EXPECTED_NUMBER_OF_RUNS:
        raise RuntimeError(
            "Se esperaban exactamente "
            f"{EXPECTED_NUMBER_OF_RUNS} ejecuciones, "
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

    if runs_table["training_seed"].duplicated().any():
        raise RuntimeError(
            "Se han encontrado semillas de entrenamiento duplicadas."
        )

    summary_table = create_summary_table(
        runs_table
    )

    confusion_columns = [
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    summed_confusion = {
        column: int(runs_table[column].sum())
        for column in confusion_columns
    }

    mean_confusion = {
        column: float(runs_table[column].mean())
        for column in confusion_columns
    }

    runs_output_path = (
        METRICS_DIRECTORY
        / "breast_cancer_hybrid_runs.csv"
    )

    summary_output_path = (
        METRICS_DIRECTORY
        / "breast_cancer_hybrid_summary.csv"
    )

    json_output_path = (
        METRICS_DIRECTORY
        / "breast_cancer_hybrid_summary.json"
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
        "dataset": "Breast Cancer Wisconsin",
        "model": "hybrid quantum-classical classifier",
        "number_of_runs": int(len(runs_table)),
        "split_seed": int(
            runs_table["split_seed"].iloc[0]
        ),
        "training_seeds": [
            int(seed)
            for seed in runs_table[
                "training_seed"
            ].tolist()
        ],
        "metrics": metric_summary,
        "summed_confusion_matrix": [
            [
                summed_confusion["true_negatives"],
                summed_confusion["false_positives"],
            ],
            [
                summed_confusion["false_negatives"],
                summed_confusion["true_positives"],
            ],
        ],
        "mean_confusion_matrix": [
            [
                mean_confusion["true_negatives"],
                mean_confusion["false_positives"],
            ],
            [
                mean_confusion["false_negatives"],
                mean_confusion["true_positives"],
            ],
        ],
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

    print("\n" + "=" * 75)
    print("BREAST CANCER — INDIVIDUAL RUNS")
    print("=" * 75)

    display_columns = [
        "training_seed",
        "best_epoch",
        "best_validation_loss",
        "accuracy",
        "precision_malignant",
        "recall_malignant",
        "f1_malignant",
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

    print("\n" + "=" * 75)
    print("AGGREGATED RESULTS")
    print("=" * 75)

    print(
        summary_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSummed confusion matrix:")
    print(
        np.array(
            aggregate_summary[
                "summed_confusion_matrix"
            ]
        )
    )

    print("\nGenerated files:")
    print(f"  {runs_output_path}")
    print(f"  {summary_output_path}")
    print(f"  {json_output_path}")


if __name__ == "__main__":
    main()