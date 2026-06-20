"""Agregación y comparación de los modelos clásicos e híbridos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIRECTORY = PROJECT_ROOT / "results" / "metrics"

EXPECTED_SEEDS = {42, 123, 456, 789, 2026}

DATASET_CONFIGURATIONS = {
    "breast_cancer": {
        "display_name": "Breast Cancer Wisconsin",
        "metrics": (
            "accuracy",
            "precision_malignant",
            "recall_malignant",
            "f1_malignant",
            "specificity_benign",
        ),
    },
    "wine_quality": {
        "display_name": "Wine Quality",
        "metrics": (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
        ),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    """Carga un archivo JSON."""
    with path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        return json.load(input_file)


def summarize_values(
    values: pd.Series,
) -> dict[str, float]:
    """Calcula los estadísticos descriptivos de una serie."""
    numeric_values = values.astype(float)

    return {
        "mean": float(numeric_values.mean()),
        "standard_deviation": float(
            numeric_values.std(ddof=1)
        ),
        "minimum": float(numeric_values.min()),
        "maximum": float(numeric_values.max()),
    }


def summarize_mlp_dataset(
    dataset_name: str,
) -> dict[str, Any]:
    """Agrega las cinco ejecuciones del MLP de un dataset."""
    configuration = DATASET_CONFIGURATIONS[
        dataset_name
    ]

    input_paths = sorted(
        METRICS_DIRECTORY.glob(
            f"{dataset_name}_mlp_split_42_seed_*.json"
        )
    )

    if len(input_paths) != 5:
        raise RuntimeError(
            f"Se esperaban cinco ejecuciones MLP para "
            f"{dataset_name}, pero se encontraron "
            f"{len(input_paths)}."
        )

    rows: list[dict[str, float | int | bool]] = []

    for input_path in input_paths:
        summary = load_json(input_path)

        seed = int(
            summary["random_seeds"][
                "model_and_training"
            ]
        )

        training_result = summary["training_result"]
        metrics = summary["test_results"]["metrics"]
        execution_times = summary[
            "execution_times_seconds"
        ]

        row: dict[
            str,
            float | int | bool,
        ] = {
            "training_seed": seed,
            "best_epoch": int(
                training_result["best_epoch"]
            ),
            "best_validation_loss": float(
                training_result[
                    "best_validation_loss"
                ]
            ),
            "epochs_completed": int(
                training_result["epochs_completed"]
            ),
            "stopped_early": bool(
                training_result["stopped_early"]
            ),
            "training_time_seconds": float(
                execution_times["training"]
            ),
        }

        for metric_name in configuration["metrics"]:
            row[metric_name] = float(
                metrics[metric_name]
            )

        rows.append(row)

    runs_table = (
        pd.DataFrame(rows)
        .sort_values("training_seed")
        .reset_index(drop=True)
    )

    detected_seeds = set(
        runs_table["training_seed"].astype(int)
    )

    if detected_seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Las semillas encontradas no coinciden con "
            f"las esperadas: {sorted(detected_seeds)}."
        )

    summary_columns = [
        *configuration["metrics"],
        "best_epoch",
        "best_validation_loss",
        "epochs_completed",
        "training_time_seconds",
    ]

    summary_rows = []

    for column_name in summary_columns:
        statistics = summarize_values(
            runs_table[column_name]
        )

        summary_rows.append(
            {
                "metric": column_name,
                **statistics,
            }
        )

    summary_table = pd.DataFrame(
        summary_rows
    )

    runs_path = (
        METRICS_DIRECTORY
        / f"{dataset_name}_mlp_runs.csv"
    )

    summary_path = (
        METRICS_DIRECTORY
        / f"{dataset_name}_mlp_summary.csv"
    )

    json_path = (
        METRICS_DIRECTORY
        / f"{dataset_name}_mlp_summary.json"
    )

    runs_table.to_csv(
        runs_path,
        index=False,
    )

    summary_table.to_csv(
        summary_path,
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

    output_summary = {
        "dataset": configuration["display_name"],
        "model": "classical MLP",
        "number_of_runs": 5,
        "split_seed": 42,
        "training_seeds": sorted(
            EXPECTED_SEEDS
        ),
        "metrics": metric_summary,
    }

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output_summary,
            output_file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 90)
    print(
        f"{configuration['display_name'].upper()} "
        "— MLP RUNS"
    )
    print("=" * 90)

    print(
        runs_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nMLP aggregated results:")

    print(
        summary_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return output_summary


def load_single_baseline(
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    """Carga un resultado determinista de scikit-learn."""
    path = (
        METRICS_DIRECTORY
        / (
            f"{dataset_name}_{model_name}"
            "_split_42_seed_42.json"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el resultado: {path}"
        )

    return load_json(path)


def load_hybrid_summary(
    dataset_name: str,
) -> dict[str, Any]:
    """Carga el resumen de las ejecuciones híbridas."""
    path = (
        METRICS_DIRECTORY
        / f"{dataset_name}_hybrid_summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el resumen híbrido: {path}"
        )

    return load_json(path)


def create_comparison_table(
    dataset_name: str,
    mlp_summary: dict[str, Any],
) -> pd.DataFrame:
    """Compara los cuatro modelos de un dataset."""
    configuration = DATASET_CONFIGURATIONS[
        dataset_name
    ]

    hybrid_summary = load_hybrid_summary(
        dataset_name
    )

    logistic_result = load_single_baseline(
        dataset_name,
        "logistic_regression",
    )

    svm_result = load_single_baseline(
        dataset_name,
        "rbf_svm",
    )

    rows: list[dict[str, Any]] = []

    model_sources = (
        (
            "Hybrid VQC",
            hybrid_summary,
            5,
            True,
        ),
        (
            "Logistic regression",
            logistic_result,
            1,
            False,
        ),
        (
            "SVM RBF",
            svm_result,
            1,
            False,
        ),
        (
            "Classical MLP",
            mlp_summary,
            5,
            True,
        ),
    )

    for (
        model_display_name,
        source,
        number_of_runs,
        is_aggregate,
    ) in model_sources:
        row: dict[str, Any] = {
            "model": model_display_name,
            "number_of_runs": number_of_runs,
        }

        if is_aggregate:
            metrics = source["metrics"]

            for metric_name in configuration["metrics"]:
                row[
                    f"{metric_name}_mean"
                ] = float(
                    metrics[metric_name]["mean"]
                )

                row[
                    f"{metric_name}_std"
                ] = float(
                    metrics[metric_name][
                        "standard_deviation"
                    ]
                )

        else:
            metrics = source[
                "test_results"
            ]["metrics"]

            for metric_name in configuration["metrics"]:
                row[
                    f"{metric_name}_mean"
                ] = float(
                    metrics[metric_name]
                )

                # No existe variabilidad entre ejecuciones,
                # porque estos modelos se ajustaron una vez.
                row[
                    f"{metric_name}_std"
                ] = np.nan

        if is_aggregate:
            time_information = source["metrics"].get(
                "training_time_seconds"
            )

            row["training_time_mean_seconds"] = (
                float(time_information["mean"])
                if time_information is not None
                else np.nan
            )

            row["training_time_std_seconds"] = (
                float(
                    time_information[
                        "standard_deviation"
                    ]
                )
                if time_information is not None
                else np.nan
            )

        else:
            row["training_time_mean_seconds"] = float(
                source[
                    "execution_times_seconds"
                ]["training"]
            )

            row["training_time_std_seconds"] = np.nan

        rows.append(row)

    comparison_table = pd.DataFrame(rows)

    output_path = (
        METRICS_DIRECTORY
        / f"{dataset_name}_model_comparison.csv"
    )

    comparison_table.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 90)
    print(
        f"{configuration['display_name'].upper()} "
        "— MODEL COMPARISON"
    )
    print("=" * 90)

    print(
        comparison_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nGenerated: {output_path}")

    return comparison_table


def main() -> None:
    """Genera los resúmenes MLP y las comparaciones finales."""
    breast_mlp_summary = summarize_mlp_dataset(
        "breast_cancer"
    )

    wine_mlp_summary = summarize_mlp_dataset(
        "wine_quality"
    )

    create_comparison_table(
        "breast_cancer",
        breast_mlp_summary,
    )

    create_comparison_table(
        "wine_quality",
        wine_mlp_summary,
    )


if __name__ == "__main__":
    main()