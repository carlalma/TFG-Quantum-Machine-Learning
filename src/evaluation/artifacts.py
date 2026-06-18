"""Guardado de resultados y figuras experimentales."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training import TrainingHistory


def save_experiment_summary(
    summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Guarda el resumen de un experimento en formato JSON."""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            summary,
            output_file,
            indent=4,
            ensure_ascii=False,
        )


def save_training_history(
    history: TrainingHistory,
    output_path: str | Path,
) -> None:
    """Guarda las pérdidas de cada época en formato CSV."""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_table = pd.DataFrame(
        {
            "epoch": range(
                1,
                len(history.training_loss) + 1,
            ),
            "training_loss": history.training_loss,
            "validation_loss": history.validation_loss,
        }
    )

    history_table.to_csv(
        path,
        index=False,
    )


def plot_training_history(
    history: TrainingHistory,
    output_path: str | Path,
) -> None:
    """Genera la curva de entrenamiento y validación."""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history.training_loss) + 1,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        epochs,
        history.training_loss,
        label="Training loss",
    )

    axis.plot(
        epochs,
        history.validation_loss,
        label="Validation loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(
        "Hybrid model training history"
    )

    axis.legend()
    axis.grid(True)

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: str | Path,
    *,
    class_names: tuple[str, str] = (
        "Benign",
        "Malignant",
    ),
) -> None:
    """Genera una matriz de confusión binaria."""
    if matrix.shape != (2, 2):
        raise ValueError(
            "La matriz de confusión debe tener forma (2, 2)."
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(matrix)

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_xticks(
        [0, 1],
        labels=class_names,
    )

    axis.set_yticks(
        [0, 1],
        labels=class_names,
    )

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Test confusion matrix")

    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)