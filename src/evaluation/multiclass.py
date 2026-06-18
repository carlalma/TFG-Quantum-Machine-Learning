"""Evaluación de clasificadores multiclase."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class MulticlassEvaluationResult:
    """Resultados obtenidos sobre una partición multiclase."""

    metrics: dict[str, float | int]
    per_class_metrics: dict[str, dict[str, float | int]]
    confusion_matrix: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray


def evaluate_multiclass_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    class_labels: tuple[int, ...] = (0, 1, 2),
    class_names: tuple[str, ...] = (
        "quality_5",
        "quality_6",
        "quality_7",
    ),
) -> MulticlassEvaluationResult:
    """
    Evalúa un clasificador multiclase.

    Las etiquetas codificadas 0, 1 y 2 representan,
    respectivamente, las calidades originales 5, 6 y 7.
    """
    if len(class_labels) < 2:
        raise ValueError(
            "La evaluación necesita al menos dos clases."
        )

    if len(class_labels) != len(class_names):
        raise ValueError(
            "Debe proporcionarse un nombre por cada clase."
        )

    model.eval()

    true_label_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            logits = model(input_batch)

            if logits.ndim != 2:
                raise ValueError(
                    "La salida multiclase debe tener dos dimensiones."
                )

            if logits.shape[1] != len(class_labels):
                raise ValueError(
                    "El número de logits no coincide con "
                    "el número de clases."
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            true_label_batches.append(
                target_batch.reshape(-1).cpu().numpy()
            )

            probability_batches.append(
                probabilities.cpu().numpy()
            )

    if not true_label_batches:
        raise ValueError(
            "El cargador de evaluación no contiene muestras."
        )

    true_labels = np.concatenate(
        true_label_batches
    ).astype(np.int64)

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    ).astype(np.float64)

    predicted_labels = probabilities.argmax(
        axis=1
    ).astype(np.int64)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(class_labels),
    )

    metrics: dict[str, float | int] = {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "precision_macro": float(
            precision_score(
                true_labels,
                predicted_labels,
                labels=list(class_labels),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_labels,
                predicted_labels,
                labels=list(class_labels),
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=list(class_labels),
                average="macro",
                zero_division=0,
            )
        ),
        "number_of_samples": int(
            len(true_labels)
        ),
    }

    (
        precision_per_class,
        recall_per_class,
        f1_per_class,
        support_per_class,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(class_labels),
        average=None,
        zero_division=0,
    )

    per_class_metrics: dict[
        str,
        dict[str, float | int],
    ] = {}

    for index, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "encoded_label": int(
                class_labels[index]
            ),
            "precision": float(
                precision_per_class[index]
            ),
            "recall": float(
                recall_per_class[index]
            ),
            "f1": float(
                f1_per_class[index]
            ),
            "support": int(
                support_per_class[index]
            ),
        }

    return MulticlassEvaluationResult(
        metrics=metrics,
        per_class_metrics=per_class_metrics,
        confusion_matrix=matrix,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
    )