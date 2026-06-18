"""Evaluación de clasificadores binarios."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class BinaryEvaluationResult:
    """Resultados obtenidos sobre una partición binaria."""

    metrics: dict[str, float | int]
    confusion_matrix: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    probabilities: np.ndarray


def evaluate_binary_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    threshold: float = 0.5,
) -> BinaryEvaluationResult:
    """
    Evalúa un clasificador binario.

    La etiqueta 1 representa la clase positiva, que en Breast Cancer
    Wisconsin corresponde al diagnóstico maligno.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "El umbral debe pertenecer al intervalo abierto (0, 1)."
        )

    model.eval()

    true_label_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            logits = model(input_batch)

            probabilities = torch.sigmoid(
                logits
            ).reshape(-1)

            true_labels = target_batch.reshape(-1)

            probability_batches.append(
                probabilities.cpu().numpy()
            )

            true_label_batches.append(
                true_labels.cpu().numpy()
            )

    if not true_label_batches:
        raise ValueError(
            "El cargador de evaluación no contiene muestras."
        )

    true_labels = np.concatenate(
        true_label_batches
    ).astype(np.int64)

    probabilities = np.concatenate(
        probability_batches
    ).astype(np.float64)

    predicted_labels = (
        probabilities >= threshold
    ).astype(np.int64)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    (
        true_negatives,
        false_positives,
        false_negatives,
        true_positives,
    ) = matrix.ravel()

    specificity_denominator = (
        true_negatives + false_positives
    )

    specificity = (
        true_negatives / specificity_denominator
        if specificity_denominator > 0
        else 0.0
    )

    metrics: dict[str, float | int] = {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "precision_malignant": float(
            precision_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_malignant": float(
            recall_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_malignant": float(
            f1_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity_benign": float(specificity),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "number_of_samples": int(len(true_labels)),
        "classification_threshold": float(threshold),
    }

    return BinaryEvaluationResult(
        metrics=metrics,
        confusion_matrix=matrix,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
    )