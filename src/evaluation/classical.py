"""Evaluación de los modelos clásicos de scikit-learn."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassicalEvaluationResult:
    """Resultados de un clasificador clásico."""

    metrics: dict[str, float | int]
    confusion_matrix: np.ndarray
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    scores: np.ndarray | None
    score_type: str | None
    per_class_metrics: (
        dict[str, dict[str, float | int]] | None
    ) = None


def _extract_scores(
    model: Any,
    features: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Extrae probabilidades o puntuaciones cuando estén disponibles."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(
            model.predict_proba(features),
            dtype=np.float64,
        )

        return probabilities, "predict_proba"

    if hasattr(model, "decision_function"):
        decision_scores = np.asarray(
            model.decision_function(features),
            dtype=np.float64,
        )

        return decision_scores, "decision_function"

    return None, None


def evaluate_classical_binary_classifier(
    model: Any,
    features: np.ndarray,
    targets: np.ndarray,
) -> ClassicalEvaluationResult:
    """Evalúa un modelo clásico sobre una tarea binaria."""
    true_labels = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(-1)

    predicted_labels = np.asarray(
        model.predict(features),
        dtype=np.int64,
    ).reshape(-1)

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "El número de predicciones no coincide "
            "con el número de etiquetas."
        )

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
        "specificity_benign": float(
            specificity
        ),
        "true_negatives": int(
            true_negatives
        ),
        "false_positives": int(
            false_positives
        ),
        "false_negatives": int(
            false_negatives
        ),
        "true_positives": int(
            true_positives
        ),
        "number_of_samples": int(
            len(true_labels)
        ),
    }

    scores, score_type = _extract_scores(
        model,
        features,
    )

    return ClassicalEvaluationResult(
        metrics=metrics,
        confusion_matrix=matrix,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        scores=scores,
        score_type=score_type,
    )


def evaluate_classical_multiclass_classifier(
    model: Any,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    class_labels: tuple[int, ...] = (0, 1, 2),
    class_names: tuple[str, ...] = (
        "quality_5",
        "quality_6",
        "quality_7",
    ),
) -> ClassicalEvaluationResult:
    """Evalúa un modelo clásico sobre una tarea multiclase."""
    if len(class_labels) != len(class_names):
        raise ValueError(
            "Debe proporcionarse un nombre por cada clase."
        )

    true_labels = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(-1)

    predicted_labels = np.asarray(
        model.predict(features),
        dtype=np.int64,
    ).reshape(-1)

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

    scores, score_type = _extract_scores(
        model,
        features,
    )

    return ClassicalEvaluationResult(
        metrics=metrics,
        confusion_matrix=matrix,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        scores=scores,
        score_type=score_type,
        per_class_metrics=per_class_metrics,
    )