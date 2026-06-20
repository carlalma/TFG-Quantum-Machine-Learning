"""Evaluación y almacenamiento de resultados experimentales."""

from .artifacts import (
    plot_confusion_matrix,
    plot_training_history,
    save_experiment_summary,
    save_training_history,
)
from .binary import (
    BinaryEvaluationResult,
    evaluate_binary_classifier,
)
from .multiclass import (
    MulticlassEvaluationResult,
    evaluate_multiclass_classifier,
)
from .classical import (
    ClassicalEvaluationResult,
    evaluate_classical_binary_classifier,
    evaluate_classical_multiclass_classifier,
)

__all__ = [
    "BinaryEvaluationResult",
    "MulticlassEvaluationResult",
    "evaluate_binary_classifier",
    "evaluate_multiclass_classifier",
    "plot_confusion_matrix",
    "plot_training_history",
    "save_experiment_summary",
    "save_training_history",
    "ClassicalEvaluationResult",
    "evaluate_classical_binary_classifier",
    "evaluate_classical_multiclass_classifier",
]