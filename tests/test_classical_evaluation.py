"""Pruebas de evaluación de los modelos clásicos."""

import numpy as np

from src.evaluation import (
    evaluate_classical_binary_classifier,
    evaluate_classical_multiclass_classifier,
)


class FixedBinaryEstimator:
    """Clasificador binario basado en el signo de la entrada."""

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return (
            features[:, 0] >= 0.0
        ).astype(np.int64)

    def decision_function(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features[:, 0]


class FixedMulticlassEstimator:
    """Clasificador basado en la mayor de tres entradas."""

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features[:, :3].argmax(
            axis=1
        )

    def decision_function(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return features[:, :3]


def test_classical_binary_evaluation() -> None:
    """Comprueba una clasificación binaria perfecta."""
    features = np.array(
        [
            [-2.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [0, 1, 0, 1],
        dtype=np.int64,
    )

    result = evaluate_classical_binary_classifier(
        FixedBinaryEstimator(),
        features,
        targets,
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["recall_malignant"] == 1.0
    assert result.score_type == "decision_function"

    np.testing.assert_array_equal(
        result.confusion_matrix,
        np.array(
            [
                [2, 0],
                [0, 2],
            ]
        ),
    )


def test_classical_multiclass_evaluation() -> None:
    """Comprueba una clasificación multiclase perfecta."""
    features = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [0, 1, 2, 0, 1, 2],
        dtype=np.int64,
    )

    result = (
        evaluate_classical_multiclass_classifier(
            FixedMulticlassEstimator(),
            features,
            targets,
        )
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["f1_macro"] == 1.0
    assert result.score_type == "decision_function"

    assert (
        result.per_class_metrics is not None
    )

    assert (
        result.per_class_metrics[
            "quality_7"
        ]["support"]
        == 2
    )

    np.testing.assert_array_equal(
        result.confusion_matrix,
        np.eye(
            3,
            dtype=np.int64,
        )
        * 2,
    )
    