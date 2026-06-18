"""Pruebas de evaluación y generación de resultados."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    plot_confusion_matrix,
    plot_training_history,
    save_experiment_summary,
    save_training_history,
)
from src.training import TrainingHistory


class FixedLogitModel(nn.Module):
    """Modelo que utiliza la primera característica como logit."""

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return input_data[:, :1]


def test_binary_evaluation_with_perfect_predictions() -> None:
    """Comprueba métricas binarias conocidas."""
    features = torch.tensor(
        [
            [-3.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=torch.float32,
    )

    data_loader = DataLoader(
        TensorDataset(
            features,
            targets,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = evaluate_binary_classifier(
        FixedLogitModel(),
        data_loader,
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["precision_malignant"] == 1.0
    assert result.metrics["recall_malignant"] == 1.0
    assert result.metrics["f1_malignant"] == 1.0

    np.testing.assert_array_equal(
        result.confusion_matrix,
        np.array(
            [
                [2, 0],
                [0, 2],
            ]
        ),
    )


def test_experiment_data_is_saved(
    tmp_path: Path,
) -> None:
    """Comprueba el guardado de JSON y CSV."""
    history = TrainingHistory(
        training_loss=[0.8, 0.6, 0.4],
        validation_loss=[0.9, 0.7, 0.5],
    )

    summary_path = (
        tmp_path / "summary.json"
    )

    history_path = (
        tmp_path / "history.csv"
    )

    save_experiment_summary(
        {
            "dataset": "test",
            "accuracy": 1.0,
        },
        summary_path,
    )

    save_training_history(
        history,
        history_path,
    )

    assert summary_path.exists()
    assert history_path.exists()
    assert summary_path.stat().st_size > 0
    assert history_path.stat().st_size > 0

class FixedMulticlassModel(nn.Module):
    """Modelo que utiliza las tres primeras entradas como logits."""

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return input_data[:, :3]

def test_experiment_figures_are_saved(
    tmp_path: Path,
) -> None:
    """Comprueba la generación de las figuras."""
    history = TrainingHistory(
        training_loss=[0.8, 0.6, 0.4],
        validation_loss=[0.9, 0.7, 0.5],
    )

    training_figure = (
        tmp_path / "training.png"
    )

    confusion_figure = (
        tmp_path / "confusion.png"
    )

    plot_training_history(
        history,
        training_figure,
    )

    plot_confusion_matrix(
        np.array(
            [
                [8, 1],
                [2, 9],
            ]
        ),
        confusion_figure,
    )

    assert training_figure.exists()
    assert confusion_figure.exists()
    assert training_figure.stat().st_size > 0
    assert confusion_figure.stat().st_size > 0

def test_multiclass_evaluation_with_perfect_predictions() -> None:
    """Comprueba métricas multiclase conocidas."""
    features = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [0, 1, 2, 0, 1, 2],
        dtype=torch.int64,
    )

    data_loader = DataLoader(
        TensorDataset(
            features,
            targets,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = evaluate_multiclass_classifier(
        FixedMulticlassModel(),
        data_loader,
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["precision_macro"] == 1.0
    assert result.metrics["recall_macro"] == 1.0
    assert result.metrics["f1_macro"] == 1.0

    np.testing.assert_array_equal(
        result.confusion_matrix,
        np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
            ]
        ),
    )

    np.testing.assert_allclose(
        result.probabilities.sum(axis=1),
        np.ones(6),
        atol=1e-6,
    )

    assert (
        result.per_class_metrics[
            "quality_5"
        ]["support"]
        == 2
    )

    assert (
        result.per_class_metrics[
            "quality_6"
        ]["support"]
        == 2
    )

    assert (
        result.per_class_metrics[
            "quality_7"
        ]["support"]
        == 2
    )


def test_multiclass_confusion_figure_is_saved(
    tmp_path: Path,
) -> None:
    """Comprueba una matriz de confusión de tres clases."""
    output_path = (
        tmp_path
        / "multiclass_confusion.png"
    )

    plot_confusion_matrix(
        np.array(
            [
                [8, 1, 0],
                [2, 7, 1],
                [0, 2, 6],
            ]
        ),
        output_path,
        class_names=(
            "Quality 5",
            "Quality 6",
            "Quality 7",
        ),
        title=(
            "Wine Quality test confusion matrix"
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0  