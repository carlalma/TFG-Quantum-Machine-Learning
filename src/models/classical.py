"""Modelos clásicos utilizados como referencia experimental."""

from __future__ import annotations

from typing import Literal

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch import Tensor, nn


ClassicalTask = Literal[
    "binary",
    "multiclass",
]


def _get_class_weight(
    task: ClassicalTask,
) -> str | None:
    """
    Devuelve la ponderación utilizada en cada problema.

    Breast Cancer utiliza pérdida no ponderada.
    Wine Quality utiliza ponderación inversa de frecuencias.
    """
    if task == "binary":
        return None

    if task == "multiclass":
        return "balanced"

    raise ValueError(
        f"Tarea de clasificación no reconocida: {task}"
    )


def build_logistic_regression(
    *,
    task: ClassicalTask,
    seed: int = 42,
) -> LogisticRegression:
    """Construye el modelo de regresión logística."""
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight=_get_class_weight(task),
        random_state=seed,
    )


def build_rbf_svm(
    *,
    task: ClassicalTask,
    seed: int = 42,
) -> SVC:
    """Construye la máquina de vectores soporte con kernel RBF."""
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight=_get_class_weight(task),
        decision_function_shape="ovr",
        probability=False,
        random_state=seed,
    )


class ClassicalMLPClassifier(nn.Module):
    """
    Perceptrón multicapa clásico con una capa oculta.

    Arquitectura:
        4 entradas -> 4 neuronas ocultas -> salida
    """

    def __init__(
        self,
        *,
        number_of_classes: int,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if number_of_classes < 2:
            raise ValueError(
                "El número de clases debe ser al menos dos."
            )

        self.number_of_classes = number_of_classes
        self.input_dimension = 4
        self.hidden_dimension = 4

        output_dimension = (
            1
            if number_of_classes == 2
            else number_of_classes
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            self.network = nn.Sequential(
                nn.Linear(
                    self.input_dimension,
                    self.hidden_dimension,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.hidden_dimension,
                    output_dimension,
                ),
            )

    @property
    def number_of_trainable_parameters(self) -> int:
        """Devuelve el número de parámetros entrenables."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        input_data: Tensor,
    ) -> Tensor:
        """Calcula los logits del clasificador."""
        if input_data.shape[-1] != self.input_dimension:
            raise ValueError(
                "El MLP necesita exactamente "
                "cuatro características de entrada."
            )

        return self.network(input_data)


def build_binary_classical_mlp(
    *,
    seed: int = 42,
) -> ClassicalMLPClassifier:
    """Construye el MLP para Breast Cancer Wisconsin."""
    return ClassicalMLPClassifier(
        number_of_classes=2,
        seed=seed,
    )


def build_multiclass_classical_mlp(
    *,
    seed: int = 42,
) -> ClassicalMLPClassifier:
    """Construye el MLP para Wine Quality."""
    return ClassicalMLPClassifier(
        number_of_classes=3,
        seed=seed,
    )