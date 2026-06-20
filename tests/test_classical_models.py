"""Pruebas de los modelos clásicos de referencia."""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.models import (
    build_binary_classical_mlp,
    build_logistic_regression,
    build_multiclass_classical_mlp,
    build_rbf_svm,
)


def create_test_inputs() -> torch.Tensor:
    """Crea dos muestras con cuatro características."""
    return torch.tensor(
        [
            [0.1, -0.4, 0.8, 1.2],
            [-0.3, 0.7, -1.1, 0.8],
        ],
        dtype=torch.float32,
    )


def test_logistic_regression_configuration() -> None:
    """Comprueba la configuración de la regresión logística."""
    binary_model = build_logistic_regression(
        task="binary",
        seed=42,
    )

    multiclass_model = build_logistic_regression(
        task="multiclass",
        seed=42,
    )

    assert isinstance(
        binary_model,
        LogisticRegression,
    )

    assert binary_model.solver == "lbfgs"
    assert binary_model.class_weight is None
    assert multiclass_model.class_weight == "balanced"


def test_rbf_svm_configuration() -> None:
    """Comprueba la configuración de la SVM."""
    binary_model = build_rbf_svm(
        task="binary",
        seed=42,
    )

    multiclass_model = build_rbf_svm(
        task="multiclass",
        seed=42,
    )

    assert isinstance(binary_model, SVC)
    assert binary_model.kernel == "rbf"
    assert binary_model.gamma == "scale"
    assert binary_model.class_weight is None
    assert multiclass_model.class_weight == "balanced"


def test_binary_mlp_parameter_count() -> None:
    """Comprueba los 25 parámetros del MLP binario."""
    model = build_binary_classical_mlp(
        seed=42
    )

    assert model.number_of_trainable_parameters == 25


def test_multiclass_mlp_parameter_count() -> None:
    """Comprueba los 35 parámetros del MLP multiclase."""
    model = build_multiclass_classical_mlp(
        seed=42
    )

    assert model.number_of_trainable_parameters == 35


def test_classical_mlp_output_shapes() -> None:
    """Comprueba las formas de salida de ambos MLP."""
    input_data = create_test_inputs()

    binary_model = build_binary_classical_mlp(
        seed=42
    )

    multiclass_model = (
        build_multiclass_classical_mlp(
            seed=42
        )
    )

    assert binary_model(input_data).shape == (2, 1)
    assert multiclass_model(input_data).shape == (2, 3)


def test_classical_mlp_initialization_is_reproducible() -> None:
    """Comprueba la inicialización controlada por semilla."""
    first_model = build_binary_classical_mlp(
        seed=42
    )

    second_model = build_binary_classical_mlp(
        seed=42
    )

    for first_parameter, second_parameter in zip(
        first_model.parameters(),
        second_model.parameters(),
    ):
        torch.testing.assert_close(
            first_parameter,
            second_parameter,
        )