"""Pruebas de los modelos híbridos cuántico-clásicos."""

import torch

from src.models import (
    HybridClassifier,
    build_binary_hybrid_model,
    build_multiclass_hybrid_model,
)


def create_test_inputs() -> torch.Tensor:
    """Crea muestras de entrada dentro del intervalo angular."""
    return torch.tensor(
        [
            [0.1, 0.4, 0.8, 1.2],
            [0.3, 0.7, 1.1, 1.8],
        ],
        dtype=torch.float32,
    )


def test_binary_model_has_expected_parameter_count() -> None:
    """Comprueba los 21 parámetros del modelo binario."""
    model = build_binary_hybrid_model(seed=42)

    assert model.number_of_quantum_parameters == 16
    assert model.number_of_classical_parameters == 5
    assert model.number_of_trainable_parameters == 21


def test_multiclass_model_has_expected_parameter_count() -> None:
    """Comprueba los 31 parámetros del modelo multiclase."""
    model = build_multiclass_hybrid_model(seed=42)

    assert model.number_of_quantum_parameters == 16
    assert model.number_of_classical_parameters == 15
    assert model.number_of_trainable_parameters == 31


def test_binary_model_produces_one_logit_per_sample() -> None:
    """Comprueba la forma de la salida binaria."""
    model = build_binary_hybrid_model(seed=42)
    input_data = create_test_inputs()

    logits = model(input_data)

    assert logits.shape == (2, 1)


def test_multiclass_model_produces_three_logits() -> None:
    """Comprueba la forma de la salida multiclase."""
    model = build_multiclass_hybrid_model(seed=42)
    input_data = create_test_inputs()

    logits = model(input_data)

    assert logits.shape == (2, 3)


def test_binary_probabilities_are_valid() -> None:
    """Comprueba que las probabilidades binarias pertenecen a [0, 1]."""
    model = build_binary_hybrid_model(seed=42)
    input_data = create_test_inputs()

    with torch.no_grad():
        probabilities = model.predict_probabilities(input_data)

    assert probabilities.shape == (2, 1)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_multiclass_probabilities_sum_to_one() -> None:
    """Comprueba la normalización softmax."""
    model = build_multiclass_hybrid_model(seed=42)
    input_data = create_test_inputs()

    with torch.no_grad():
        probabilities = model.predict_probabilities(input_data)

    assert probabilities.shape == (2, 3)

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_gradients_reach_both_model_components() -> None:
    """Comprueba la retropropagación cuántico-clásica."""
    model = build_binary_hybrid_model(seed=42)
    input_data = create_test_inputs()

    logits = model(input_data)
    loss = logits.mean()

    loss.backward()

    quantum_gradient = model.quantum_layer.weight.grad
    classical_weight_gradient = model.classical_layer.weight.grad
    classical_bias_gradient = model.classical_layer.bias.grad

    assert quantum_gradient is not None
    assert classical_weight_gradient is not None
    assert classical_bias_gradient is not None

    assert torch.all(torch.isfinite(quantum_gradient))
    assert torch.all(torch.isfinite(classical_weight_gradient))
    assert torch.all(torch.isfinite(classical_bias_gradient))


def test_initialization_is_reproducible() -> None:
    """Comprueba que una semilla produce los mismos parámetros."""
    first_model = build_binary_hybrid_model(seed=42)
    second_model = build_binary_hybrid_model(seed=42)

    torch.testing.assert_close(
        first_model.quantum_layer.weight,
        second_model.quantum_layer.weight,
    )

    torch.testing.assert_close(
        first_model.classical_layer.weight,
        second_model.classical_layer.weight,
    )

    torch.testing.assert_close(
        first_model.classical_layer.bias,
        second_model.classical_layer.bias,
    )


def test_invalid_input_dimension_is_rejected() -> None:
    """Comprueba que solo se aceptan cuatro características."""
    model = HybridClassifier(
        number_of_classes=2,
        seed=42,
    )

    invalid_input = torch.zeros(
        size=(2, 3),
        dtype=torch.float32,
    )

    try:
        model(invalid_input)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Se esperaba ValueError para tres características."
        )
    