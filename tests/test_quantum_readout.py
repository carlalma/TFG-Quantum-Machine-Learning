"""Pruebas de la lectura del circuito cuántico."""

import numpy as np

from src.quantum import (
    build_pauli_z_observables,
    build_quantum_readout,
)


def test_four_pauli_z_observables_are_created() -> None:
    """Comprueba que existe un observable por qubit."""
    observables = build_pauli_z_observables(
        number_of_qubits=4
    )

    assert len(observables) == 4

    for observable in observables:
        assert observable.num_qubits == 4


def test_observables_act_on_expected_qubits() -> None:
    """Comprueba la correspondencia entre observables y qubits."""
    observables = build_pauli_z_observables(
        number_of_qubits=4
    )

    pauli_labels = [
        observable.paulis.to_labels()[0]
        for observable in observables
    ]

    # Qiskit representa los qubits en orden q3 q2 q1 q0.
    assert pauli_labels == [
        "IIIZ",
        "IIZI",
        "IZII",
        "ZIII",
    ]


def test_quantum_readout_has_expected_dimensions() -> None:
    """Comprueba entradas, pesos y salidas."""
    quantum_readout = build_quantum_readout()

    assert quantum_readout.num_inputs == 4
    assert quantum_readout.num_weights == 16
    assert quantum_readout.output_shape == (4,)


def test_zero_parameters_produce_expected_output() -> None:
    """
    Comprueba el resultado del circuito con entradas y pesos nulos.

    Con todas las rotaciones a cero, el estado permanece en |0000>
    y el valor esperado de Z para cada qubit es igual a uno.
    """
    quantum_readout = build_quantum_readout()

    input_data = np.zeros(
        shape=(2, 4),
        dtype=np.float64,
    )

    weights = np.zeros(
        shape=16,
        dtype=np.float64,
    )

    output = quantum_readout.forward(
        input_data=input_data,
        weights=weights,
    )

    assert output.shape == (2, 4)

    np.testing.assert_allclose(
        output,
        np.ones((2, 4)),
        atol=1e-8,
    )


def test_quantum_outputs_are_valid_expectation_values() -> None:
    """Comprueba que las salidas pertenecen al intervalo [-1, 1]."""
    quantum_readout = build_quantum_readout()

    input_data = np.array(
        [
            [0.1, 0.4, 0.8, 1.2],
            [0.2, 0.7, 1.1, 2.0],
        ],
        dtype=np.float64,
    )

    weights = np.linspace(
        start=-0.2,
        stop=0.2,
        num=16,
        dtype=np.float64,
    )

    output = quantum_readout.forward(
        input_data=input_data,
        weights=weights,
    )

    assert output.shape == (2, 4)
    assert np.all(output >= -1.0 - 1e-8)
    assert np.all(output <= 1.0 + 1e-8)


def test_quantum_readout_computes_gradients() -> None:
    """Comprueba que el circuito admite propagación de gradientes."""
    quantum_readout = build_quantum_readout()

    input_data = np.array(
        [[0.1, 0.2, 0.3, 0.4]],
        dtype=np.float64,
    )

    weights = np.full(
        shape=16,
        fill_value=0.1,
        dtype=np.float64,
    )

    input_gradients, weight_gradients = (
        quantum_readout.backward(
            input_data=input_data,
            weights=weights,
        )
    )

    assert input_gradients is not None
    assert weight_gradients is not None

    assert input_gradients.shape[-1] == 4
    assert weight_gradients.shape[-1] == 16
    