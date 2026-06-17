"""Pruebas de la arquitectura del circuito cuántico."""

from src.quantum import build_variational_circuit


def test_circuit_has_expected_dimensions() -> None:
    """Comprueba qubits y parámetros del diseño base."""
    model = build_variational_circuit()

    assert model.circuit.num_qubits == 4
    assert model.circuit.num_clbits == 0

    assert len(model.input_parameters) == 4
    assert len(model.weight_parameters) == 16

    assert len(model.circuit.parameters) == 20


def test_circuit_has_expected_gates() -> None:
    """Comprueba las puertas de codificación y del ansatz."""
    model = build_variational_circuit()

    operation_counts = model.circuit.count_ops()

    # 4 RY de codificación + 8 RY entrenables.
    assert operation_counts.get("ry", 0) == 12

    # 2 capas × 4 rotaciones RZ.
    assert operation_counts.get("rz", 0) == 8

    # 2 capas × 4 CNOT circulares.
    assert operation_counts.get("cx", 0) == 8


def test_all_parameters_can_be_assigned() -> None:
    """Comprueba que el circuito puede instanciarse numéricamente."""
    model = build_variational_circuit()

    parameter_values = {
        parameter: 0.1
        for parameter in model.circuit.parameters
    }

    bound_circuit = model.circuit.assign_parameters(
        parameter_values,
        inplace=False,
    )

    assert len(bound_circuit.parameters) == 0


def test_invalid_configuration_is_rejected() -> None:
    """Comprueba la validación de la configuración."""
    try:
        build_variational_circuit(number_of_qubits=1)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Se esperaba ValueError para un único qubit."
        )

    try:
        build_variational_circuit(number_of_layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Se esperaba ValueError para cero capas."
        )