"""Construcción del circuito cuántico variacional"""

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector


@dataclass(frozen=True)
class VariationalCircuit:
    """Circuito junto con sus parámetros de entrada y train."""

    circuit: QuantumCircuit
    input_parameters: tuple[Parameter, ...]
    weight_parameters: tuple[Parameter, ...]


def build_variational_circuit(
    *,
    number_of_qubits: int = 4,
    number_of_layers: int = 2,
) -> VariationalCircuit:
    """
    Construye el circuito cuántico variacional del sistema híbrido.

    Estructura:
        1. Codificación angular mediante RY.
        2. Rotaciones entrenables RY y RZ.
        3. Entrelazamiento CNOT circular.
        4. Repetición del ansatz con dos capas.

    Args:
        number_of_qubits: Número de qubits y características de entrada.
        number_of_layers: Número de capas variacionales.

    Returns:
        Circuito y colecciones ordenadas de parámetros.

    Raises:
        ValueError: Si el número de qubits o capas no es válido.
    """
    if number_of_qubits < 2:
        raise ValueError(
            "El circuito necesita al menos dos qubits "
            "para generar entrelazamiento."
        )

    if number_of_layers < 1:
        raise ValueError(
            "El circuito debe contener al menos una capa variacional."
        )

    input_parameters = ParameterVector(
        name="x",
        length=number_of_qubits,
    )

    number_of_parameters_per_rotation = (
        number_of_qubits * number_of_layers
    )

    theta_parameters = ParameterVector(
        name="theta",
        length=number_of_parameters_per_rotation,
    )

    phi_parameters = ParameterVector(
        name="phi",
        length=number_of_parameters_per_rotation,
    )

    circuit = QuantumCircuit(
        number_of_qubits,
        name="hybrid_vqc",
    )

    # Bloque 1: codificación angular de las características.
    for qubit in range(number_of_qubits):
        circuit.ry(input_parameters[qubit], qubit)

    # Bloque 2: ansatz variacional.
    for layer in range(number_of_layers):
        parameter_offset = layer * number_of_qubits

        # Rotaciones locales entrenables.
        for qubit in range(number_of_qubits):
            parameter_index = parameter_offset + qubit

            circuit.ry(
                theta_parameters[parameter_index],
                qubit,
            )
            circuit.rz(
                phi_parameters[parameter_index],
                qubit,
            )

        # Entrelazamiento circular:
        # q0 → q1 → q2 → q3 → q0.
        for control_qubit in range(number_of_qubits):
            target_qubit = (
                control_qubit + 1
            ) % number_of_qubits

            circuit.cx(
                control_qubit,
                target_qubit,
            )

    weight_parameters = (
        tuple(theta_parameters)
        + tuple(phi_parameters)
    )

    return VariationalCircuit(
        circuit=circuit,
        input_parameters=tuple(input_parameters),
        weight_parameters=weight_parameters,
    )