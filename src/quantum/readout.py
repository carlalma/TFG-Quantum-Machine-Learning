"""Lectura de los valores esperados del circuito cuántico."""

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .circuit import (
    VariationalCircuit,
    build_variational_circuit,
)


def build_pauli_z_observables(
    number_of_qubits: int = 4,
) -> tuple[SparsePauliOp, ...]:
    """
    Construye un observable Pauli Z independiente para cada qubit.

    Para cuatro qubits se obtienen los observables asociados a:

        <Z_0>, <Z_1>, <Z_2>, <Z_3>

    Cada observable actúa con Z sobre un único qubit y con la identidad
    sobre los restantes.

    Args:
        number_of_qubits: Número total de qubits del circuito.

    Returns:
        Tupla con un observable Pauli Z por qubit.

    Raises:
        ValueError: Si el número de qubits no es positivo.
    """
    if number_of_qubits < 1:
        raise ValueError(
            "El número de qubits debe ser mayor que cero."
        )

    return tuple(
        SparsePauliOp.from_sparse_list(
            [
                (
                    "Z",
                    [qubit],
                    1.0,
                )
            ],
            num_qubits=number_of_qubits,
        )
        for qubit in range(number_of_qubits)
    )


def build_quantum_readout(
    variational_circuit: VariationalCircuit | None = None,
    *,
    seed: int = 42,
) -> EstimatorQNN:
    """
    Construye la interfaz de lectura diferenciable del circuito.

    Esta función no introduce un nuevo circuito ni una arquitectura
    cuántica adicional. Envuelve el VQC existente para:

        1. Recibir cuatro características de entrada.
        2. Gestionar los dieciséis parámetros entrenables.
        3. Calcular cuatro valores esperados de Pauli Z.
        4. Permitir la propagación de gradientes.

    Args:
        variational_circuit:
            Circuito previamente construido. Si no se proporciona,
            se utiliza la configuración base de cuatro qubits y
            dos capas.
        seed:
            Semilla del estimador de estado.

    Returns:
        Interfaz EstimatorQNN que produce cuatro valores esperados.
    """
    model = (
        variational_circuit
        if variational_circuit is not None
        else build_variational_circuit()
    )

    observables = build_pauli_z_observables(
        number_of_qubits=model.circuit.num_qubits
    )

    estimator = StatevectorEstimator(seed=seed)

    return EstimatorQNN(
        circuit=model.circuit,
        estimator=estimator,
        observables=observables,
        input_params=model.input_parameters,
        weight_params=model.weight_parameters,
        input_gradients=True,
        default_precision=0.0,
    )