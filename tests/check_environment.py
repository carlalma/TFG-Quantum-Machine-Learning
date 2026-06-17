"""Comprobación mínima del entorno local de Qiskit."""

import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def main() -> None:
    """Construye y ejecuta un circuito de prueba de un qubit."""
    circuit = QuantumCircuit(1, 1)

    # Crea una superposición equiprobable.
    circuit.h(0)
    circuit.measure(0, 0)

    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)

    result = simulator.run(
        compiled_circuit,
        shots=1_000,
        seed_simulator=42,
    ).result()

    counts = result.get_counts()

    print(f"Qiskit: {qiskit.__version__}")
    print(f"Qiskit Aer: {qiskit_aer.__version__}")
    print(f"Resultados: {counts}")


if __name__ == "__main__":
    main()