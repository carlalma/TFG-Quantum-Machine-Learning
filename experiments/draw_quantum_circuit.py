"""Generación de la figura del circuito cuántico variacional."""

from pathlib import Path

import matplotlib.pyplot as plt

from src.quantum import build_variational_circuit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "figures"
    / "variational_quantum_circuit.png"
)


def main() -> None:
    """Genera y guarda el dibujo del circuito."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    variational_circuit = build_variational_circuit()

    figure = variational_circuit.circuit.draw(
        output="mpl",
        fold=-1,
        scale=0.8,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("Circuit figure generated:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()