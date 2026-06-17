"""Modelos híbridos cuántico-clásicos del proyecto."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from qiskit_machine_learning.connectors import TorchConnector

from src.quantum import build_quantum_readout


class HybridClassifier(nn.Module):
    """
    Clasificador híbrido formado por un VQC y una capa lineal clásica.

    El circuito cuántico recibe cuatro características y devuelve cuatro
    valores esperados de Pauli Z. La capa clásica transforma estos valores
    en uno o varios logits según el problema de clasificación.
    """

    def __init__(
        self,
        *,
        number_of_classes: int,
        seed: int = 42,
    ) -> None:
        """
        Inicializa el modelo híbrido.

        Args:
            number_of_classes:
                Número de clases del problema. Para clasificación binaria
                se emplea una única salida. Para clasificación multiclase
                se utiliza una salida por clase.
            seed:
                Semilla utilizada para inicializar los parámetros
                cuánticos y clásicos.

        Raises:
            ValueError: Si hay menos de dos clases.
        """
        super().__init__()

        if number_of_classes < 2:
            raise ValueError(
                "El número de clases debe ser al menos dos."
            )

        self.number_of_classes = number_of_classes
        self.seed = seed

        quantum_readout = build_quantum_readout(seed=seed)

        random_generator = np.random.default_rng(seed)

        initial_quantum_weights = random_generator.uniform(
            low=-0.1,
            high=0.1,
            size=quantum_readout.num_weights,
        ).astype(np.float32)

        self.quantum_layer = TorchConnector(
            quantum_readout,
            initial_weights=initial_quantum_weights,
        )

        quantum_output_dimension = quantum_readout.output_shape[0]

        classical_output_dimension = (
            1
            if number_of_classes == 2
            else number_of_classes
        )

        # fork_rng permite inicializar la capa de forma reproducible
        # sin alterar permanentemente el estado aleatorio global.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)

            self.classical_layer = nn.Linear(
                in_features=quantum_output_dimension,
                out_features=classical_output_dimension,
                bias=True,
            )

    @property
    def number_of_quantum_parameters(self) -> int:
        """Devuelve el número de parámetros entrenables cuánticos."""
        return int(self.quantum_layer.weight.numel())

    @property
    def number_of_classical_parameters(self) -> int:
        """Devuelve el número de parámetros de la capa clásica."""
        return sum(
            parameter.numel()
            for parameter in self.classical_layer.parameters()
            if parameter.requires_grad
        )

    @property
    def number_of_trainable_parameters(self) -> int:
        """Devuelve el número total de parámetros entrenables."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def extract_quantum_features(
        self,
        input_data: Tensor,
    ) -> Tensor:
        """
        Ejecuta el circuito y devuelve sus cuatro valores esperados.

        Args:
            input_data:
                Tensor de forma (muestras, 4) o una única muestra
                de forma (4,).

        Returns:
            Tensor con los cuatro valores esperados por muestra.
        """
        if input_data.shape[-1] != 4:
            raise ValueError(
                "El modelo híbrido necesita exactamente "
                "cuatro características de entrada."
            )

        quantum_features = self.quantum_layer(input_data)

        # Se mantiene la compatibilidad con el tipo numérico
        # utilizado por la capa lineal de PyTorch.
        return quantum_features.to(
            dtype=self.classical_layer.weight.dtype
        )

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Ejecuta el modelo y devuelve logits sin normalizar.

        Las funciones sigmoide y softmax no se aplican durante el
        entrenamiento porque se incorporan en las funciones de pérdida.
        """
        quantum_features = self.extract_quantum_features(input_data)

        return self.classical_layer(quantum_features)

    def predict_probabilities(
        self,
        input_data: Tensor,
    ) -> Tensor:
        """Convierte los logits del modelo en probabilidades."""
        logits = self.forward(input_data)

        if self.number_of_classes == 2:
            return torch.sigmoid(logits)

        return torch.softmax(logits, dim=-1)

    def predict(self, input_data: Tensor) -> Tensor:
        """Obtiene las etiquetas predichas por el modelo."""
        probabilities = self.predict_probabilities(input_data)

        if self.number_of_classes == 2:
            return (
                probabilities.squeeze(-1) >= 0.5
            ).to(dtype=torch.int64)

        return probabilities.argmax(dim=-1)


def build_binary_hybrid_model(
    *,
    seed: int = 42,
) -> HybridClassifier:
    """Construye el modelo híbrido para Breast Cancer Wisconsin."""
    return HybridClassifier(
        number_of_classes=2,
        seed=seed,
    )


def build_multiclass_hybrid_model(
    *,
    seed: int = 42,
) -> HybridClassifier:
    """Construye el modelo híbrido para Wine Quality."""
    return HybridClassifier(
        number_of_classes=3,
        seed=seed,
    )
