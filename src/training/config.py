"""Configuración del proceso de entrenamiento."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparámetros comunes de los experimentos híbridos."""

    maximum_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    patience: int = 15
    minimum_improvement: float = 1e-4
    seed: int = 42

    """Aadir progreso visible """
    verbose: bool = False
    report_every: int = 1

    def __post_init__(self) -> None:
        """Valida los valores de configuración."""
        if self.maximum_epochs < 1:
            raise ValueError(
                "maximum_epochs debe ser mayor que cero."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size debe ser mayor que cero."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate debe ser positivo."
            )

        if self.patience < 1:
            raise ValueError(
                "patience debe ser mayor que cero."
            )

        if self.minimum_improvement < 0:
            raise ValueError(
                "minimum_improvement no puede ser negativo."
            )
        if self.report_every < 1:
            raise ValueError(
                "report_every debe ser mayor que cero."
            )