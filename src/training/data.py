"""Preparación de los datos para el entrenamiento con PyTorch."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import PreparedDataset


ClassificationTask = Literal["binary", "multiclass"]


@dataclass(frozen=True)
class TrainingDataLoaders:
    """Cargadores de entrenamiento, validación y prueba."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader


def _create_tensor_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    task: ClassificationTask,
) -> TensorDataset:
    """Convierte características y etiquetas en tensores."""
    feature_tensor = torch.as_tensor(
        features,
        dtype=torch.float32,
    )

    if task == "binary":
        target_tensor = torch.as_tensor(
            targets,
            dtype=torch.float32,
        ).reshape(-1, 1)

    elif task == "multiclass":
        target_tensor = torch.as_tensor(
            targets,
            dtype=torch.int64,
        ).reshape(-1)

    else:
        raise ValueError(
            f"Tipo de clasificación no reconocido: {task}"
        )

    if len(feature_tensor) != len(target_tensor):
        raise ValueError(
            "Las características y las etiquetas deben contener "
            "el mismo número de muestras."
        )

    return TensorDataset(
        feature_tensor,
        target_tensor,
    )


def create_training_data_loaders(
    prepared_dataset: PreparedDataset,
    *,
    task: ClassificationTask,
    batch_size: int = 16,
    seed: int = 42,
) -> TrainingDataLoaders:
    """
    Crea los cargadores utilizados durante el entrenamiento.

    La rama cuántica utiliza las cuatro características escaladas
    al intervalo angular [0, pi].
    """
    if batch_size < 1:
        raise ValueError(
            "batch_size debe ser mayor que cero."
        )

    train_dataset = _create_tensor_dataset(
        prepared_dataset.X_train_quantum,
        prepared_dataset.y_train,
        task=task,
    )

    validation_dataset = _create_tensor_dataset(
        prepared_dataset.X_validation_quantum,
        prepared_dataset.y_validation,
        task=task,
    )

    test_dataset = _create_tensor_dataset(
        prepared_dataset.X_test_quantum,
        prepared_dataset.y_test,
        task=task,
    )

    random_generator = torch.Generator()
    random_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=random_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return TrainingDataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
    )