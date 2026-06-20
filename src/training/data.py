"""Preparación de los datos para el entrenamiento con PyTorch."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import PreparedDataset


ClassificationTask = Literal["binary", "multiclass"] 
FeatureRepresentation = Literal[
    "quantum",
    "classical",
]


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
    feature_array = np.array(
        features,
        dtype=np.float32,
        copy=True,
    )

    feature_tensor = torch.from_numpy(
        feature_array
    )

    if task == "binary":
        target_array = np.array(
            targets,
            dtype=np.float32,
            copy=True,
        ).reshape(-1, 1)

    elif task == "multiclass":
        target_array = np.array(
            targets,
            dtype=np.int64,
            copy=True,
        ).reshape(-1)

    else:
        raise ValueError(
            f"Tipo de clasificación no reconocido: {task}"
        )

    target_tensor = torch.from_numpy(
        target_array
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
    feature_representation: FeatureRepresentation = "quantum",
) -> TrainingDataLoaders:
    """
    Crea los cargadores utilizados durante el entrenamiento.

    La representación cuántica utiliza características escaladas
    a [0, pi]. La representación clásica utiliza características
    estandarizadas.
    """
    if batch_size < 1:
        raise ValueError(
            "batch_size debe ser mayor que cero."
        )

    if feature_representation == "quantum":
        training_features = (
            prepared_dataset.X_train_quantum
        )

        validation_features = (
            prepared_dataset.X_validation_quantum
        )

        test_features = (
            prepared_dataset.X_test_quantum
        )

    elif feature_representation == "classical":
        training_features = (
            prepared_dataset.X_train_classical
        )

        validation_features = (
            prepared_dataset.X_validation_classical
        )

        test_features = (
            prepared_dataset.X_test_classical
        )

    else:
        raise ValueError(
            "La representación debe ser "
            "'quantum' o 'classical'."
        )

    train_dataset = _create_tensor_dataset(
        training_features,
        prepared_dataset.y_train,
        task=task,
    )

    validation_dataset = _create_tensor_dataset(
        validation_features,
        prepared_dataset.y_validation,
        task=task,
    )

    test_dataset = _create_tensor_dataset(
        test_features,
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