"""Herramientas para el entrenamiento de los modelos."""

from .config import TrainingConfig
from .data import (
    ClassificationTask,
    TrainingDataLoaders,
    FeatureRepresentation,
    create_training_data_loaders,
)
from .trainer import (
    TrainingHistory,
    TrainingResult,
    build_loss_function,
    compute_balanced_class_weights,
    fit_classifier,
    train_classifier,
)

__all__ = [
    "ClassificationTask",
    "TrainingConfig",
    "TrainingDataLoaders",
    "TrainingHistory",
    "TrainingResult",
    "build_loss_function",
    "compute_balanced_class_weights",
    "create_training_data_loaders",
    "fit_classifier",
    "train_classifier",
    "FeatureRepresentation",
]