"""Entrenamiento de los clasificadores híbridos."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import isfinite

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data import PreparedDataset

from .config import TrainingConfig
from .data import (
    ClassificationTask,
    TrainingDataLoaders,
    create_training_data_loaders,
)


@dataclass
class TrainingHistory:
    """Evolución de las pérdidas durante el entrenamiento."""

    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingResult:
    """Resumen del proceso de entrenamiento."""

    history: TrainingHistory
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    stopped_early: bool


def compute_balanced_class_weights(
    targets: np.ndarray,
    *,
    number_of_classes: int,
) -> np.ndarray:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de clase.

    Los pesos se calculan únicamente con las etiquetas del conjunto
    de entrenamiento.
    """
    target_array = np.asarray(
        targets,
        dtype=np.int64,
    ).reshape(-1)

    if target_array.size == 0:
        raise ValueError(
            "No pueden calcularse pesos con un conjunto vacío."
        )

    if number_of_classes < 2:
        raise ValueError(
            "number_of_classes debe ser al menos dos."
        )

    class_counts = np.bincount(
        target_array,
        minlength=number_of_classes,
    )

    if len(class_counts) != number_of_classes:
        raise ValueError(
            "Las etiquetas contienen clases fuera del rango esperado."
        )

    if np.any(class_counts == 0):
        raise ValueError(
            "Todas las clases deben estar presentes en entrenamiento."
        )

    number_of_samples = target_array.size

    weights = number_of_samples / (
        number_of_classes * class_counts
    )

    return weights.astype(np.float32)


def build_loss_function(
    *,
    task: ClassificationTask,
    training_targets: np.ndarray,
    number_of_classes: int,
) -> nn.Module:
    """Construye la función de pérdida correspondiente al problema."""
    if task == "binary":
        return nn.BCEWithLogitsLoss()

    if task == "multiclass":
        class_weights = compute_balanced_class_weights(
            training_targets,
            number_of_classes=number_of_classes,
        )

        return nn.CrossEntropyLoss(
            weight=torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )
        )

    raise ValueError(
        f"Tipo de clasificación no reconocido: {task}"
    )


def _execute_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    *,
    optimizer: Adam | None,
) -> float:
    """
    Ejecuta una época de entrenamiento o validación.

    La presencia del optimizador indica que deben calcularse gradientes
    y actualizarse los parámetros.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    accumulated_loss = 0.0
    processed_samples = 0

    for input_batch, target_batch in data_loader:
        batch_size = input_batch.shape[0]

        if is_training:
            optimizer.zero_grad(set_to_none=True)

            logits = model(input_batch)
            loss = loss_function(logits, target_batch)

            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                logits = model(input_batch)
                loss = loss_function(logits, target_batch)

        loss_value = float(loss.detach().item())

        if not isfinite(loss_value):
            raise RuntimeError(
                "El entrenamiento ha producido una pérdida no finita."
            )

        accumulated_loss += loss_value * batch_size
        processed_samples += batch_size

    if processed_samples == 0:
        raise ValueError(
            "El cargador de datos no contiene ninguna muestra."
        )

    return accumulated_loss / processed_samples


def train_classifier(
    model: nn.Module,
    data_loaders: TrainingDataLoaders,
    loss_function: nn.Module,
    *,
    config: TrainingConfig,
) -> TrainingResult:
    """
    Entrena un modelo con Adam, validación y parada temprana.

    Al finalizar se restaura el estado correspondiente a la menor
    pérdida de validación observada.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    history = TrainingHistory()

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None

    epochs_without_improvement = 0
    stopped_early = False

    for epoch_index in range(config.maximum_epochs):
        training_loss = _execute_epoch(
            model,
            data_loaders.train,
            loss_function,
            optimizer=optimizer,
        )

        validation_loss = _execute_epoch(
            model,
            data_loaders.validation,
            loss_function,
            optimizer=None,
        )

        history.training_loss.append(training_loss)
        history.validation_loss.append(validation_loss)

        improvement = (
            best_validation_loss - validation_loss
        )

        if improvement > config.minimum_improvement:
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        current_epoch = epoch_index + 1

        if config.verbose and (
            current_epoch == 1
            or current_epoch % config.report_every == 0
            or current_epoch == config.maximum_epochs
        ):
            print(
                f"Epoch {current_epoch:03d}/"
                f"{config.maximum_epochs:03d} | "
                f"train_loss={training_loss:.6f} | "
                f"validation_loss={validation_loss:.6f} | "
                f"best_validation={best_validation_loss:.6f}",
                flush=True,
            )

        if epochs_without_improvement >= config.patience:
            stopped_early = True

            if config.verbose:
                print(
                    "Early stopping activated at epoch "
                    f"{current_epoch}. Best epoch: {best_epoch}.",
                    flush=True,
                )

            break

    if best_model_state is None:
        raise RuntimeError(
            "No se ha podido guardar ningún estado válido del modelo."
        )

    # Se recuperan los parámetros de la mejor época, no los de la última.
    model.load_state_dict(best_model_state)

    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=len(history.training_loss),
        stopped_early=stopped_early,
    )


def fit_classifier(
    model: nn.Module,
    prepared_dataset: PreparedDataset,
    *,
    config: TrainingConfig | None = None,
) -> tuple[TrainingResult, TrainingDataLoaders]:
    """
    Prepara los datos y entrena un clasificador binario o multiclase.

    El modelo debe exponer el atributo number_of_classes.
    """
    training_config = (
        config
        if config is not None
        else TrainingConfig()
    )

    number_of_classes = getattr(
        model,
        "number_of_classes",
        None,
    )

    if number_of_classes is None:
        raise AttributeError(
            "El modelo debe definir number_of_classes."
        )

    task: ClassificationTask = (
        "binary"
        if number_of_classes == 2
        else "multiclass"
    )

    data_loaders = create_training_data_loaders(
        prepared_dataset,
        task=task,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
    )

    loss_function = build_loss_function(
        task=task,
        training_targets=prepared_dataset.y_train,
        number_of_classes=number_of_classes,
    )

    result = train_classifier(
        model,
        data_loaders,
        loss_function,
        config=training_config,
    )

    return result, data_loaders
