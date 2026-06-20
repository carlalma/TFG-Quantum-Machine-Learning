"""Pruebas del sistema de entrenamiento."""

import numpy as np
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn

from src.data import PreparedDataset
from src.training import (
    TrainingConfig,
    build_loss_function,
    compute_balanced_class_weights,
    create_training_data_loaders,
    fit_classifier,
)


class TinyBinaryClassifier(nn.Module):
    """Modelo ligero para comprobar el entrenamiento binario."""

    number_of_classes = 2

    def __init__(self) -> None:
        super().__init__()

        torch.manual_seed(42)
        self.linear = nn.Linear(4, 1)

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(input_data)


class TinyMulticlassClassifier(nn.Module):
    """Modelo ligero para comprobar el entrenamiento multiclase."""

    number_of_classes = 3

    def __init__(self) -> None:
        super().__init__()

        torch.manual_seed(42)
        self.linear = nn.Linear(4, 3)

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(input_data)


def create_prepared_dataset(
    *,
    multiclass: bool,
) -> PreparedDataset:
    """Crea datos sintéticos con la estructura del pipeline real."""
    random_generator = np.random.default_rng(seed=42)

    number_of_samples = 120

    features = random_generator.uniform(
        low=0.0,
        high=np.pi,
        size=(number_of_samples, 4),
    )

    if multiclass:
        scores = np.column_stack(
            [
                features[:, 0],
                features[:, 1],
                features[:, 2] + features[:, 3],
            ]
        )

        targets = scores.argmax(axis=1).astype(np.int64)

    else:
        targets = (
            features[:, 0] + features[:, 1]
            > features[:, 2] + features[:, 3]
        ).astype(np.int64)

    return PreparedDataset(
        X_train_classical=features[:84],
        X_validation_classical=features[84:102],
        X_test_classical=features[102:],
        X_train_quantum=features[:84],
        X_validation_quantum=features[84:102],
        X_test_quantum=features[102:],
        y_train=targets[:84],
        y_validation=targets[84:102],
        y_test=targets[102:],
        selected_feature_names=[
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        feature_selector=SelectKBest(k=4),
        classical_scaler=StandardScaler(),
        quantum_scaler=MinMaxScaler(),
    )


def test_balanced_class_weights() -> None:
    """Comprueba el cálculo inverso de frecuencias."""
    targets = np.array(
        [0, 0, 0, 1, 1, 2],
        dtype=np.int64,
    )

    weights = compute_balanced_class_weights(
        targets,
        number_of_classes=3,
    )

    np.testing.assert_allclose(
        weights,
        np.array(
            [2 / 3, 1.0, 2.0],
            dtype=np.float32,
        ),
    )


def test_binary_data_loaders_use_expected_types() -> None:
    """Comprueba las dimensiones de los lotes binarios."""
    prepared = create_prepared_dataset(
        multiclass=False
    )

    data_loaders = create_training_data_loaders(
        prepared,
        task="binary",
        batch_size=16,
        seed=42,
    )

    input_batch, target_batch = next(
        iter(data_loaders.train)
    )

    assert input_batch.shape[1] == 4
    assert target_batch.shape[1] == 1
    assert input_batch.dtype == torch.float32
    assert target_batch.dtype == torch.float32


def test_multiclass_data_loaders_use_expected_types() -> None:
    """Comprueba las dimensiones de los lotes multiclase."""
    prepared = create_prepared_dataset(
        multiclass=True
    )

    data_loaders = create_training_data_loaders(
        prepared,
        task="multiclass",
        batch_size=16,
        seed=42,
    )

    input_batch, target_batch = next(
        iter(data_loaders.train)
    )

    assert input_batch.shape[1] == 4
    assert target_batch.ndim == 1
    assert input_batch.dtype == torch.float32
    assert target_batch.dtype == torch.int64


def test_binary_training_produces_valid_history() -> None:
    """Comprueba el ciclo completo de entrenamiento binario."""
    prepared = create_prepared_dataset(
        multiclass=False
    )

    model = TinyBinaryClassifier()

    config = TrainingConfig(
        maximum_epochs=10,
        batch_size=16,
        learning_rate=0.05,
        patience=4,
        seed=42,
    )

    result, _ = fit_classifier(
        model,
        prepared,
        config=config,
    )

    assert 1 <= result.epochs_completed <= 10
    assert 1 <= result.best_epoch <= result.epochs_completed
    assert np.isfinite(result.best_validation_loss)

    assert len(result.history.training_loss) == (
        result.epochs_completed
    )

    assert len(result.history.validation_loss) == (
        result.epochs_completed
    )

    assert result.best_validation_loss == min(
        result.history.validation_loss
    )


def test_multiclass_training_uses_weighted_loss() -> None:
    """Comprueba el entrenamiento multiclase ponderado."""
    prepared = create_prepared_dataset(
        multiclass=True
    )

    loss_function = build_loss_function(
        task="multiclass",
        training_targets=prepared.y_train,
        number_of_classes=3,
    )

    assert isinstance(
        loss_function,
        nn.CrossEntropyLoss,
    )

    assert loss_function.weight is not None
    assert loss_function.weight.shape == (3,)

    model = TinyMulticlassClassifier()

    config = TrainingConfig(
        maximum_epochs=5,
        batch_size=16,
        learning_rate=0.05,
        patience=3,
        seed=42,
    )

    result, _ = fit_classifier(
        model,
        prepared,
        config=config,
    )

    assert 1 <= result.epochs_completed <= 5
    assert np.isfinite(result.best_validation_loss)

def test_data_loaders_can_use_classical_features() -> None:
        """Comprueba la selección de la rama clásica."""
        prepared = create_prepared_dataset(
            multiclass=False
        )

        prepared.X_train_classical = np.full(
            prepared.X_train_classical.shape,
            fill_value=-2.0,
            dtype=np.float64,
        )

        prepared.X_train_quantum = np.full(
            prepared.X_train_quantum.shape,
            fill_value=2.0,
            dtype=np.float64,
        )

        data_loaders = create_training_data_loaders(
            prepared,
            task="binary",
            batch_size=16,
            seed=42,
            feature_representation="classical",
        )

        input_batch, _ = next(
            iter(data_loaders.train)
        )

        assert torch.all(input_batch == -2.0)