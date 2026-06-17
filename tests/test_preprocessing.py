"""Pruebas del pipeline de preprocesamiento."""

import numpy as np
import pandas as pd

from src.data import preprocess_dataset, split_dataset


def create_synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Genera un conjunto reproducible para probar el pipeline."""
    random_generator = np.random.default_rng(seed=42)

    number_of_samples = 200
    number_of_features = 8

    X = pd.DataFrame(
        random_generator.normal(
            size=(number_of_samples, number_of_features)
        ),
        columns=[
            f"feature_{index}"
            for index in range(number_of_features)
        ],
    )

    # Las cuatro primeras variables contienen señal de clasificación.
    score = (
        1.5 * X["feature_0"]
        - 1.2 * X["feature_1"]
        + 0.8 * X["feature_2"]
        + 0.5 * X["feature_3"]
    )

    y = pd.Series(
        (score > score.median()).astype(int),
        name="target",
    )

    return X, y


def test_split_dataset_preserves_all_samples() -> None:
    """Comprueba que las particiones no pierden muestras."""
    X, y = create_synthetic_dataset()

    splits = split_dataset(X, y, random_state=42)

    total_samples = (
        len(splits.X_train)
        + len(splits.X_validation)
        + len(splits.X_test)
    )

    assert total_samples == len(X)
    assert len(splits.X_train) == len(splits.y_train)
    assert len(splits.X_validation) == len(splits.y_validation)
    assert len(splits.X_test) == len(splits.y_test)


def test_split_dataset_uses_expected_proportions() -> None:
    """Comprueba aproximadamente la división 70/15/15."""
    X, y = create_synthetic_dataset()

    splits = split_dataset(X, y, random_state=42)

    assert len(splits.X_train) == 140
    assert len(splits.X_validation) == 30
    assert len(splits.X_test) == 30


def test_preprocessing_selects_four_features() -> None:
    """Comprueba la dimensión final de las entradas."""
    X, y = create_synthetic_dataset()

    prepared = preprocess_dataset(
        X,
        y,
        number_of_features=4,
        random_state=42,
    )

    assert prepared.X_train_classical.shape == (140, 4)
    assert prepared.X_validation_classical.shape == (30, 4)
    assert prepared.X_test_classical.shape == (30, 4)

    assert prepared.X_train_quantum.shape == (140, 4)
    assert prepared.X_validation_quantum.shape == (30, 4)
    assert prepared.X_test_quantum.shape == (30, 4)

    assert len(prepared.selected_feature_names) == 4


def test_quantum_features_are_in_angular_range() -> None:
    """Comprueba que los datos cuánticos pertenecen a [0, pi]."""
    X, y = create_synthetic_dataset()

    prepared = preprocess_dataset(
        X,
        y,
        number_of_features=4,
        random_state=42,
    )

    quantum_partitions = (
        prepared.X_train_quantum,
        prepared.X_validation_quantum,
        prepared.X_test_quantum,
    )

    for partition in quantum_partitions:
        assert np.all(partition >= 0.0)
        assert np.all(partition <= np.pi)


def test_preprocessing_is_reproducible() -> None:
    """Comprueba que una misma semilla produce el mismo resultado."""
    X, y = create_synthetic_dataset()

    first_result = preprocess_dataset(X, y, random_state=42)
    second_result = preprocess_dataset(X, y, random_state=42)

    np.testing.assert_array_equal(
        first_result.X_train_quantum,
        second_result.X_train_quantum,
    )

    assert (
        first_result.selected_feature_names
        == second_result.selected_feature_names
    )