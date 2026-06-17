
"""División y preprocesamiento de los datasets del proyecto."""

from dataclasses import dataclass
from math import pi

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class DatasetSplits:
    """Particiones originales de un conjunto de datos."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


@dataclass
class PreparedDataset:
    """Datos preparados para las ramas clásica y cuántica."""

    X_train_classical: np.ndarray
    X_validation_classical: np.ndarray
    X_test_classical: np.ndarray

    X_train_quantum: np.ndarray
    X_validation_quantum: np.ndarray
    X_test_quantum: np.ndarray

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray

    selected_feature_names: list[str]

    feature_selector: SelectKBest
    classical_scaler: StandardScaler
    quantum_scaler: MinMaxScaler

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """
    Divide un dataset de forma estratificada.

    La configuración predeterminada produce aproximadamente:
    - 70 % entrenamiento.
    - 15 % validación.
    - 15 % prueba.
    """
    if len(X) != len(y):
        raise ValueError("X e y deben contener el mismo número de muestras.")

    if test_size <= 0 or validation_size <= 0:
        raise ValueError(
            "Los tamaños de validación y prueba deben ser positivos."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "La suma de validation_size y test_size debe ser menor que 1."
        )

    number_of_samples = len(X)

    # Se calculan cantidades enteras para evitar redondeos distintos
    # durante las dos llamadas a train_test_split.
    number_of_test_samples = round(number_of_samples * test_size)
    number_of_validation_samples = round(
        number_of_samples * validation_size
    )

    number_of_train_samples = (
        number_of_samples
        - number_of_test_samples
        - number_of_validation_samples
    )

    if number_of_train_samples <= 0:
        raise ValueError(
            "La configuración no deja suficientes muestras "
            "para el conjunto de entrenamiento."
        )

    # Primera división: se separa el conjunto de prueba.
    X_train_validation, X_test, y_train_validation, y_test = (
        train_test_split(
            X,
            y,
            test_size=number_of_test_samples,
            stratify=y,
            random_state=random_state,
        )
    )

    # Segunda división: se extrae un número exacto de muestras
    # para validación del conjunto restante.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=number_of_validation_samples,
        stratify=y_train_validation,
        random_state=random_state,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_validation=X_validation.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_validation=y_validation.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )

def prepare_features(
    splits: DatasetSplits,
    *,
    number_of_features: int = 4,
) -> PreparedDataset:
    """
    Selecciona las mejores características y prepara ambas ramas.

    Rama clásica:
        Estandarización con media cero y desviación típica unitaria.

    Rama cuántica:
        Escalado angular al intervalo [0, pi].
    """
    if number_of_features <= 0:
        raise ValueError("number_of_features debe ser mayor que cero.")

    if number_of_features > splits.X_train.shape[1]:
        raise ValueError(
            "No pueden seleccionarse más características de las disponibles."
        )

    # El selector se ajusta únicamente con entrenamiento.
    feature_selector = SelectKBest(
        score_func=f_classif,
        k=number_of_features,
    )

    X_train_selected = feature_selector.fit_transform(
        splits.X_train,
        splits.y_train,
    )

    X_validation_selected = feature_selector.transform(
        splits.X_validation
    )

    X_test_selected = feature_selector.transform(splits.X_test)

    selected_feature_names = (
        splits.X_train.columns[
            feature_selector.get_support()
        ]
        .tolist()
    )

    # Rama utilizada por regresión logística, SVM y red neuronal clásica.
    classical_scaler = StandardScaler()

    X_train_classical = classical_scaler.fit_transform(
        X_train_selected
    )

    X_validation_classical = classical_scaler.transform(
        X_validation_selected
    )

    X_test_classical = classical_scaler.transform(
        X_test_selected
    )

    # Rama utilizada como entrada del circuito cuántico.
    # clip=True mantiene validación y prueba dentro de [0, pi].
    quantum_scaler = MinMaxScaler(
        feature_range=(0.0, pi),
        clip=True,
    )

    X_train_quantum = quantum_scaler.fit_transform(
        X_train_selected
    )

    X_validation_quantum = quantum_scaler.transform(
        X_validation_selected
    )

    X_test_quantum = quantum_scaler.transform(
        X_test_selected
    )

    return PreparedDataset(
        X_train_classical=X_train_classical.astype(np.float64),
        X_validation_classical=X_validation_classical.astype(np.float64),
        X_test_classical=X_test_classical.astype(np.float64),
        X_train_quantum=X_train_quantum.astype(np.float64),
        X_validation_quantum=X_validation_quantum.astype(np.float64),
        X_test_quantum=X_test_quantum.astype(np.float64),
        y_train=splits.y_train.to_numpy(dtype=np.int64),
        y_validation=splits.y_validation.to_numpy(dtype=np.int64),
        y_test=splits.y_test.to_numpy(dtype=np.int64),
        selected_feature_names=selected_feature_names,
        feature_selector=feature_selector,
        classical_scaler=classical_scaler,
        quantum_scaler=quantum_scaler,
    )


def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    number_of_features: int = 4,
    random_state: int = 42,
) -> PreparedDataset:
    """Ejecuta la división y la preparación completa de un dataset."""
    splits = split_dataset(
        X=X,
        y=y,
        random_state=random_state,
    )

    return prepare_features(
        splits=splits,
        number_of_features=number_of_features,
    )