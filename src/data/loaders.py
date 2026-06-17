"""Carga y limpieza inicial de los datasets del proyecto."""

from pathlib import Path

import pandas as pd


class DatasetSchemaError(ValueError):
    """Indica que un dataset no contiene la estructura esperada."""


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV y comprueba que el archivo existe."""
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"El archivo debe tener extensión CSV: {csv_path}")

    return pd.read_csv(csv_path)


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    """Comprueba que estén presentes las columnas obligatorias."""
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise DatasetSchemaError(
            f"{dataset_name} no contiene las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )


def load_breast_cancer(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga y prepara Breast Cancer Wisconsin.

    Devuelve:
        X: treinta características predictoras numéricas.
        y: etiquetas binarias, donde 0 = benigno y 1 = maligno.
    """
    dataframe = _read_csv(path)

    _validate_columns(
        dataframe=dataframe,
        required_columns={"diagnosis"},
        dataset_name="Breast Cancer Wisconsin",
    )

    columns_to_drop = [
        column
        for column in ("id", "Unnamed: 32")
        if column in dataframe.columns
    ]

    dataframe = dataframe.drop(columns=columns_to_drop)

    y = dataframe.pop("diagnosis").map(
        {
            "B": 0,
            "M": 1,
        }
    )

    if y.isna().any():
        invalid_labels = dataframe.loc[
            y.isna()
        ].index.tolist()

        raise DatasetSchemaError(
            "Breast Cancer contiene etiquetas distintas de 'B' y 'M'. "
            f"Filas afectadas: {invalid_labels[:10]}"
        )

    try:
        X = dataframe.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError(
            "Breast Cancer contiene características no numéricas."
        ) from error

    return X, y.astype("int64").rename("diagnosis")


def load_wine_quality(
    path: str | Path,
    selected_classes: tuple[int, ...] = (5, 6, 7),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga y prepara Wine Quality.

    Solo conserva las puntuaciones indicadas en selected_classes.

    Devuelve:
        X: once características fisicoquímicas.
        y: etiquetas consecutivas 0, 1 y 2.
    """
    dataframe = _read_csv(path)

    _validate_columns(
        dataframe=dataframe,
        required_columns={"quality"},
        dataset_name="Wine Quality",
    )

    columns_to_drop = [
        column
        for column in ("Id", "id")
        if column in dataframe.columns
    ]

    dataframe = dataframe.drop(columns=columns_to_drop)

    dataframe = dataframe[
        dataframe["quality"].isin(selected_classes)
    ].copy()

    if dataframe.empty:
        raise DatasetSchemaError(
            "No se encontraron muestras para las clases "
            f"{selected_classes}."
        )

    label_mapping = {
        original_label: encoded_label
        for encoded_label, original_label in enumerate(selected_classes)
    }

    y = dataframe.pop("quality").map(label_mapping)

    if y.isna().any():
        raise DatasetSchemaError(
            "No se pudieron transformar correctamente las clases de Wine."
        )

    try:
        X = dataframe.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise DatasetSchemaError(
            "Wine Quality contiene características no numéricas."
        ) from error

    return X, y.astype("int64").rename("quality")

