"""Pruebas de los cargadores de datos."""

from pathlib import Path

from src.data import load_breast_cancer, load_wine_quality


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BREAST_CANCER_PATH = (
    PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"
)

WINE_QUALITY_PATH = (
    PROJECT_ROOT / "data" / "raw" / "wine_quality.csv"
)


def test_breast_cancer_loader() -> None:
    """Comprueba la limpieza de Breast Cancer Wisconsin."""
    X, y = load_breast_cancer(BREAST_CANCER_PATH)

    assert len(X) == len(y)
    assert X.shape[1] == 30
    assert set(y.unique()) == {0, 1}
    assert "id" not in X.columns
    assert "Unnamed: 32" not in X.columns
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0


def test_wine_quality_loader() -> None:
    """Comprueba la limpieza y filtrado de Wine Quality."""
    X, y = load_wine_quality(WINE_QUALITY_PATH)

    assert len(X) == len(y)
    assert X.shape[1] == 11
    assert set(y.unique()) == {0, 1, 2}
    assert "Id" not in X.columns
    assert "id" not in X.columns
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0