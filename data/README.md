# Datasets

Los archivos CSV utilizados por el proyecto no se almacenan en el repositorio.

## Breast Cancer Wisconsin

- Fuente: Kaggle https://www.kaggle.com/datasets/mragpavank/breast-cancer
- Archivo local: `data/raw/breast_cancer.csv`
- Variable objetivo: `diagnosis`
- Tipo de problema: clasificación binaria
- Columnas eliminadas durante la limpieza: `id` y `Unnamed: 32`

## Wine Quality

- Fuente: Kaggle  https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
- Archivo local: `data/raw/wine_quality.csv`
- Variable objetivo: `quality`
- Tipo de problema: clasificación multiclase
- Clases utilizadas: 5, 6 y 7
- Columna eliminada durante la limpieza: `Id`

## Características seleccionadas

La selección se ha realizado mediante `SelectKBest` con `f_classif`,
ajustado exclusivamente sobre el conjunto de entrenamiento y con
`random_state=42` para la división de los datos.

### Breast Cancer Wisconsin

1. 'concave points_mean'
2. 'radius_worst'
3. 'perimeter_worst'
4. 'concave points_worst'

### Wine Quality

1. 'volatile acidity'
2. 'citric acid'
3. 'sulphates'
4. 'alcohol'