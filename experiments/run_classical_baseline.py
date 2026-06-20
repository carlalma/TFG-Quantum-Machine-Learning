"""Ejecución de los modelos clásicos de referencia."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import (
    PreparedDataset,
    load_breast_cancer,
    load_wine_quality,
    preprocess_dataset,
)
from src.evaluation import (
    ClassicalEvaluationResult,
    MulticlassEvaluationResult,
    BinaryEvaluationResult,
    evaluate_binary_classifier,
    evaluate_classical_binary_classifier,
    evaluate_classical_multiclass_classifier,
    evaluate_multiclass_classifier,
    plot_confusion_matrix,
    plot_training_history,
    save_experiment_summary,
    save_training_history,
)
from src.models import (
    build_binary_classical_mlp,
    build_logistic_regression,
    build_multiclass_classical_mlp,
    build_rbf_svm,
)
from src.training import TrainingConfig, fit_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = (
    "breast_cancer",
    "wine_quality",
)

MODEL_NAMES = (
    "logistic_regression",
    "rbf_svm",
    "mlp",
)

WINE_LABEL_MAPPING = {
    0: 5,
    1: 6,
    2: 7,
}


def get_package_version(package_name: str) -> str:
    """Devuelve la versión instalada de una dependencia."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def parse_arguments() -> Namespace:
    """Procesa los argumentos de la ejecución."""
    parser = ArgumentParser(
        description=(
            "Train and evaluate a classical baseline "
            "using the same data partitions as the hybrid model."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Model initialization and training seed. "
            "It mainly affects the MLP."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed seed used for the dataset partitions.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs used by the MLP.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size used by the MLP.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Adam learning rate used by the MLP.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience used by the MLP.",
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="Epoch interval between MLP progress messages.",
    )

    return parser.parse_args()


def load_prepared_dataset(
    dataset_name: str,
    *,
    split_seed: int,
) -> tuple[PreparedDataset, int]:
    """Carga y prepara uno de los dos datasets."""
    if dataset_name == "breast_cancer":
        dataset_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "breast_cancer.csv"
        )

        features, targets = load_breast_cancer(
            dataset_path
        )

    elif dataset_name == "wine_quality":
        dataset_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "wine_quality.csv"
        )

        features, targets = load_wine_quality(
            dataset_path,
            selected_classes=(5, 6, 7),
        )

    else:
        raise ValueError(
            f"Dataset no reconocido: {dataset_name}"
        )

    prepared_dataset = preprocess_dataset(
        features,
        targets,
        number_of_features=4,
        random_state=split_seed,
    )

    return prepared_dataset, int(len(targets))


def save_sklearn_predictions(
    evaluation: ClassicalEvaluationResult,
    output_path: Path,
    *,
    dataset_name: str,
) -> None:
    """Guarda predicciones y puntuaciones de un estimador clásico."""
    table_data: dict[str, Any] = {
        "true_label": evaluation.true_labels,
        "predicted_label": evaluation.predicted_labels,
    }

    if dataset_name == "wine_quality":
        table_data["true_quality"] = [
            WINE_LABEL_MAPPING[int(label)]
            for label in evaluation.true_labels
        ]

        table_data["predicted_quality"] = [
            WINE_LABEL_MAPPING[int(label)]
            for label in evaluation.predicted_labels
        ]

    if evaluation.scores is not None:
        scores = np.asarray(
            evaluation.scores
        )

        if scores.ndim == 1:
            table_data[
                evaluation.score_type
                or "model_score"
            ] = scores

        elif scores.ndim == 2:
            if (
                dataset_name == "breast_cancer"
                and scores.shape[1] == 2
                and evaluation.score_type
                == "predict_proba"
            ):
                table_data[
                    "probability_malignant"
                ] = scores[:, 1]

            else:
                for column_index in range(
                    scores.shape[1]
                ):
                    table_data[
                        (
                            f"{evaluation.score_type}"
                            f"_class_{column_index}"
                        )
                    ] = scores[:, column_index]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(table_data).to_csv(
        output_path,
        index=False,
    )


def save_torch_predictions(
    evaluation: (
        BinaryEvaluationResult
        | MulticlassEvaluationResult
    ),
    output_path: Path,
    *,
    dataset_name: str,
) -> None:
    """Guarda las predicciones generadas por el MLP."""
    if dataset_name == "breast_cancer":
        if not isinstance(
            evaluation,
            BinaryEvaluationResult,
        ):
            raise TypeError(
                "Se esperaba una evaluación binaria."
            )

        predictions_table = pd.DataFrame(
            {
                "true_label": (
                    evaluation.true_labels
                ),
                "predicted_label": (
                    evaluation.predicted_labels
                ),
                "probability_malignant": (
                    evaluation.probabilities
                ),
            }
        )

    else:
        if not isinstance(
            evaluation,
            MulticlassEvaluationResult,
        ):
            raise TypeError(
                "Se esperaba una evaluación multiclase."
            )

        predictions_table = pd.DataFrame(
            {
                "true_encoded_label": (
                    evaluation.true_labels
                ),
                "predicted_encoded_label": (
                    evaluation.predicted_labels
                ),
                "true_quality": [
                    WINE_LABEL_MAPPING[int(label)]
                    for label in evaluation.true_labels
                ],
                "predicted_quality": [
                    WINE_LABEL_MAPPING[int(label)]
                    for label in evaluation.predicted_labels
                ],
                "probability_quality_5": (
                    evaluation.probabilities[:, 0]
                ),
                "probability_quality_6": (
                    evaluation.probabilities[:, 1]
                ),
                "probability_quality_7": (
                    evaluation.probabilities[:, 2]
                ),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_table.to_csv(
        output_path,
        index=False,
    )


def build_sklearn_metadata(
    model: Any,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Obtiene información específica del estimador entrenado."""
    if model_name == "logistic_regression":
        return {
            "coefficient_parameters": int(
                model.coef_.size
                + model.intercept_.size
            ),
            "iterations": [
                int(value)
                for value in model.n_iter_.tolist()
            ],
        }

    if model_name == "rbf_svm":
        return {
            "support_vectors": int(
                model.n_support_.sum()
            ),
            "support_vectors_per_class": [
                int(value)
                for value in model.n_support_.tolist()
            ],
            "fit_status": int(
                model.fit_status_
            ),
        }

    return {}


def run_sklearn_model(
    prepared_dataset: PreparedDataset,
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
) -> tuple[
    Any,
    ClassicalEvaluationResult,
    float,
    float,
]:
    """Entrena y evalúa regresión logística o SVM."""
    task = (
        "binary"
        if dataset_name == "breast_cancer"
        else "multiclass"
    )

    if model_name == "logistic_regression":
        model = build_logistic_regression(
            task=task,
            seed=seed,
        )

    elif model_name == "rbf_svm":
        model = build_rbf_svm(
            task=task,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Modelo no reconocido: {model_name}"
        )

    training_start = perf_counter()

    model.fit(
        prepared_dataset.X_train_classical,
        prepared_dataset.y_train,
    )

    training_time = (
        perf_counter() - training_start
    )

    evaluation_start = perf_counter()

    if dataset_name == "breast_cancer":
        evaluation = (
            evaluate_classical_binary_classifier(
                model,
                prepared_dataset.X_test_classical,
                prepared_dataset.y_test,
            )
        )

    else:
        evaluation = (
            evaluate_classical_multiclass_classifier(
                model,
                prepared_dataset.X_test_classical,
                prepared_dataset.y_test,
            )
        )

    evaluation_time = (
        perf_counter() - evaluation_start
    )

    return (
        model,
        evaluation,
        training_time,
        evaluation_time,
    )


def main() -> None:
    """Ejecuta un modelo clásico y guarda sus resultados."""
    arguments = parse_arguments()

    experiment_name = (
        f"{arguments.dataset}"
        f"_{arguments.model}"
        f"_split_{arguments.split_seed}"
        f"_seed_{arguments.seed}"
    )

    metrics_directory = (
        PROJECT_ROOT / "results" / "metrics"
    )

    figures_directory = (
        PROJECT_ROOT / "results" / "figures"
    )

    models_directory = (
        PROJECT_ROOT / "results" / "models"
    )

    for directory in (
        metrics_directory,
        figures_directory,
        models_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("\n" + "=" * 72)
    print("CLASSICAL BASELINE EXPERIMENT")
    print("=" * 72)
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {arguments.dataset}")
    print(f"Model: {arguments.model}")
    print(f"Split seed: {arguments.split_seed}")
    print(f"Training seed: {arguments.seed}")

    total_start = perf_counter()

    prepared_dataset, number_of_samples = (
        load_prepared_dataset(
            arguments.dataset,
            split_seed=arguments.split_seed,
        )
    )

    print(
        "Selected features:",
        prepared_dataset.selected_feature_names,
    )

    print(
        "Partitions:",
        {
            "training": len(
                prepared_dataset.y_train
            ),
            "validation": len(
                prepared_dataset.y_validation
            ),
            "test": len(
                prepared_dataset.y_test
            ),
        },
    )

    summary_path = (
        metrics_directory
        / f"{experiment_name}.json"
    )

    predictions_path = (
        metrics_directory
        / f"{experiment_name}_predictions.csv"
    )

    confusion_path = (
        figures_directory
        / f"{experiment_name}_confusion.png"
    )

    history_path: Path | None = None
    loss_figure_path: Path | None = None

    if arguments.model in (
        "logistic_regression",
        "rbf_svm",
    ):
        (
            model,
            evaluation,
            training_time,
            evaluation_time,
        ) = run_sklearn_model(
            prepared_dataset,
            dataset_name=arguments.dataset,
            model_name=arguments.model,
            seed=arguments.seed,
        )

        save_sklearn_predictions(
            evaluation,
            predictions_path,
            dataset_name=arguments.dataset,
        )

        model_metadata = build_sklearn_metadata(
            model,
            model_name=arguments.model,
        )

        training_result: dict[str, Any] = {
            "training_method": (
                "scikit-learn estimator"
            ),
            **model_metadata,
        }

        model_path = (
            models_directory
            / f"{experiment_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

    else:
        if arguments.dataset == "breast_cancer":
            model = build_binary_classical_mlp(
                seed=arguments.seed
            )

        else:
            model = (
                build_multiclass_classical_mlp(
                    seed=arguments.seed
                )
            )

        training_config = TrainingConfig(
            maximum_epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            patience=arguments.patience,
            minimum_improvement=1e-4,
            seed=arguments.seed,
            verbose=True,
            report_every=arguments.report_every,
        )

        training_start = perf_counter()

        result, data_loaders = fit_classifier(
            model,
            prepared_dataset,
            config=training_config,
            feature_representation="classical",
        )

        training_time = (
            perf_counter() - training_start
        )

        evaluation_start = perf_counter()

        if arguments.dataset == "breast_cancer":
            evaluation = evaluate_binary_classifier(
                model,
                data_loaders.test,
            )

        else:
            evaluation = (
                evaluate_multiclass_classifier(
                    model,
                    data_loaders.test,
                )
            )

        evaluation_time = (
            perf_counter() - evaluation_start
        )

        save_torch_predictions(
            evaluation,
            predictions_path,
            dataset_name=arguments.dataset,
        )

        history_path = (
            metrics_directory
            / f"{experiment_name}_history.csv"
        )

        loss_figure_path = (
            figures_directory
            / f"{experiment_name}_loss.png"
        )

        save_training_history(
            result.history,
            history_path,
        )

        plot_training_history(
            result.history,
            loss_figure_path,
        )

        training_result = {
            "training_method": "PyTorch and Adam",
            "configuration": asdict(
                training_config
            ),
            "best_epoch": (
                result.best_epoch
            ),
            "best_validation_loss": (
                result.best_validation_loss
            ),
            "epochs_completed": (
                result.epochs_completed
            ),
            "stopped_early": (
                result.stopped_early
            ),
        }

        model_metadata = {
            "trainable_parameters": (
                model.number_of_trainable_parameters
            ),
            "architecture": "4-4-output",
        }

        model_path = (
            models_directory
            / f"{experiment_name}.pt"
        )

        torch.save(
            {
                "model_state_dict": (
                    model.state_dict()
                ),
                "training_result": (
                    training_result
                ),
                "selected_features": (
                    prepared_dataset
                    .selected_feature_names
                ),
            },
            model_path,
        )

    if arguments.dataset == "breast_cancer":
        class_names = (
            "Benign",
            "Malignant",
        )

        per_class_metrics = None

    else:
        class_names = (
            "Quality 5",
            "Quality 6",
            "Quality 7",
        )

        per_class_metrics = (
            evaluation.per_class_metrics
        )

    plot_confusion_matrix(
        evaluation.confusion_matrix,
        confusion_path,
        class_names=class_names,
        title=(
            f"{arguments.dataset} "
            f"{arguments.model} confusion matrix"
        ),
    )

    total_time = (
        perf_counter() - total_start
    )

    summary = {
        "experiment": {
            "name": experiment_name,
            "dataset": arguments.dataset,
            "model": arguments.model,
            "task": (
                "binary"
                if arguments.dataset
                == "breast_cancer"
                else "multiclass"
            ),
        },
        "random_seeds": {
            "dataset_split": (
                arguments.split_seed
            ),
            "model_and_training": (
                arguments.seed
            ),
        },
        "dataset": {
            "samples": number_of_samples,
            "selected_features": (
                prepared_dataset
                .selected_feature_names
            ),
            "partitions": {
                "training": int(
                    len(
                        prepared_dataset
                        .y_train
                    )
                ),
                "validation": int(
                    len(
                        prepared_dataset
                        .y_validation
                    )
                ),
                "test": int(
                    len(
                        prepared_dataset
                        .y_test
                    )
                ),
            },
        },
        "model": model_metadata,
        "training_result": training_result,
        "test_results": {
            "metrics": evaluation.metrics,
            "per_class_metrics": (
                per_class_metrics
            ),
            "confusion_matrix": (
                evaluation
                .confusion_matrix
                .tolist()
            ),
        },
        "execution_times_seconds": {
            "training": training_time,
            "evaluation": evaluation_time,
            "total": total_time,
        },
        "software_environment": {
            "python": python_version(),
            "numpy": get_package_version(
                "numpy"
            ),
            "pandas": get_package_version(
                "pandas"
            ),
            "scikit_learn": (
                get_package_version(
                    "scikit-learn"
                )
            ),
            "torch": get_package_version(
                "torch"
            ),
        },
    }

    save_experiment_summary(
        summary,
        summary_path,
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETED")
    print("=" * 72)

    print(
        f"Training time: "
        f"{training_time:.6f} seconds"
    )

    print(
        f"Evaluation time: "
        f"{evaluation_time:.6f} seconds"
    )

    print("\nTest metrics:")

    for metric_name, metric_value in (
        evaluation.metrics.items()
    ):
        if isinstance(metric_value, float):
            print(
                f"  {metric_name}: "
                f"{metric_value:.6f}"
            )
        else:
            print(
                f"  {metric_name}: "
                f"{metric_value}"
            )

    if per_class_metrics is not None:
        print("\nPer-class metrics:")

        for class_name, class_metrics in (
            per_class_metrics.items()
        ):
            print(f"  {class_name}: {class_metrics}")

    print("\nConfusion matrix:")
    print(evaluation.confusion_matrix)

    print("\nGenerated files:")
    print(f"  Summary: {summary_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Confusion: {confusion_path}")
    print(f"  Model: {model_path}")

    if history_path is not None:
        print(f"  History: {history_path}")

    if loss_figure_path is not None:
        print(f"  Loss figure: {loss_figure_path}")


if __name__ == "__main__":
    main()