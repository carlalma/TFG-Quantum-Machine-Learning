"""Experimento híbrido sobre Wine Quality."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from src.data import (
    load_wine_quality,
    preprocess_dataset,
)
from src.evaluation import (
    evaluate_multiclass_classifier,
    plot_confusion_matrix,
    plot_training_history,
    save_experiment_summary,
    save_training_history,
)
from src.models import build_multiclass_hybrid_model
from src.training import TrainingConfig, fit_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENCODED_TO_ORIGINAL_LABEL = {
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


def class_distribution(
    targets: np.ndarray,
) -> dict[str, int]:
    """Calcula el número de muestras de cada calidad."""
    values, counts = np.unique(
        targets,
        return_counts=True,
    )

    return {
        f"quality_{ENCODED_TO_ORIGINAL_LABEL[int(value)]}": int(count)
        for value, count in zip(values, counts)
    }


def parse_arguments() -> Namespace:
    """Procesa los argumentos recibidos desde la terminal."""
    parser = ArgumentParser(
        description=(
            "Train and evaluate the hybrid quantum-classical model "
            "on Wine Quality."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="wine_quality_hybrid",
        help="Base name used for the generated artifacts.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed used for model initialization, batch shuffling "
            "and training."
        ),
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used exclusively for the dataset partitions.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training mini-batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="Number of epochs between progress messages.",
    )

    return parser.parse_args()


def main() -> None:
    """Ejecuta el experimento multiclase completo."""
    arguments = parse_arguments()

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "wine_quality.csv"
    )

    experiment_name = (
        f"{arguments.run_name}"
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

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("WINE QUALITY HYBRID EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Dataset split seed: {arguments.split_seed}")
    print(f"Model and training seed: {arguments.seed}")

    total_start_time = perf_counter()

    print("\nLoading and preprocessing dataset...")

    features, targets = load_wine_quality(
        dataset_path,
        selected_classes=(5, 6, 7),
    )

    prepared_dataset = preprocess_dataset(
        features,
        targets,
        number_of_features=4,
        random_state=arguments.split_seed,
    )

    partition_sizes = {
        "training": int(
            len(prepared_dataset.y_train)
        ),
        "validation": int(
            len(prepared_dataset.y_validation)
        ),
        "test": int(
            len(prepared_dataset.y_test)
        ),
    }

    partition_distributions = {
        "training": class_distribution(
            prepared_dataset.y_train
        ),
        "validation": class_distribution(
            prepared_dataset.y_validation
        ),
        "test": class_distribution(
            prepared_dataset.y_test
        ),
    }

    print(
        "Selected features:",
        prepared_dataset.selected_feature_names,
    )

    print(
        "Partition sizes:",
        partition_sizes,
    )

    print(
        "Class distributions:",
        partition_distributions,
    )

    print("\nBuilding hybrid model...")

    model = build_multiclass_hybrid_model(
        seed=arguments.seed
    )

    print(
        "Trainable parameters:",
        {
            "quantum": model.number_of_quantum_parameters,
            "classical": model.number_of_classical_parameters,
            "total": model.number_of_trainable_parameters,
        },
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

    print("\nStarting training...")

    training_start_time = perf_counter()

    training_result, data_loaders = fit_classifier(
        model,
        prepared_dataset,
        config=training_config,
    )

    training_time_seconds = (
        perf_counter() - training_start_time
    )

    print("\nEvaluating restored best model...")

    evaluation_start_time = perf_counter()

    evaluation_result = evaluate_multiclass_classifier(
        model,
        data_loaders.test,
        class_labels=(0, 1, 2),
        class_names=(
            "quality_5",
            "quality_6",
            "quality_7",
        ),
    )

    evaluation_time_seconds = (
        perf_counter() - evaluation_start_time
    )

    total_time_seconds = (
        perf_counter() - total_start_time
    )

    summary_path = (
        metrics_directory
        / f"{experiment_name}.json"
    )

    history_path = (
        metrics_directory
        / f"{experiment_name}_history.csv"
    )

    predictions_path = (
        metrics_directory
        / f"{experiment_name}_predictions.csv"
    )

    loss_figure_path = (
        figures_directory
        / f"{experiment_name}_loss.png"
    )

    confusion_figure_path = (
        figures_directory
        / f"{experiment_name}_confusion.png"
    )

    checkpoint_path = (
        models_directory
        / f"{experiment_name}.pt"
    )

    true_quality = [
        ENCODED_TO_ORIGINAL_LABEL[int(label)]
        for label in evaluation_result.true_labels
    ]

    predicted_quality = [
        ENCODED_TO_ORIGINAL_LABEL[int(label)]
        for label in evaluation_result.predicted_labels
    ]

    predictions_table = pd.DataFrame(
        {
            "true_encoded_label": (
                evaluation_result.true_labels
            ),
            "predicted_encoded_label": (
                evaluation_result.predicted_labels
            ),
            "true_quality": true_quality,
            "predicted_quality": predicted_quality,
            "probability_quality_5": (
                evaluation_result.probabilities[:, 0]
            ),
            "probability_quality_6": (
                evaluation_result.probabilities[:, 1]
            ),
            "probability_quality_7": (
                evaluation_result.probabilities[:, 2]
            ),
        }
    )

    predictions_table.to_csv(
        predictions_path,
        index=False,
    )

    summary = {
        "experiment": {
            "name": experiment_name,
            "dataset": "Wine Quality",
            "task": "multiclass classification",
            "classes": {
                "0": 5,
                "1": 6,
                "2": 7,
            },
        },
        "random_seeds": {
            "dataset_split": arguments.split_seed,
            "model_and_training": arguments.seed,
        },
        "dataset": {
            "filtered_samples": int(len(targets)),
            "selected_features": (
                prepared_dataset.selected_feature_names
            ),
            "partitions": partition_sizes,
            "class_distributions": (
                partition_distributions
            ),
        },
        "model": {
            "number_of_qubits": 4,
            "number_of_variational_layers": 2,
            "quantum_parameters": (
                model.number_of_quantum_parameters
            ),
            "classical_parameters": (
                model.number_of_classical_parameters
            ),
            "total_trainable_parameters": (
                model.number_of_trainable_parameters
            ),
        },
        "training_configuration": asdict(
            training_config
        ),
        "training_result": {
            "best_epoch": (
                training_result.best_epoch
            ),
            "best_validation_loss": (
                training_result.best_validation_loss
            ),
            "epochs_completed": (
                training_result.epochs_completed
            ),
            "stopped_early": (
                training_result.stopped_early
            ),
        },
        "test_results": {
            "metrics": evaluation_result.metrics,
            "per_class_metrics": (
                evaluation_result.per_class_metrics
            ),
            "confusion_matrix": (
                evaluation_result
                .confusion_matrix
                .tolist()
            ),
        },
        "execution_times_seconds": {
            "training": training_time_seconds,
            "evaluation": evaluation_time_seconds,
            "total": total_time_seconds,
        },
        "software_environment": {
            "python": python_version(),
            "qiskit": get_package_version(
                "qiskit"
            ),
            "qiskit_machine_learning": (
                get_package_version(
                    "qiskit-machine-learning"
                )
            ),
            "torch": get_package_version(
                "torch"
            ),
            "numpy": get_package_version(
                "numpy"
            ),
            "pandas": get_package_version(
                "pandas"
            ),
            "scikit_learn": get_package_version(
                "scikit-learn"
            ),
        },
    }

    save_experiment_summary(
        summary,
        summary_path,
    )

    save_training_history(
        training_result.history,
        history_path,
    )

    plot_training_history(
        training_result.history,
        loss_figure_path,
    )

    plot_confusion_matrix(
        evaluation_result.confusion_matrix,
        confusion_figure_path,
        class_names=(
            "Quality 5",
            "Quality 6",
            "Quality 7",
        ),
        title=(
            "Wine Quality test confusion matrix"
        ),
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "selected_features": (
                prepared_dataset.selected_feature_names
            ),
            "split_seed": arguments.split_seed,
            "training_seed": arguments.seed,
            "training_configuration": asdict(
                training_config
            ),
            "test_metrics": (
                evaluation_result.metrics
            ),
            "per_class_metrics": (
                evaluation_result.per_class_metrics
            ),
        },
        checkpoint_path,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"Epochs completed: "
        f"{training_result.epochs_completed}"
    )

    print(
        f"Best epoch: "
        f"{training_result.best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{training_result.best_validation_loss:.6f}"
    )

    print(
        f"Training time: "
        f"{training_time_seconds:.2f} seconds"
    )

    print("\nTest metrics:")

    for metric_name, metric_value in (
        evaluation_result.metrics.items()
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

    print("\nPer-class metrics:")

    for class_name, class_metrics in (
        evaluation_result.per_class_metrics.items()
    ):
        print(f"  {class_name}:")

        for metric_name, metric_value in (
            class_metrics.items()
        ):
            if isinstance(metric_value, float):
                print(
                    f"    {metric_name}: "
                    f"{metric_value:.6f}"
                )
            else:
                print(
                    f"    {metric_name}: "
                    f"{metric_value}"
                )

    print("\nConfusion matrix:")
    print(evaluation_result.confusion_matrix)

    print("\nGenerated files:")
    print(f"  Summary: {summary_path}")
    print(f"  History: {history_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Loss figure: {loss_figure_path}")
    print(
        f"  Confusion figure: "
        f"{confusion_figure_path}"
    )
    print(f"  Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()