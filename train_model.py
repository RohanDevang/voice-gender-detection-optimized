"""
Training script for voice gender classification.

This script loads MFCC features from audio files, performs a train/test
split, optimises an SVM classifier using cross‑validation and evaluates
the resulting model on hold‑out data.  Additional metrics beyond
accuracy are computed to provide deeper insight into performance on
balanced or imbalanced datasets.

Usage
-----

Run the script from the command line specifying the dataset directory.

.. code-block:: bash

    python train_model.py path/to/dataset --num_mfcc 20 --model_out svm_model.pkl

The trained model will be saved to the given output path.

"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import load_dataset


def train_model(
    dataset_path: str,
    num_mfcc: int = 13,
    test_size: float = 0.2,
    random_state: int = 42,
    cv: int = 5,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Train an SVM classifier on MFCC features and evaluate it.

    Parameters
    ----------
    dataset_path : str
        Root directory of the dataset with subfolders per class.
    num_mfcc : int, optional
        Number of MFCC coefficients to compute per audio sample.
    test_size : float, optional
        Fraction of the dataset to reserve for the test split.
    random_state : int, optional
        Seed for random number generators to ensure reproducibility.
    cv : int, optional
        Number of cross‑validation folds for hyper‑parameter tuning.

    Returns
    -------
    model : Pipeline
        The best performing classifier pipeline found via cross‑validation.
    metrics : dict
        Mapping of metric names to scores on the test set.
    X_test : np.ndarray
        Feature vectors for the test set.
    y_test : np.ndarray
        Ground truth labels for the test set.
    y_pred : np.ndarray
        Predicted labels for the test set.
    """
    # Extract features and labels.
    X, y = load_dataset(dataset_path, num_mfcc=num_mfcc)
    # Perform a stratified train/test split to maintain class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # Define a pipeline consisting of scaling and an SVM classifier.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ])
    # Hyper‑parameter grid for cross‑validation.
    param_grid = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
    }
    # GridSearchCV performs exhaustive search over the parameter grid.
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    # Use the best estimator found.
    best_model: Pipeline = grid_search.best_estimator_
    # Make predictions on the test set.
    y_pred = best_model.predict(X_test)
    # Compute evaluation metrics.
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return best_model, metrics, X_test, y_test, y_pred


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, labels: Optional[Tuple[str, str]] = ("Male", "Female")) -> None:
    """Plot the confusion matrix for predictions vs ground truth.

    Parameters
    ----------
    y_test : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : tuple of str, optional
        Display labels for the confusion matrix axes.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def save_model(model: Pipeline, path: str) -> None:
    """Persist a trained model to disk using joblib.

    Parameters
    ----------
    model : Pipeline
        Trained scikit‑learn pipeline to persist.
    path : str
        File path where the model should be saved.
    """
    joblib.dump(model, path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a voice gender classifier and evaluate it.")
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset root directory containing class subfolders.",
    )
    parser.add_argument(
        "--num_mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients to compute per audio sample (default: 13).",
    )
    parser.add_argument(
        "--model_out",
        default="svm_model.pkl",
        help="Filename for saving the trained model (default: svm_model.pkl).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    model, metrics, X_test, y_test, y_pred = train_model(
        args.dataset_path,
        num_mfcc=args.num_mfcc,
    )
    print(f"Best model hyper‑parameters: {model.get_params()}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    # Plot and display confusion matrix for the test set.
    plot_confusion_matrix(y_test, y_pred)
    # Save the trained model.
    save_model(model, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
