from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd
from joblib import parallel_backend  # type: ignore[import-not-found]
from nilearn.connectome import sym_matrix_to_vec  # type: ignore[import-not-found]
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore[import-not-found]
from sklearn.impute import SimpleImputer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-not-found]
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_validate  # type: ignore[import-not-found]
from sklearn.pipeline import Pipeline  # type: ignore[import-not-found]
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from ..base import ConnectivityMatrix


def training_pipeline(
    connectivity_data: NDArray[np.float32],
    target_labels: NDArray[np.float64] | NDArray[np.str_],
    task_type: str,
    n_splits: int,
    n_pca: int,
    n_jobs: int = 4,
    random_state: int = 1,
) -> pd.DataFrame:
    """Runs a cross-validation pipeline for age or sex prediction.

    Args:
        connectivity_data: Vectorized connectivity matrix.
        target_labels: Target vector (ages or genders).
        task_type (str): Type of task ('classification' or 'regression').
        n_splits (int): Number of repetitions for cross-validation.
        n_pca (int): Number of principal components to extract.
        n_jobs (int): Number of cores for parallel calculation.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Statistics (mean, std) of the scores obtained.
    """
    connectivity_data = np.asarray(connectivity_data, dtype=np.float32, order="C")

    if task_type == "classification":
        y_train = LabelEncoder().fit_transform(target_labels)
        estimator = LogisticRegression(max_iter=5000, solver="saga", penalty="l2", n_jobs=n_jobs, random_state=random_state)
        cv_strategy = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
        scoring_metrics = {"accuracy": "accuracy", "roc_auc": "roc_auc"}
    else:
        y_train = np.asarray(target_labels)
        estimator = Ridge(alpha=1.0)
        cv_strategy = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
        scoring_metrics = {"mae": "neg_mean_absolute_error", "r2": "r2"}

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, svd_solver="randomized", random_state=random_state)),
            ("estimator", estimator),
        ]
    )

    with parallel_backend("threading", n_jobs=n_jobs):
        cv_results = cross_validate(
            pipe,
            connectivity_data,
            y_train,
            cv=cv_strategy,
            scoring=scoring_metrics,
            n_jobs=n_jobs,
        )

    scores_df = pd.DataFrame({k.replace("test_", ""): v for k, v in cv_results.items() if k.startswith("test_")})
    return scores_df.agg(["mean", "std"]).T


def age_sex_scores(
    connectivity_matrices: List[ConnectivityMatrix],
    ages: NDArray[np.float64],
    genders: NDArray[np.str_],
    n_splits: int,
    n_pca: int,
    n_jobs: int = 4,
    random_state: int = 42,
) -> Dict[str, float]:
    """Computes age and sex prediction scores via connectivity.

    Args:
        connectivity_matrices (List[ConnectivityMatrix]): List of matrix objects.
        ages: Vector of subject ages.
        genders: Vector of subject genders.
        n_splits (int): Number of splits for cross-validation.
        n_pca (int): Number of PCA components.
        n_jobs (int): Number of joblib threads.
        random_state (int): Random seed.

    Returns:
        Dict[str, float]: Dictionary containing AUC, Accuracy, MAE, and R2.
    """
    loaded_mats = np.asarray([cm.load() for cm in connectivity_matrices], dtype=np.float32)
    connectivity_features = sym_matrix_to_vec(loaded_mats, discard_diagonal=True)

    sex_summary = training_pipeline(
        connectivity_features,
        genders,
        task_type="classification",
        n_splits=n_splits,
        n_pca=n_pca,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    age_summary = training_pipeline(
        connectivity_features,
        ages,
        task_type="regression",
        n_splits=n_splits,
        n_pca=n_pca,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    return {
        "sex_auc": float(sex_summary.loc["roc_auc", "mean"]),  # type: ignore[arg-type]
        "sex_auc_std": float(sex_summary.loc["roc_auc", "std"]),  # type: ignore[arg-type]
        "sex_accuracy": float(sex_summary.loc["accuracy", "mean"]),  # type: ignore[arg-type]
        "age_mae": float(-age_summary.loc["mae", "mean"]),  # type: ignore[arg-type, operator]
        "age_mae_std": float(age_summary.loc["mae", "std"]),  # type: ignore[arg-type]
        "age_r2": float(age_summary.loc["r2", "mean"]),  # type: ignore[arg-type]
    }
