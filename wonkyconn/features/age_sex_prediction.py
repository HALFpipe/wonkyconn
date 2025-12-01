from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import parallel_backend
from nilearn.connectome import sym_matrix_to_vec
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------
# Global caps
# ---------------------------------------------------------------------
FIXED_N_SPLITS = 100
MAX_PCA = 100


# ---------------------------------------------------------------------
# RunReport for concise summary
# ---------------------------------------------------------------------
class RunReport:
    def __init__(self):
        self.sex_error: Optional[str] = None
        self.age_error: Optional[str] = None
        self.notes: Dict[str, str] = {}

    def set_note(self, key: str, msg: str):
        self.notes[key] = msg

    def set_sex_error(self, msg: str):
        self.sex_error = msg

    def set_age_error(self, msg: str):
        self.age_error = msg

    def short_summary(self) -> str:
        parts = []
        if self.sex_error:
            parts.append(f"sex prediction failed: {self.sex_error}")
        if self.age_error:
            parts.append(f"age prediction failed: {self.age_error}")
        if not parts:
            parts.append("both predictions succeeded")
        return " | ".join(parts)


# ---------------------------------------------------------------------
# Helper: safe PCA dimension
# ---------------------------------------------------------------------
def _pca_dim(n_samples: int, n_features: int, requested: int = MAX_PCA) -> int:
    # FIX: Anticipate that Cross-Validation (test_size=0.2) will reduce
    # the number of training samples to roughly 80% of the total.
    n_train_samples = int(n_samples * 0.8)

    # Ensure we stay below the training sample limit (-1 for safety)
    safe_limit = max(2, n_train_samples - 1)

    dim = min(MAX_PCA, requested, n_features, safe_limit)
    return max(2, dim)


# ---------------------------------------------------------------------
# Core training / CV
# ---------------------------------------------------------------------
def training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,  # "classification" or "regression" (required)
    model_type: str = "default",  # Kept for metadata compatibility
    n_splits: int = FIXED_N_SPLITS,
    random_state: int = 1,
    n_pca: int = MAX_PCA,
    n_jobs: int = 4,
    report: Optional[RunReport] = None,
):
    """
    Run CV with explicit task_type.
    Models are hardcoded to LogisticRegression (classification) and Ridge (regression).

    Returns
    -------
    df_scores, summary, meta
    """

    start = time.time()
    X = np.asarray(X, dtype=np.float32, order="C")
    y = np.asarray(y)

    # If classification, label-encode string labels
    if task_type == "classification" and y.dtype.kind in {"U", "S", "O"}:
        y = LabelEncoder().fit_transform(y)

    # PCA bound
    n_components = _pca_dim(
        n_samples=len(y),
        n_features=(X.shape[1] if X.size else 0),
        requested=n_pca,
    )

    # -------------------------------------------------
    # CLASSIFICATION BRANCH (sex) -> LogisticRegression
    # -------------------------------------------------
    if task_type == "classification":
        # Always use LogisticRegression
        estimator = LogisticRegression(
            max_iter=5_000,
            solver="saga",
            penalty="l2",
            C=1.0,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        # ----- CV split logic (StratifiedShuffleSplit)
        class_counts = np.bincount(y.astype(int)) if y.size else np.array([0])
        min_class = int(class_counts.min()) if class_counts.size > 0 else 0
        k = n_splits

        # si une classe n'a qu'un seul sujet → impossible de stratifier correctement
        if min_class < 2:
            k = 2
            if report is not None:
                report.set_note(
                    "sex_cv",
                    f"extremely small minority class (min_class={min_class}); using n_splits={k}",
                )
        else:
            if k > min_class:
                if report is not None:
                    report.set_note(
                        "sex_cv",
                        f"requested {k} splits > smallest class {min_class}; using {min_class}",
                    )
                k = max(2, min_class)

        cv = StratifiedShuffleSplit(
            n_splits=k,
            test_size=0.2,
            random_state=random_state,
        )

        multiclass = np.unique(y).size > 2
        scoring = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc_ovr" if multiclass else "roc_auc",
            "f1": "f1_weighted" if multiclass else "f1",
        }

    # -------------------------------------------------
    # REGRESSION BRANCH (age) -> Ridge
    # -------------------------------------------------
    elif task_type == "regression":
        # Always use Ridge
        estimator = Ridge(alpha=1.0)

        # ----- CV split logic (ShuffleSplit)
        if len(y) <= 2:
            k = 2
            if report is not None:
                report.set_note(
                    "age_cv",
                    f"very small sample size (n={len(y)}); using n_splits=2",
                )
        else:
            k = n_splits

        cv = ShuffleSplit(
            n_splits=k,
            test_size=0.2,
            random_state=random_state,
        )

        scoring = {
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "r2": "r2",
        }

    else:
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")

    # ----- full sklearn pipeline
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    iterated_power=3,
                    random_state=random_state,
                ),
            ),
            ("estimator", estimator),
        ]
    )

    # Run CV. We keep error_score="raise" and let caller catch.
    with parallel_backend("threading", n_jobs=n_jobs):
        out = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
            pre_dispatch="2*n_jobs",
            error_score="raise",
        )

    df_scores = pd.DataFrame({key.replace("test_", ""): v for key, v in out.items() if key.startswith("test_")})
    summary = df_scores.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    meta = {
        "task_type": task_type,
        "splits_used": k,
        "splits_requested": n_splits,
        "pca_components": n_components,
        "model_type": str(estimator),  # More accurate than passed string
        "runtime_s": time.time() - start,
    }

    return df_scores, summary, meta


# ---------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------
def age_sex_scores(
    connectivity_matrices: List["ConnectivityMatrix"],
    ages: np.ndarray,
    genders: np.ndarray,
    *,
    n_splits: int = FIXED_N_SPLITS,
    random_state: int = 42,
    n_pca: int = MAX_PCA,
    n_jobs: int = 4,
    clf_model: str = "logreg",  # Only used for API compatibility, logic is hardcoded
    reg_model: str = "ridge",  # Only used for API compatibility, logic is hardcoded
) -> Dict[str, float]:
    """
    Compute sex (classification) and age (regression) metrics.
    Models: LogisticRegression (sex) and Ridge (age).
    """

    report = RunReport()

    # Build X
    if not connectivity_matrices:
        X = np.empty((0, 0), dtype=np.float32)
    else:
        # IMPORTANT : convertir la liste en ndarray AVANT d'appeler sym_matrix_to_vec
        mats = np.asarray([cm.load() for cm in connectivity_matrices], dtype=np.float32)
        X = sym_matrix_to_vec(
            mats,
            discard_diagonal=True,
        ).astype(np.float32)

    ages = np.asarray(ages).astype(float, copy=False)
    genders = np.asarray(genders)

    # Bound PCA from current data
    n_components = _pca_dim(
        n_samples=len(genders),
        n_features=(X.shape[1] if X.size else 0),
        requested=n_pca,
    )

    # add std in the output dict
    out: Dict[str, float] = dict(
        sex_auc=np.nan,
        sex_auc_std=np.nan,
        sex_accuracy=np.nan,
        age_mae=np.nan,
        age_mae_std=np.nan,
        age_r2=np.nan,
    )

    # ---- SEX (classification -> LogisticRegression)
    try:
        _, sum_sex, meta_sex = training_pipeline(
            X,
            genders,
            task_type="classification",
            model_type=clf_model,
            n_splits=n_splits,
            random_state=random_state,
            n_pca=n_components,
            n_jobs=n_jobs,
            report=report,
        )

        if "roc_auc" in sum_sex.index:
            out["sex_auc"] = float(sum_sex.loc["roc_auc", "mean"])
            out["sex_auc_std"] = float(sum_sex.loc["roc_auc", "std"])

        if "accuracy" in sum_sex.index:
            out["sex_accuracy"] = float(sum_sex.loc["accuracy", "mean"])

        report.set_note(
            "sex_meta",
            (
                f"sex OK | splits={meta_sex['splits_used']}/{meta_sex['splits_requested']} "
                f"| PCA={meta_sex['pca_components']} | model={meta_sex['model_type']} "
                f"| t={meta_sex['runtime_s']:.2f}s"
            ),
        )
    except Exception as exc:
        report.set_sex_error(str(exc))

    # ---- AGE (regression -> Ridge)
    try:
        _, sum_age, meta_age = training_pipeline(
            X,
            ages,
            task_type="regression",
            model_type=reg_model,
            n_splits=n_splits,
            random_state=random_state,
            n_pca=n_components,
            n_jobs=n_jobs,
            report=report,
        )

        mae_key = "neg_mean_absolute_error"
        if mae_key in sum_age.index:
            # mean : on remet en positif
            out["age_mae"] = float(-sum_age.loc[mae_key, "mean"])
            # std : l'écart-type reste positif
            out["age_mae_std"] = float(sum_age.loc[mae_key, "std"])

        if "r2" in sum_age.index:
            out["age_r2"] = float(sum_age.loc["r2", "mean"])

        report.set_note(
            "age_meta",
            (
                f"age OK | splits={meta_age['splits_used']}/{meta_age['splits_requested']} "
                f"| PCA={meta_age['pca_components']} | model={meta_age['model_type']} "
                f"| t={meta_age['runtime_s']:.2f}s"
            ),
        )
    except Exception as exc:
        report.set_age_error(str(exc))

    # ---- one final line, no spam
    print("[age_sex_prediction] " + report.short_summary())

    return out
