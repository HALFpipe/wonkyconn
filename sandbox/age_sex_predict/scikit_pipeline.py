import os
import time

import numpy as np
import pandas as pd
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

# we are using the raw recording features here. All the outputs from 'extract.py should contain it'
BASELINE_FEAT = "./outputs/brainlm.vitmae_650M.direct_transfer.gigaconnectome"


def get_baseline_data(timeseries_length=140):
    # load data
    features_direct = load_from_disk(BASELINE_FEAT)

    ts_flatten = [
        np.array(example).reshape(3, 424, timeseries_length)[0].T.flatten() for example in features_direct["padded_recording"]
    ]

    correlation_baseline = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    ts = [np.array(example).reshape(3, 424, timeseries_length)[0].T for example in features_direct["padded_recording"]]
    fc = correlation_baseline.fit_transform(ts)
    labels = features_direct["Sex"], (np.array(features_direct["Candidate_Age"]) / 12).tolist()
    return ts_flatten, fc, labels


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from joblib import parallel_backend  # << ajouter
from sklearn.linear_model import LogisticRegression, Ridge


def training_pipeline(X, y, *, n_splits=10, random_state=1, n_pca=100, n_jobs=4, model_type="svm"):
    """
    Pipeline PCA + modèle de classification/régression
    model_type = "svm" (par défaut) | "logreg" (classification) | "ridge" (régression)
    """
    start = time.time()
    X = np.asarray(X, dtype=np.float32, order="C")
    y = np.asarray(y)

    # Classification vs régression
    is_classification = isinstance(y[0], str) or len(np.unique(y)) <= 10

    if isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)

    # Choix du modèle
    if is_classification:
        if model_type == "logreg":
            estimator = LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0, n_jobs=n_jobs, random_state=random_state)
        else:  # "svm"
            estimator = SVC(C=1, class_weight="balanced", kernel="rbf", gamma="scale")
        scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"}
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)
    else:
        if model_type == "ridge":
            estimator = Ridge(alpha=1.0, random_state=random_state)
        else:  # "svm"
            estimator = SVR()
        scoring = {
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "r2": "r2",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state)

    # Pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca, svd_solver="randomized", iterated_power=3, random_state=random_state)),
            ("estimator", estimator),
        ],
        memory=None,
    )

    # Cross-validation
    with parallel_backend("threading", n_jobs=n_jobs):
        out = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=False, pre_dispatch="2*n_jobs"
        )

    df_scores = pd.DataFrame({k.replace("test_", ""): v for k, v in out.items() if k.startswith("test_")})
    summary = df_scores.agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    print(f"Total time = {time.time() - start:.2f} sec | model={model_type}")

    return df_scores, summary


def create_baseline_data():
    ts, fc, (sex, age) = get_baseline_data(timeseries_length=140)
    for bn_feat, feat_name in zip((ts, fc), ("timeseries", "connectivity")):
        for target, target_name in zip((sex, age), ("sex", "age")):
            output_path = f"outputs/x-{feat_name}_y-{target_name}_prediction.tsv"
            scores = svm_pipeline(bn_feat, target)
            pd.DataFrame(scores).to_csv(output_path, sep="\t")
