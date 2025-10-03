from __future__ import (
    annotations,
)  # seann: added future import for annotations to allow type hints in function signatures

from itertools import chain
from typing import Iterable

import numpy as np
import pandas as pd
from numpy import typing as npt
from patsy.highlevel import dmatrix
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from ..base import ConnectivityMatrix
from ..correlation import correlation_p_value, partial_correlation


def calculate_qcfc(
    data_frame: pd.DataFrame,
    connectivity_matrices: Iterable[ConnectivityMatrix],
    metric_key: str = "MeanFramewiseDisplacement",
) -> pd.DataFrame:
    """
    metric calculation: quality control / functional connectivity

    For each edge, we then computed the correlation between the weight of
    that edge and the mean relative RMS motion.
    QC-FC relationships were calculated as partial correlations that
    accounted for participant age and sex

    Parameters:
        data_frame (pd.DataFrame): The data frame containing the covariates "age" and "gender".
                                   It needs to have one row for each connectivity matrix.
        connectivity_matrices (Iterable[ConnectivityMatrix]): The connectivity matrices to calculate QCFC for.
        metric_key (str, optional): The key of the metric to use for QCFC calculation. Defaults to "MeanFramewiseDisplacement".

    Returns:
        pd.DataFrame: The QCFC values between connectivity matrices and the metric.

    """
    metrics = np.asarray(
        [connectivity_matrix.metadata.get(metric_key, np.nan) for connectivity_matrix in connectivity_matrices]
    )
    if np.isnan(metrics).all():
        raise ValueError(f"None of the connectivity matrices have a metric with key '{metric_key}'")
    covariates = np.asarray(dmatrix("age + gender", data_frame))

    connectivity_arrays = [
        connectivity_matrix.load()
        for connectivity_matrix in tqdm(
            connectivity_matrices,
            desc="Loading connectivity matrices",
            leave=False,
        )
    ]

    # Ensure that all arrays are square and have the same shape
    (n,) = set(chain.from_iterable(a.shape for a in connectivity_arrays))

    # Extract the lower triangles
    i, j = np.tril_indices(n, k=-1)
    connectivity_array = np.concatenate(
        [a[i, j, np.newaxis] for a in connectivity_arrays],
        axis=1,
    )

    correlation, count = partial_correlation(connectivity_array, metrics, covariates)
    p_value = correlation_p_value(correlation, count)

    qcfc = pd.DataFrame(dict(i=i, j=j, correlation=correlation, p_value=p_value))
    qcfc = qcfc.set_index(["i", "j"])

    return qcfc


# seann: added type for series
def calculate_median_absolute(x: "pd.Series[float]") -> float:
    """Calculate Absolute median value"""
    return x.abs().median()


# seann: added type for series
def significant_level(x: "pd.Series[float]", alpha: float = 0.05, correction: str | None = None) -> npt.NDArray[np.bool_]:
    """
    Apply FDR correction to a pandas.Series p-value object.

    Parameters
    ----------

    x : pandas.Series
        Uncorrected p-values.

    alpha : float
        Alpha threshold.

    method : None or str
        Default as None for no multiple comparison
        Multiple comparison methods.
        See statsmodels.stats.multitest.multipletests

    Returns
    -------
    ndarray, boolean
        Mask for data passing multiple comparison test.
    """
    if isinstance(correction, str):
        res, _, _, _ = multipletests(x, alpha=alpha, method=correction)
    else:
        res = x < 0.05
    return res


def calculate_qcfc_percentage(qcfc: pd.DataFrame) -> float:
    """
    Calculate the percentage of significant QC-FC relationships.

    Parameters
    ----------
    qcfc : pd.DataFrame
        The QC-FC values between connectivity matrices and the metric.

    Returns
    -------
    float
        The percentage of significant QC-FC relationships.
    """
    # seann: cast mean p_value to float for type consistency
    return 100 * float(significant_level(qcfc.p_value).mean())
