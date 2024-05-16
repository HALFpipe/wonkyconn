from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import scipy
from numpy import typing as npt
from patsy.highlevel import dmatrix
from tqdm.auto import tqdm

from ..base import ConnectivityMatrix

from statsmodels.stats.multitest import multipletests


class PartialCorrelationResult(NamedTuple):
    correlation: float
    p_value: float


def partial_correlation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64] | None = None,
) -> PartialCorrelationResult:
    """A minimal implementation of partial correlation.

    Parameters
    ----------
    x, y : np.ndarray
        Variable of interest.

    cov : None, np.ndarray
        Variable to be removed from variable of interest.
        If None, do a normal pearson's correlation.

    Returns
    -------
    dict
        Correlation and p-value.
    """
    if cov is not None:
        beta_cov_x = np.linalg.lstsq(cov, x)[0]
        beta_cov_y = np.linalg.lstsq(cov, y)[0]
        resid_x = x - cov.dot(beta_cov_x)
        resid_y = y - cov.dot(beta_cov_y)
        r, p_value = scipy.stats.pearsonr(resid_x, resid_y)
    else:
        r, p_value = scipy.stats.pearsonr(x, y)
    return PartialCorrelationResult(r, p_value)


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
        data_frame (pd.DataFrame): The data frame containing the covariates "age" and "gender". It needs to have one row for each connectivity matrix.
        connectivity_matrices (Iterable[ConnectivityMatrix]): The connectivity matrices to calculate QCFC for.
        metric_key (str, optional): The key of the metric to use for QCFC calculation. Defaults to "MeanFramewiseDisplacement".

    Returns:
        pd.DataFrame: The QCFC values between connectivity matrices and the metric.

    """
    metrics = pd.Series(
        [
            connectivity_matrix.metadata.get(metric_key, np.nan)
            for connectivity_matrix in connectivity_matrices
        ]
    )
    covariates = dmatrix("age + gender", data_frame)

    connectivity_array = np.concatenate(
        [
            connectivity_matrix.load()[:, :, np.newaxis]
            for connectivity_matrix in tqdm(
                connectivity_matrices, desc="Loading connectivity matrices", leave=False
            )
        ],
        axis=2,
    )
    n, _, _ = connectivity_array.shape

    indices = list(zip(*np.tril_indices(n, k=-1), strict=True))

    records = list()
    for i, j in tqdm(indices, desc="Calculating QC-FC", leave=False):
        record = partial_correlation(
            connectivity_array[i, j], metrics, covariates
        )._asdict()
        record["i"] = i
        record["j"] = j
        records.append(record)

    qcfc = pd.DataFrame.from_records(records, index=["i", "j"])
    return qcfc


def calculate_median_absolute(x: pd.Series) -> float:
    """Calculate Absolute median value"""
    return x.abs().median()


def significant_level(
    x: pd.Series, alpha: float = 0.05, correction: str | None = None
) -> npt.NDArray[np.bool_]:
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
        Mutiple comparison methods.
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
    return 100 * significant_level(qcfc.p_value).mean()
