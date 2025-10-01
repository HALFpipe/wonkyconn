"""GCOR feature implementation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy import typing as npt

from .base import MeanAndSEMResult
from ..base import ConnectivityMatrix

# seann: AFNI's `gcor2` computes GCOR as ||(1/M) Σ u_i||^2 for unit-variance time series,
# which equals the average of the pairwise dot products u_i · u_j forming R.

def _validate_square_matrix(matrix: npt.NDArray[np.float64]) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Connectivity matrix must be a square 2D array to compute GCOR."
        )


def compute_gcor(matrix: npt.NDArray[np.float64]) -> float:
    """Compute the GCOR value from a correlation matrix."""
    _validate_square_matrix(matrix)
    # seann: relmat already stores R, so averaging equals ||(1/M) Σ u_i||^2 from AFNI.
    value = float(np.nanmean(matrix, dtype=np.float64))
    return value


def calculate_gcor(
    connectivity_matrices: Iterable[ConnectivityMatrix],
) -> MeanAndSEMResult:
    """Aggregate GCOR values for a collection of connectivity matrices."""
    gcor_values: list[float] = []

    for connectivity_matrix in connectivity_matrices:
        # seann: load ROI correlation matrix and compute GCOR per subject
        matrix = np.asarray(connectivity_matrix.load(), dtype=np.float64)
        gcor_values.append(compute_gcor(matrix))

    if not gcor_values:
        return MeanAndSEMResult.empty()

    data = np.asarray(gcor_values, dtype=np.float64)
    mean_value = float(np.nanmean(data, dtype=np.float64))

    valid_mask = np.isfinite(data)
    valid_count = int(valid_mask.sum())
    if valid_count < 2:
        sem_value = np.nan
    else:
        # seann: SEM requires unbiased std (ddof=1) over finite entries
        sem_value = float(
            np.nanstd(data, ddof=1, dtype=np.float64) / np.sqrt(valid_count)
        )

    return MeanAndSEMResult(mean=mean_value, sem=sem_value)
