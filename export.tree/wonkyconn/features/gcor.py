"""GCOR feature implementation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy import typing as npt

from ..base import ConnectivityMatrix


# seann: AFNI's `gcor2` computes GCOR as ||(1/M) Σ u_i||^2 for unit-variance time series,
# which equals the average of the pairwise dot products u_i · u_j forming R.
def compute_gcor(matrix: npt.NDArray[np.float64]) -> float:
    """Compute the GCOR value from a correlation matrix."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix must be a square 2D array to compute GCOR.")
    # seann: relmat already stores R, so averaging equals ||(1/M) Σ u_i||^2 from AFNI.
    value = float(np.nanmean(matrix, dtype=np.float64))
    return value


def calculate_gcor(
    connectivity_matrices: Iterable[ConnectivityMatrix],
) -> float:
    """Aggregate GCOR values for a collection of connectivity matrices."""
    # seann: load ROI correlation matrix and compute GCOR per subject
    data = np.fromiter(
        (
            compute_gcor(np.asarray(connectivity_matrix.load(), dtype=np.float64))
            for connectivity_matrix in connectivity_matrices
        ),
        dtype=np.float64,
    )
    return np.nanmean(data, dtype=np.float64).item()
