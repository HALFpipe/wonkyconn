"""Calculate degree of freedom"""

from functools import partial
from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd

from ..base import ConnectivityMatrix


class DegreesOfFreedomLossResult(NamedTuple):
    confound_regression_percentage: float
    motion_scrubbing_percentage: float
    nonsteady_states_detector_percentage: float


def calculate_degrees_of_freedom_loss(
    connectivity_matrices: list[ConnectivityMatrix],
) -> DegreesOfFreedomLossResult:
    """Calculate the percent of degrees of freedom lost during denoising.

    Args:
        connectivity_matrices (list[ConnectivityMatrix]): Connectivity matrices to evaluate.

    Returns:
        DegreesOfFreedomLossResult: Percentages for confound regression, motion
            scrubbing, and non-steady-state volume removal.
    """
    count: list[int] = [connectivity_matrix.metadata["NumberOfVolumes"] for connectivity_matrix in connectivity_matrices]

    calculate = partial(calculate_for_key, connectivity_matrices, count)
    return DegreesOfFreedomLossResult(
        confound_regression_percentage=calculate(
            keys=["ConfoundRegressors"],
            predicate=lambda confound_regressor: not confound_regressor.startswith("motion_outlier"),
        ),
        motion_scrubbing_percentage=calculate(
            keys=["NumberOfVolumesDiscardedByMotionScrubbing", "ConfoundRegressors"],
            predicate=lambda confound_regressor: confound_regressor.startswith("motion_outlier"),
        ),
        nonsteady_states_detector_percentage=calculate(
            keys=["NumberOfVolumesDiscardedByNonsteadyStatesDetector", "DummyScans"],
        ),
    )


def _get_value(connectivity_matrix: ConnectivityMatrix, keys: list[str]) -> float | Sequence[str] | None:
    metadata = connectivity_matrix.metadata
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, Sequence):
            return tuple(value)
    return None


def calculate_for_key(
    connectivity_matrices: list[ConnectivityMatrix],
    count: Sequence[int],
    keys: list[str],
    predicate: Callable[[str], bool] | None = None,
) -> float:
    """Get the mean percentage for a given metadata key.

    Args:
        connectivity_matrices (list[ConnectivityMatrix]): List of connectivity matrices.
        count (Sequence[int]): The total number of volumes for each connectivity matrix.
        keys (list[str]): Metadata keys in decreasing priority. Values can be numeric or
            a sequence of strings.
        predicate (Callable[[str], bool] | None): Optional filter applied when the
            metadata value is a sequence of strings.

    Returns:
        float: The mean percentage.
    """

    values: Sequence[float | Sequence[str] | None] = [
        _get_value(connectivity_matrix, keys) for connectivity_matrix in connectivity_matrices
    ]

    if all(value is None for value in values):
        return np.nan

    proportions: list[float] = []
    for value, c in zip(values, count, strict=True):
        if isinstance(value, float):
            proportions.append(value / c)
        elif isinstance(value, Sequence):
            if predicate is None:
                k = len(value)
            else:
                k = sum(1 for v in value if predicate(v))
            proportions.append(k / c)
        else:
            raise ValueError(f"Unexpected value for `{keys}`: {value}")

    percentages = pd.Series(proportions) * 100
    return percentages.mean()
