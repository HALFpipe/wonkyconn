"""Calculate degree of freedom"""

from functools import partial
from typing import NamedTuple, Sequence

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
    """
    Calculate the percent of degrees of freedom lost during denoising.

    Parameters:
    - bids_file (BIDSFile): The BIDS file for which to calculate the degrees of freedom.

    Returns:
    - float: The percentage of degrees of freedom lost.

    """
    # seann: ensure count is a list of integers instead of a numpy array
    count: list[int] = [
        connectivity_matrix.metadata["NumberOfVolumes"]
        for connectivity_matrix in connectivity_matrices
    ]

    calculate = partial(_calculate_for_key, connectivity_matrices, count)
    return DegreesOfFreedomLossResult(
        confound_regression_percentage=calculate({"ConfoundRegressors"}),
        motion_scrubbing_percentage=calculate(
            {"NumberOfVolumesDiscardedByMotionScrubbing"}
        ),
        nonsteady_states_detector_percentage=calculate(
            {"NumberOfVolumesDiscardedByNonsteadyStatesDetector", "DummyScans"}
        ),
    )


def _get_values(
    connectivity_matrix: ConnectivityMatrix, keys: set[str]
) -> float | Sequence[str] | None:
    metadata = connectivity_matrix.metadata
    values: set[float | tuple[str, ...]] = set()
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (int, float)):
            values.add(float(value))
        elif isinstance(value, Sequence):
            values.add(tuple(value))
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"Multiple values found for keys {keys} in metadata: {values}")
    return next(iter(values))


# seann: ensure function accepts sequence of integers
def _calculate_for_key(
    connectivity_matrices: list[ConnectivityMatrix],
    count: Sequence[int],
    keys: set[str],
) -> float:
    values: Sequence[float | Sequence[str] | None] = [
        _get_values(connectivity_matrix, keys)
        for connectivity_matrix in connectivity_matrices
    ]

    if all(value is None for value in values):
        return np.nan

    proportions: list[float] = []
    for value, c in zip(values, count, strict=True):
        if isinstance(value, float):
            proportions.append(value / c)
        elif isinstance(value, Sequence):
            proportions.append(len(value) / c)
        else:
            raise ValueError(f"Unexpected value for `{keys}`: {value}")

    percentages = pd.Series(proportions) * 100
    return percentages.mean()
