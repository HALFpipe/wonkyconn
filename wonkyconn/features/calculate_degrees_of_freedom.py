"""Calculate degree of freedom"""

from functools import partial
from typing import NamedTuple, Sequence

import numpy as np
from numpy import typing as npt
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

    count: npt.NDArray[np.int64] = np.asarray(
        [
            connectivity_matrix.load().shape[0]
            for connectivity_matrix in connectivity_matrices
        ]
    )

    calculate = partial(_calculate_for_key, connectivity_matrices, count)
    return DegreesOfFreedomLossResult(
        confound_regression_percentage=calculate("ConfoundRegressors"),
        motion_scrubbing_percentage=calculate(
            "NumberOfVolumesDiscardedByMotionScrubbing"
        ),
        nonsteady_states_detector_percentage=calculate(
            "NumberOfVolumesDiscardedByNonsteadyStatesDetector"
        ),
    )


def _calculate_for_key(
    connectivity_matrices: list[ConnectivityMatrix], count: list[int], key: str
) -> float:
    values: Sequence[int | list[str] | None] = [
        connectivity_matrix.metadata.get(key, None)
        for connectivity_matrix in connectivity_matrices
    ]

    if all(value is None for value in values):
        return np.nan

    proportions: list[float] = []
    if key.startswith("NumberOf"):
        for value, c in zip(values, count, strict=True):
            if isinstance(value, int):
                proportions.append(value / c)
            else:
                raise ValueError(f"Unexpected value for `{key}`: {value}")

    else:
        for value, c in zip(values, count, strict=True):
            if isinstance(value, list):
                proportions.append(len(value) / c)
            else:
                raise ValueError(f"Unexpected value for `{key}`: {value}")
    percentages = pd.Series(proportions) * 100
    return percentages.mean()
