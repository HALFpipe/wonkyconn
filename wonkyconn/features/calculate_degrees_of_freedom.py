"""Calculate degree of freedom"""

import numpy as np
from ..base import ConnectivityMatrix


def calculate_degrees_of_freedom_loss(
    connectivity_matrices: list[ConnectivityMatrix],
) -> float:
    """
    Calculate the percent of degrees of freedom lost during denoising.

    Parameters:
    - bids_file (BIDSFile): The BIDS file for which to calculate the degrees of freedom.

    Returns:
    - float: The percentage of degrees of freedom lost.

    """

    values: list[float] = []

    for connectivity_matrix in connectivity_matrices:
        metadata = connectivity_matrix.metadata

        total = 0

        confound_regressors = metadata.get("ConfoundRegressors", list())
        total += len(confound_regressors)

        total += metadata.get("NumberOfVolumesDiscardedByMotionScrubbing", 0)
        total += metadata.get("NumberOfVolumesDiscardedByNonsteadyStatesDetector", 0)

        # TODO Support ICA-AROMA

        values.append(total / connectivity_matrix.load().shape[0])

    return float(np.mean(values))
