import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.stats import spearmanr


def calculate_distance_dependence(qcfc: pd.DataFrame, distance_matrix: npt.NDArray[np.float64]) -> float:
    """
    Calculate the Spearman correlation between the distance matrix and the QC-FC correlation values.

    Parameters:
    - qcfc (pd.DataFrame): The qcfc DataFrame containing the correlation values with
      a multi-index of the lower triangular indices
    - distance_matrix (npt.NDArray[np.float64]): The distance matrix of the atlas regions.

    Returns:
    - float: The distance dependence value.

    """
    i, j = map(np.asarray, zip(*qcfc.index, strict=False))
    distance_vector = distance_matrix[i, j]
    r, _ = spearmanr(distance_vector, qcfc.correlation, nan_policy="omit")
    return np.abs(r)
