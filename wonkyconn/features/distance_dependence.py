import numpy as np
import pandas as pd

from ..atlas import Atlas
from scipy.stats import spearmanr


def calculate_distance_dependence(qcfc: pd.DataFrame, atlas: Atlas) -> float:
    """
    Calculate the Spearman correlation between the distance matrix and the QC-FC correlation values.

    Parameters:
    - qcfc (pd.DataFrame): The qcfc DataFrame containing the correlation values with a multi-index of the lower triangular indices
    - atlas (Atlas): The Atlas object used to calculate the distance matrix.

    Returns:
    - float: The distance dependence value.

    """
    distance_matrix = atlas.get_distance_matrix()
    i, j = map(np.asarray, zip(*qcfc.index))
    distance_vector = distance_matrix[i, j]
    r, _ = spearmanr(distance_vector, qcfc.correlation, nan_policy="omit")
    return np.abs(r)
