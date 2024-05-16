from pathlib import Path

import numpy as np
import pandas as pd

from wonkyconn.atlas import Atlas
from scipy.stats import spearmanr


def calculate_distance_dependence(qcfc: pd.DataFrame, atlas: Atlas) -> float:
    distance_matrix = atlas.get_distance_matrix()
    i, j = map(list, zip(*qcfc.index))
    distance_vector = distance_matrix[i, j]
    r, _ = spearmanr(distance_vector, qcfc.correlation)
    return r
