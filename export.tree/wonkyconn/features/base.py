from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class MeanAndSEMResult:
    mean: float
    sem: float

    @classmethod
    def empty(cls) -> Self:
        return cls(mean=np.nan, sem=np.nan)
