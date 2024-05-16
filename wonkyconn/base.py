from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt


@dataclass
class ConnectivityMatrix:
    path: Path
    metadata: dict[str, Any]

    def load(self) -> npt.NDArray[np.float64]:
        return np.loadtxt(self.path, delimiter="\t", skiprows=1)
