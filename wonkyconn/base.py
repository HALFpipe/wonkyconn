from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt


@dataclass
class ConnectivityMatrix:
    """
    Represents a connectivity matrix.

    Attributes:
        path (Path): The path to the ".tsv" file containing the connectivity matrix.
        metadata (dict[str, Any]): Additional metadata associated with the connectivity matrix.
    """

    path: Path
    metadata: dict[str, Any]

    def load(self) -> npt.NDArray[np.float64]:
        """
        Load the connectivity matrix from the file.

        Returns:
            ndarray: The loaded connectivity matrix as a NumPy array.
        """
        return np.loadtxt(self.path, delimiter="\t", skiprows=0)

    @cached_property
    def region_count(self) -> int:
        """
        Get the number of regions in the connectivity matrix.

        Returns:
            int: The number of regions.
        """
        return self.load().shape[0]
