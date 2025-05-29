import numpy as np
import scipy
from numba import guvectorize
from numpy import typing as npt


def correlation_p_value(
    r: npt.NDArray[np.float64], m: int
) -> npt.NDArray[np.float64]:
    ab = m / 2 - 1
    distribution = scipy.stats.beta(ab, ab, loc=-1, scale=2)
    pvalue = 2 * (distribution.sf(np.abs(r)))
    return pvalue


@guvectorize(
    ["void(float64[:], float64[:], float64[:, :], float64[:])"],
    "(n),(n),(n,m)->()",
    nopython=True,
)
def partial_correlation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """A minimal implementation of partial correlation.

    Parameters
    ----------
    x, y : np.ndarray
        Variable of interest.

    cov : np.ndarray
        Variable to be removed from variable of interest.

    Returns
    -------
    dict
        Correlation and p-value.
    """
    beta_cov_x, _, _, _ = np.linalg.lstsq(cov, x)
    beta_cov_y, _, _, _ = np.linalg.lstsq(cov, y)
    resid_x = x - cov @ beta_cov_x
    resid_y = y - cov @ beta_cov_y
    out[0] = np.corrcoef(resid_x, resid_y)[0, 1]
