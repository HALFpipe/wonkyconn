import numpy as np
from numba import guvectorize
from numpy import typing as npt
from scipy import stats


def correlation_p_value(r: npt.NDArray[np.float64], m: int) -> npt.NDArray[np.float64]:
    ab = m / 2 - 1
    distribution = stats.beta(ab, ab, loc=-1, scale=2)
    pvalue = 2 * (distribution.sf(np.abs(r)))
    return pvalue


@guvectorize(
    ["void(float64[:], float64[:], float64[:, :], float64[:], int64[:])"],
    "(n),(n),(n,m)->(),()",
    nopython=True,
)
def partial_correlation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    r: npt.NDArray[np.float64],
    count: npt.NDArray[np.int64],
) -> None:
    """A minimal implementation of partial correlation.

    Args:
        x (npt.NDArray[np.float64]): Variable of interest.
        y (npt.NDArray[np.float64]): Variable of interest.
        cov (npt.NDArray[np.float64]): Covariates to regress out.
        r (npt.NDArray[np.float64]): Output array for the correlation coefficient.
        count (npt.NDArray[np.int64]): Output array for the number of valid observations.
    """

    # Remove rows with NaN values
    mask = np.array([np.all(np.isfinite(row)) for row in np.column_stack((x, y, cov))])

    if not mask.any():
        r[0] = np.nan
        count[0] = 0
        return

    x = x[mask]
    y = y[mask]
    cov = cov[mask]

    beta_cov_x, _, _, _ = np.linalg.lstsq(cov, x)
    beta_cov_y, _, _, _ = np.linalg.lstsq(cov, y)
    resid_x = x - cov @ beta_cov_x
    resid_y = y - cov @ beta_cov_y
    r[0] = np.corrcoef(resid_x, resid_y)[0, 1]
    count[0] = mask.sum()
