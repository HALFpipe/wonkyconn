import numpy as np
import scipy

from wonkyconn.correlation import (
    correlation_p_value,
    partial_correlation,
)


def test_correlation() -> None:
    n = 100
    m = 100
    x = np.random.normal(size=(n, m))
    y = np.random.normal(size=(m,))
    cov = np.random.normal(size=(m, 2))

    correlation = partial_correlation(x, y, cov)
    p_value = correlation_p_value(correlation, m)

    for i in range(n):
        beta_cov_x, _, _, _ = np.linalg.lstsq(cov, x[i, :], rcond=None)
        beta_cov_y, _, _, _ = np.linalg.lstsq(cov, y, rcond=None)
        resid_x = x[i, :] - cov.dot(beta_cov_x)
        resid_y = y - cov.dot(beta_cov_y)
        r, p_val = scipy.stats.pearsonr(resid_x, resid_y)
        assert np.isclose(correlation[i], r)
        assert np.isclose(p_value[i], p_val)
