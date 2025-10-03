import numpy as np
import scipy

from wonkyconn.correlation import (
    correlation_p_value,
    partial_correlation,
)


def test_correlation() -> None:
    n = 100
    m = 200
    x = np.random.normal(size=(n, m))
    x[0, 0] = np.nan
    y = np.random.normal(size=(m,))
    cov = np.random.normal(size=(m, 2))

    correlation, count = partial_correlation(x, y, cov)
    p_value = correlation_p_value(correlation, count)

    assert np.all(np.isfinite(correlation))

    for i in range(n):
        x_ = x[i, :]
        mask = np.isfinite(np.column_stack((x_, y, cov))).all(axis=1)
        x_ = x_[mask]
        y_ = y[mask]
        cov_ = cov[mask]

        beta_cov_x, _, _, _ = np.linalg.lstsq(cov_, x_, rcond=None)
        beta_cov_y, _, _, _ = np.linalg.lstsq(cov_, y_, rcond=None)
        resid_x = x_ - cov_.dot(beta_cov_x)
        resid_y = y_ - cov_.dot(beta_cov_y)
        r, p_val = scipy.stats.pearsonr(resid_x, resid_y)
        assert np.isclose(correlation[i], r)
        assert np.isclose(p_value[i], p_val)
