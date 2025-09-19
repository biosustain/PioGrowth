from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import make_splrep, splev

SmoothingRange = namedtuple("SmoothingRange", ["s_min", "s", "s_max"])


def get_smoothing_range(m: int):
    """
    Compute the smoothing range for B-spline fitting in scipy interpolate functionality.
    """
    s_min, s, s_max = int(m - np.sqrt(2 * m)), m, int(m + np.sqrt(2 * m))
    s = SmoothingRange(s_min, s, s_max)
    return s


def fit_spline_and_derivatives(
    s: pd.Series,
    smoothing_factor: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B-splines to each column in the DataFrame and compute specified derivatives.
    Values cannot be missing as NaNs, i.e. on rolling median of data.

    Parameters
    ----------
    s: pd.Series
        Input Series with time series data
    smoothing_factor: float
        Smoothing factor for the spline fitting.
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the fitted spline
                                 and its derivatives.
    """
    # drop NaN values
    s = s.dropna()

    x = (s.index - s.index[0]).total_seconds().to_numpy()

    bspl = make_splrep(
        x,
        s,
        s=smoothing_factor,
        k=3,
    )
    s_fitted = pd.Series(
        splev(x, bspl),
        index=s.index,
    )

    # for order in derivative_ord_ers:
    der = bspl.derivative(nu=1)
    s_first_derivative = pd.Series(
        der(x),
        index=s.index,
    )

    return s_fitted, s_first_derivative


def fit_spline_and_derivatives_one_batch(
    df: pd.DataFrame,
    smoothing_factor: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B-splines to each column in the DataFrame and compute specified derivatives.
    Values cannot be missing as NaNs, i.e. on rolling median of data.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame with time series data.
    smoothing_factor: float
        Smoothing factor for the spline fitting.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the fitted spline
                                 and its derivatives.
    """
    assert df.isna().sum().sum() == 0, "Input DataFrame contains NaN values"
    df_fitted = pd.DataFrame(index=df.index)
    df_first_derivative = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col]
        s_fitted, s_first_derivative = fit_spline_and_derivatives(s, smoothing_factor)
        df_fitted[f"{col}"] = s_fitted
        df_first_derivative[f"{col}"] = s_first_derivative

    return df_fitted, df_first_derivative
