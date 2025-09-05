import numpy as np
import pandas as pd
from scipy.interpolate import make_splrep, splev


def smoothing_range(m: int):
    """
    Compute the smoothing range for B-spline fitting in scipy interpolate functionality.
    """
    s_min, s, s_max = int(m - np.sqrt(2 * m)), m, int(m + np.sqrt(2 * m))
    s = pd.Series([s_min, s, s_max], index=["s_min", "s", "s_max"])
    return s


def fit_spline_and_derivatives_no_nan(
    df: pd.DataFrame,
    smoothing_factor: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B-splines to each column in the DataFrame and compute specified derivatives.
    Values cannot be missing as NaNs, i.e. on rolling median of data.

    Args:
        df (pd.DataFrame): Input DataFrame with time series data.
        smoothing_factor (float): Smoothing factor for the spline fitting.
        derivative_orders (list[int]): List of derivative orders to compute.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing the fitted spline
                                 and its derivatives.
    """
    assert df.isna().sum().sum() == 0, "Input DataFrame contains NaN values"
    df_fitted = pd.DataFrame(index=df.index)
    df_first_derivative = pd.DataFrame(index=df.index)

    x = (df.index - df.index[0]).total_seconds().to_numpy()

    for col in df.columns:
        y = df[col].values
        bspl = make_splrep(
            x,
            y,
            s=smoothing_factor,
            k=3,
        )
        df_fitted[f"{col}"] = pd.Series(
            splev(x, bspl),
            index=df.index,
        )

        # for order in derivative_orders:
        der = bspl.derivative(nu=1)
        df_first_derivative[f"{col}"] = pd.Series(
            der(x),
            index=df.index,
        )

    return df_fitted, df_first_derivative
