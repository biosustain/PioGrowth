import numpy as np
import pandas as pd
from growthcurves.preprocessing import out_of_iqr as gc_out_of_iqr
from growthcurves.preprocessing import out_of_iqr_window


def out_of_iqr(s: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return a boolean Series indicating whether each value is an outlier based
    on the IQR method."""
    center = s.iloc[len(s) // 2]
    if np.isnan(center):
        return False
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    # center point out of IQR?

    return (center < lower_bound) | (center > upper_bound)


if __name__ == "__main__":
    data = pd.Series([20, 1, 2, 3, 4, 5, 20, 6, 7, 8, 9, 10, 20])
    mask = data.rolling(window=5, center=True).apply(out_of_iqr, raw=False)
    print(mask)

    # does not work on series due to indexing using values.
    mask = data.rolling(window=5, center=True).apply(
        lambda s: out_of_iqr_window(
            s,
        ),
        raw=True,
    )
    print(mask)

    # would exclude the first and last 2 points as they are NaN, but we want to label
    # them as outliers if they are out of IQR in their respective windows
    mask = mask.astype(bool)
    print(mask)

    # series is understood by numpy and values are accessed.
    mask = gc_out_of_iqr(data, window_size=5)
    print(mask)

    mask = data.to_frame("S1").apply(gc_out_of_iqr, args=(5,))

    print(mask)
