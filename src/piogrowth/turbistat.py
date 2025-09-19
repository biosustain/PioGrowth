import pandas as pd
from scipy.signal import find_peaks


def detect_peaks(
    series: pd.Series,
    distance: int = 300,
) -> pd.Series:
    """Detect peaks in a pandas Series using scipy's find_peaks function.

    Args:
        series (pd.Series): The input time series data.
        distance (int): Minimum horizontal distance (in number of samples)
                        between neighboring peaks.

    Returns:
        pd.Series: Detected peaks in the series.
    """
    s = series.dropna()
    peaks, _ = find_peaks(s, distance=distance, prominence=s.max() / 5)
    return s.iloc[peaks]
