# %% [markdown]
# # Estimating growth rates
# Outline approach used in the app to estimate the growth rate of
# biological growth gurves with lag phase.
#
# Without lag-phase and ranging from zero to one, growth curves can be formulated as:
#
# Fitting common growth models using a squared-error loss function
# (using `scipy`)
# - `t_0` is the shift of the center of the modeled curve (should correspond to
#    maximum growth)
# - `r` is the rate in the exponential, which gives the rate (not sure if this can
#    always be used to model exponential growth)
# - `a` is the saturation maximum in the growth curve
#
# Finding the maximum growth using splines
# - probably less optimal, but maximum growth range seems to work.
# - slope of sliding window on log-transformed data should give the growth rate
#
# The $\mu_{max}$ growth is the minima of the second derivative.

# %% [markdown]
# ## Setup

# %%
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from piogrowth.durations import find_max_range
from piogrowth.fit import fit_spline_and_derivatives_one_batch, get_smoothing_range


class GrowthParams(NamedTuple):
    a: float
    r: float
    t0: float
    shift_y: float

    def __repr__(self):
        return (
            f"GrowthParams(a={self.a:.3f}, r={self.r:.3f},"
            f" t0={self.t0:.3f}, shift_y={self.shift_y:.3f})"
        )


def generate_growth_curve(
    time_points: np.array,
    lag_duration: float = 2.0,
    growth_rate: float = 0.5,
    max_population: float = 1.0,
    shift: float = 0.0,
    noise_level: float = 0.02,
    random_seed: int = None,
    non_negative: bool = True,
    log_transform: bool = False,
    epsilon: float = 0.0001,
):
    """
    Generate biological growth curve with lag phase using modified logistic equation.

    Standard logistic: N(t) = K / (1 + ((K-N₀)/N₀) * exp(-r*t))
    With lag phase: Growth starts at effective time t_eff = t - lag_duration

    Parameters:
    -----------
    time_points : array-like
        Time points at which to evaluate the growth curve
    lag_duration : float
        Duration of the lag phase (same units as time_points)
    growth_rate : float
        Intrinsic/maximum specific growth rate r (1/time)
    max_population : float
        Carrying capacity K / maximum population size
    shift : float
        Vertical shift (e.g., for background OD)
    noise_level : float
        Standard deviation of Gaussian noise
    random_seed : int, optional
        Random seed for reproducibility
    non_negative : bool
        Whether to enforce non-negative population values
    log_transform : bool
        Whether to apply natural log transform to the output
    epsilon : float
        Small offset added before log to avoid log(0)

    Returns:
    --------
    population : numpy array
        Population values with noise at each time point
    population_clean : numpy array
        Population values without noise (for comparison)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    time_points = np.array(time_points)

    # Logistic growth equation with lag phase
    # During lag: keep population near initial with minimal growth
    # After lag: apply standard logistic starting from t_eff = 0
    adjusted_time = time_points - lag_duration
    population_clean = max_population / (1 + np.exp(-growth_rate * adjusted_time))

    # Add vertical shift (e.g., background OD reading)
    if shift > 0.0:
        population_clean = population_clean + shift

    # Add Gaussian noise
    noise = np.random.normal(
        0,
        noise_level * max_population,
        size=len(time_points),
    )
    population = population_clean + noise

    # Ensure non-negative
    if non_negative:
        population = np.maximum(population, epsilon)

    # Optional log transform (for exponential phase detection)
    if log_transform:
        population = np.log(population + epsilon)
        population_clean = np.log(population_clean + epsilon)

    return population, population_clean


def plot_simulated_growth_curve(
    time: Iterable[float],
    pop_clean,
    pop_noisy,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, pop_clean, "b-", linewidth=2, label="Clean signal")
    ax.plot(time, pop_noisy, "r.", alpha=0.5, label="With noise")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Population (OD or cells/mL)")
    ax.set_title("Biological Growth Curve with Lag Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


from scipy.signal import savgol_filter


def smooth(y: np.ndarray[float], window: int = 21, poly: int = 1, passes: int = 1):
    """Smooth a series with Savitzky-Golay filtering (odd window, multi-pass)."""
    n = y.size
    if n < 7:
        return y
    w = int(window) | 1  # odd
    w = min(w, n if n % 2 else n - 1)
    p = min(int(poly), w - 1)
    for _ in range(int(passes)):
        y = savgol_filter(y, w, p, mode="interp")
    return y


# %% [markdown]
# Model equations


# %%
def logistic_model(t, a, r, t0, shift=0.0):
    """Idealized growth curve model with lag phase."""
    u = np.exp(-r * (t - t0))
    return a / (1 + u) + shift


def logistic_model_d1(t, a, r, t0):
    """Idealized first-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return a * r * (u / (1 + u) ** 2)


def logistic_model_d2(t, a, r, t0):
    """Idealized second-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return a * r * (u * (u - 1) / (1 + u) ** 3)


def gompertz_model(t, a, r, lag):
    """Gompertz growth model: y(t) = a * exp(-exp(-r * (t - lag)))"""
    return a * np.exp(-np.exp(-r * (t - lag)))


def richards_model(t, a, r, lag, nu):
    """Richards growth model (generalized logistic with shape parameter nu)

    Generalized logistic function: https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    power = 1 / nu
    return a / ((1 + nu * np.exp(-r * (t - lag))) ** power)


# %% [markdown]
# ## S-Shaped growth curve (Logistic)
# Generate time points:
# - 2,880 would we every 30seconds for a day
# - 17,280 would be every 5seconds for a day

# %%
time_in_h = np.linspace(0, 24, 2880)
# Generate growth curve
max_population = 1.5
lag_duration = 10
growth_rate = 0.6
shift_y = 0.2
noise_level = 0.03

pop_noisy, model_curve = generate_growth_curve(
    time_points=time_in_h,
    lag_duration=lag_duration,
    growth_rate=growth_rate,
    max_population=max_population,
    shift=shift_y,
    noise_level=noise_level,
    random_seed=42,
)

# %% [markdown]
# Plot the model with the parameters specified above

# %%
fig, ax = plot_simulated_growth_curve(time_in_h, model_curve, pop_noisy)


# %% [markdown]
# ## Curve fitting on noisy data
# Fitted parameters with noisy data

# %%
params_fitted, pcov = curve_fit(
    logistic_model, time_in_h, pop_noisy, p0=[1.0, 0.5, 3.0, 0.0]
)
GrowthParams(*params_fitted)

# %% [markdown]
# Fitted parameters with smoothed data (removing some noise)
# - difference is here minimal with low and homoscedastic noise

# %%
params_fitted, pcov = curve_fit(
    logistic_model, time_in_h, smooth(pop_noisy, window=31), p0=[1.0, 0.5, 3.0, 0.0]
)
GrowthParams(*params_fitted)

# %% [markdown]
# ## Compare different growth models
# Fitting the noisy data from logistic model

# %% [markdown]
# Fitting the noisy data with all models

# %%
# to move to pkg

model_functions = {
    "Logistic": logistic_model,
    "Gompertz": gompertz_model,
    "Richards": richards_model,
}

inital_parameters = {
    "Logistic": [1.0, 0.5, 3.0, 0.0],
    "Gompertz": [1.0, 0.5, 3.0],
    "Richards": [1.0, 0.5, 3.0, 2.0],
}

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time_in_h, model_curve, alpha=1, label="model (w./o. noise)")

for model_name, model_func in model_functions.items():
    p0 = inital_parameters[model_name]
    params_fitted, pcov = curve_fit(model_func, time_in_h, pop_noisy, p0=p0)
    model_fit = model_func(time_in_h, *params_fitted)
    ax.plot(time_in_h, model_fit, label=f"Fitted: {model_name}", alpha=0.3)
    msg_params_fitted = ", ".join(f"{_p:.3f}" for _p in params_fitted)
    print(f"{model_name} fitted parameters: {msg_params_fitted}")
_ = ax.legend()

# %% [markdown]
# A huge condition number would indicate a problem with parameter identifiability
# - huge >>> 1e12

# %%
np.linalg.cond(pcov)

# %% [markdown]
# ### Derivatives and maxima
# does not help to find $\mu_{max}$, but could be used with first derivative as formula

# %%
from scipy.differentiate import derivative

der1 = derivative(logistic_model, time_in_h, args=(1.502, 0.598, 9.978, 0.199))
der1.df

# %%
time_in_h[der1.df.argmax()]

# %% [markdown]
# ## Prepare data for PioReactor-like analysis
# - create datetime index
# - apply rolling median smoothing (could be Savitzky-Golay filtering)

# %% [markdown]
# Convert hours (range of 24h) to timedelta and move to start data
#

# %%
start_datetime = pd.Timestamp("2025-11-21 08:00")
time_series = pd.Series(pd.to_timedelta(time_in_h, unit="h")) + start_datetime
time_series = pd.Series(time_series, dtype="datetime64[s]")
time_series

# %%
# View data in format imported from PioReactor software output
# - use the rounded timestamp as index which is created as a preprocessing step

# %%
df = pd.DataFrame(
    {
        "timestamp_rounded": time_series,
        "Reactor": pop_noisy,
    }
).set_index("timestamp_rounded")
df

# %% [markdown]
# Apply rolling median to smooth the data
# - 31 consecutives timepoints are used, setting the 16th value as the smoothed value

# %%
rolling_window = 31  # in number of samples
min_periods = 15
df_rolling = df.rolling(
    rolling_window,
    min_periods=min_periods,
    center=True,
).median()
df_rolling

# %% [markdown]
# Plot median smoothed data against Savitzky-Golay smoothed data
# - what happens in the case of outliers? (to be tested)
# - are results comparable?
#
# For now we will continue with the rolling median smoothed data

# %%
ax = df_rolling.plot(figsize=(7.4, 4), legend=False)
_ = ax.legend(["Smoothed data: rolling median"])
pd.Series(smooth(df["Reactor"], window=31, passes=1), index=df.index).plot(ax=ax)
_ = ax.legend(["Smoothed data: rolling median", "Smoothed data: Savitzky-Golay"])

# %% [markdown]
# ## Fit a linear model on the data
# - rolling window
# - maximum overall growth: maximum of first derivative
# - $mu_{max}$ growth is the maximum of the second derivative, or the maximum of the
#   log transformed data slope (estimated with a sliding window)

# %%
import numpy as np

# Fit linear model to original series
from sklearn.linear_model import LinearRegression

series_window = df_rolling.squeeze().loc["2025-11-21 15:00":"2025-11-21 22:00"]


def timeindex_to_hours(index: pd.DatetimeIndex):
    """
    Convert a pandas DatetimeIndex to elapsed hours since the first timestamp.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The DatetimeIndex to convert.

    Returns
    -------
    numpy.ndarray
        Array of elapsed hours (float) since the first timestamp in the index,
        reshaped as a column vector.
    """
    x = (index - index[0]).total_seconds().to_numpy() / 3600
    return


def fit_linear_growth(s: pd.Series):
    """
    Fit a linear regression model to a time series window.

    Parameters
    ----------
    series_window : pd.Series
        Time-indexed series of population measurements.

    Returns
    -------
    linreg : LinearRegression
        Fitted linear regression model.
    """
    X = (s.index - s.index[0]).total_seconds().to_numpy().reshape(-1, 1) / 3600  # hours
    y = s.values.reshape(-1, 1)
    linreg = LinearRegression().fit(X, y)
    return linreg


def get_slope(s: pd.Series):
    linreg_fitted = fit_linear_growth(s)
    return linreg_fitted.coef_[0][0]


def get_tangent(s, max_idx, max_slope, window_hours=1):
    """
    Compute the tangent line around the maximum slope point in log2-transformed data.

    Parameters
    ----------
    s : pd.Series
        Data with datetime index.
    max_idx : pd.Timestamp
        Index of the maximum slope point.
    s_log2 : pd.DataFrame or pd.Series
        Log2-transformed smoothed data.
    max_slope : float
        Maximum slope value.
    window_hours : float, optional
        Window size in hours around max_idx to show the tangent (default: 1).

    Returns
    -------
    tangent : np.ndarray
        Array with tangent values (NaN outside the window).
    """
    x = (s.index - s.index[0]).total_seconds() / 3600  # hours
    x0 = (max_idx - s.index[0]).total_seconds() / 3600
    y0 = (
        s.loc[max_idx].values[0]
        if hasattr(s.loc[max_idx], "values")
        else s.loc[max_idx]
    )

    # Mask for ±window_hours around max_idx
    mask = (x >= x0 - window_hours) & (x <= x0 + window_hours)
    tangent = np.full_like(x, np.nan, dtype=float)
    tangent[mask] = max_slope * (x[mask] - x0) + y0
    return tangent


# %% [markdown]
# ### (Non-) transformed data
#


# %%
s_log2 = np.log2(df_rolling.squeeze())
s_normal = df_rolling.squeeze()
slopes_log2 = s_log2.rolling(
    rolling_window + 100,
    min_periods=min_periods,
    center=True,
).apply(get_slope)

# Find max slope and its location
max_idx = slopes_log2.idxmax()
max_slope = slopes_log2.max()

tangent = get_tangent(s_log2, max_idx=max_idx, max_slope=max_slope)

# Plot original data and tangent at max (only in window)
fig, ax = plt.subplots(figsize=(8, 5))
s_normal.plot(ax=ax, label="Original data (rolling median)")
s_log2.plot(ax=ax, label="Log2 transformed data")
ax.plot(df_rolling.index, tangent, "--", label="Tangent at max slope (±1h) (log2)")
ax.scatter([max_idx], [s_log2.loc[max_idx]], color="red", zorder=3)
ax.legend()
ax.set_title("Original data with tangent at maximum slope")
plt.show()

# %% [markdown]
# maximum timepoint estimation with sliding window is not perfect. small errors can lead
# to deviations.

# %%
print(f"shift in x of logistic funtion: {lag_duration}")
max_idx - df_rolling.index[0]

# %%
center_time = df_rolling.index[0] + pd.Timedelta(hours=lag_duration)
print(f"center of logistic curves should be: {center_time}")
slopes_log2.squeeze().nlargest(5)

# %%
slopes_normal = s_normal.rolling(
    rolling_window + 100,
    min_periods=min_periods,
    center=True,
).apply(get_slope)
slopes_normal.nlargest(5)

# %%
max_idx = slopes_normal.idxmax()
max_slope = slopes_normal.max()

tangent = get_tangent(s_normal, max_idx=max_idx, max_slope=max_slope)
ax.plot(s_normal.index, tangent, "--", label="Tangent at max slope (±1h)")
ax.scatter(
    [max_idx],
    [s_normal.loc[max_idx]],
    color="red",
    zorder=4,
)
ax.legend()
fig


# %%
print(f"shift in x of logistic funtion: {lag_duration}")
max_idx - df_rolling.index[0]

# %%
center_time = df_rolling.index[0] + pd.Timedelta(hours=lag_duration)
print(f"center of logistic curves should be: {center_time}")
slopes_normal.squeeze().nlargest(15)

# %% [markdown]
# maximum timepoint estimation with sliding window is not perfect. small errors can lead
# to deviations.
#
# Looks like the growth rate in the model and from the slopes are hard to convenve.

# %% [markdown]
# if the data would have been log-transformed, exponential growth rate would be roughly
#
# $$N(t) = N_0 \cdot e^{rt}$$
#
# $$\log_2 N(t) = \log_2 \left( N_0 \cdot e^{rt} \right)$$
#
# $$\log_2 N(t) = \log_2 N_0 + rt \cdot \log_2 e$$
#
# $$\log_2 N(t) = \log_2 N_0 + \frac{r}{\ln 2}$$
#
# To recover r from the slope b, multiply by $ln(2)$.
#
# which does not work for non-transformed data

# %%
print(f"Growth rate in model:  {growth_rate:.3f}")
print(f"Non-transformed slop:  {slopes_normal.max():.3f}")
print(f"Non-transformed est. r:  {slopes_normal.max() * np.log(2):.3f}")
print(f"Log2-transformed slope: {slopes_log2.max():.3f}")
print(f"Log2-transformed est. r: {slopes_log2.max() * np.log(2):.3f}")

# %%
print("Median slope around center point (plus-minus 10mins)")
_v = slopes_normal.squeeze().loc["2025-11-21 17:50":"2025-11-21 18:10"].median()
print(f"non-transformed slope at center: {_v:.3f}")
_v = slopes_log2.squeeze().loc["2025-11-21 17:50":"2025-11-21 18:10"].median()
print(f"log-transformed slope at center: {_v:.3f}")

# %% [markdown]
# ## Use spline to find the maxium growth
# -

# %% [markdown]
# Fit spline to the smoothed data and calculate derivatives
# - use smoothing factor based on data length

# %%
spline_smoothing_value = get_smoothing_range(df_rolling.shape[0])
spline_smoothing_value

# %% [markdown]
# Identify time points where growth rate is in the top 90%
# - could be used to estimat the growth rate during maximum growth phase


# %%
from pandas.api.types import is_datetime64_any_dtype
from scipy.interpolate import make_splrep, splev


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

    if len(s) < 4:
        raise ValueError(
            "Not enough data points to fit a spline. Need at least 4 non-NaN values."
        )
    if not is_datetime64_any_dtype(s.index.dtype):
        raise TypeError("Index of the input Series must be datetime type.")
    x = (s.index - s.index[0]).total_seconds().to_numpy() / 3_600  # convert to hours

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

    der2 = bspl.derivative(nu=2)
    s_second_derivative = pd.Series(
        der2(x),
        index=s.index,
    )

    return s_fitted, s_first_derivative, s_second_derivative


def plot_fitted_data(
    splines: pd.Series,
    der1: pd.Series,
    der2: pd.Series,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 12))
    splines.plot(ax=ax, title="Fitted Spline", color="blue")
    ax2 = ax.twinx()
    der1.plot(ax=ax2, color="green", label="First Derivative")
    der2.plot(ax=ax2, color="orange", label="Second Derivative")
    # ax2.set_ylabel("Derivatives")
    # der1.plot(ax=ax2, title="First Derivative", color="green")
    # der2.plot(ax=ax2, title="Second Derivative", color="orange")
    ax.legend(["Fitted Spline"])
    ax2.legend(["First Derivative", "Second Derivative"])
    fig.tight_layout()
    return fig, ax


splines, der1, der2 = fit_spline_and_derivatives(
    df_rolling.squeeze(), smoothing_factor=5.0
)
fig, ax = plot_fitted_data(splines, der1, der2)

# %%
inflection_points = df_rolling.iloc[[der2.argmax(), der2.argmin()]]

# %%
max_min_growth_change = [der2.argmax(), der2.argmin()]
der1.iloc[max_min_growth_change]

# %%
inflection_points.index - df_rolling.index.min()

# %%
high_percentage_treshold = 95
splines, derivatives = fit_spline_and_derivatives_one_batch(
    df_rolling,
    smoothing_factor=spline_smoothing_value.s,
)
prop_high = high_percentage_treshold / 100
cutoffs = derivatives.max() * prop_high
in_high_growth = derivatives.ge(cutoffs, axis=1)
max_time_range = in_high_growth.apply(find_max_range, axis=0).T.convert_dtypes()
derivatives.describe()

# %%
t_to_max_in_h = (derivatives.idxmax() - derivatives.index.min()).dt.seconds / 3_600
t_to_max_in_h

# %% [markdown]
# - lag-phase duration should be estimated (maybe from first derivative plot?)
# - max population can be measured using OD
# - min population can be measured using OD
#
# use `od_max` from `df_rolling`

# %%
t_max_corrected = t_to_max_in_h - lag_duration
od_max = df_rolling.loc[derivatives.idxmax()].squeeze()
max_population / (
    1
    # scaling term is interesting
    + np.exp(-growth_rate * (t_max_corrected))
)

# %% [markdown]
# Find time range of maximum derivative and plot the region


# %%
def add_region_high_growth(ax, time_range, use_elapsed_time=False):
    """Add shaded region to indicate high growth phase."""
    start, end = time_range["start"], time_range["end"]
    if pd.isna(start) or pd.isna(end):
        return
    if use_elapsed_time:
        start = (start - df_rolling.index[0]).total_seconds() / 3600
        end = (end - df_rolling.index[0]).total_seconds() / 3600
    ax.axvspan(
        start,
        end,
        color="gray",
        alpha=0.3,
        label="High growth phase",
    )
    return ax


in_high_growth = derivatives.ge(cutoffs, axis=1)
max_range = find_max_range(in_high_growth["Reactor"])
fig, ax = plot_simulated_growth_curve(time_in_h, model_curve, pop_noisy)
_ = add_region_high_growth(ax, max_range, use_elapsed_time=True)

# %% [markdown]
# ## Without lag phase
# Now calculate exponential growth in the maximum growth area:

# %%
view = df_rolling.loc[max_range.start : max_range.end]
np.log(view).plot()

# %%
od_log = np.log(view["Reactor"])
X = (od_log.index - od_log.index[0]).total_seconds().values.reshape(
    -1, 1
) / 3600  # hours
y = od_log.values.reshape(-1, 1)

reg = LinearRegression().fit(X, y)
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]

print(f"Growth rate (slope): {slope:.4f} per hour")
print(f"Intercept: {intercept:.4f}")

# Plot regression line
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(X, y, "o", label="Log(OD)")
ax.plot(X, reg.predict(X), "r-", label="Linear fit")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Log(OD)")
ax.legend()


# %%
