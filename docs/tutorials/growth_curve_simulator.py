# %% [markdown]
# # Estimating growth rates
# Outline approach used in the app to estimate the growth rate of
# biological growth gurves with lag phase.
#
# Without lag-phase and ranging from zero to one, growth curves can be formulated as:
# - ...

# %% [markdown]
# ## Setup

# %%
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from piogrowth.durations import find_max_range
from piogrowth.fit import fit_spline_and_derivatives_one_batch, get_smoothing_range


def generate_growth_curve(
    time_points: np.array,
    lag_duration: float = 2.0,
    growth_rate: float = 0.5,
    max_population: float = 1.0,
    initial_population: float = 0.01,
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
    initial_population : float
        Initial population size N₀
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

    # Standard logistic growth formula:
    # N(t) = K / (1 + A * exp(-r*t))
    # where A = (K - N₀) / N₀
    A = (max_population - initial_population) / initial_population
    population_clean = max_population / (1 + A * np.exp(-growth_rate * adjusted_time))

    # During lag phase, override with slow linear growth
    # (prevents negative adjusted_time from causing exponential explosion backwards)
    lag_mask = time_points < lag_duration
    population_clean[lag_mask] = initial_population * (
        1 + 0.4 * (time_points[lag_mask] / lag_duration)
    )

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


def calculate_growth_rate(
    od_max: float,
    initial_population: float,
    t_max: float,
    lag_duration: float,
) -> float:
    """Calculate growth rate from max OD, initial population, and corrected max time.
    The estimate is based on the assumption of exponential growth after lag phase.

    Parameters:
    -----------
    od_max : float
        Maximum observed population (OD or cells/mL)
    initial_population : float
        Initial population size
    t_max : float
        Time to maximum growth rate

    Returns:
    --------
    growth_rate : float
        Estimated growth rate
    """
    t_max_corrected = t_max - lag_duration
    return (
        np.exp2((np.log2(od_max) - np.log2(initial_population)) / t_max_corrected) - 1
    )


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


# %% [markdown]
# ## S-Shaped growth curve
# Generate time points:
# - 2,880 would we every 30seconds for a day
# - 17,280 would be every 5seconds for a day

# %%
time = np.linspace(0, 24, 2880)
# Generate growth curve
max_population = 1.5
lag_duration = 3
initial_population = 0.05
growth_rate = 0.6

pop_noisy, pop_clean = generate_growth_curve(
    time_points=time,
    lag_duration=lag_duration,
    growth_rate=growth_rate,
    max_population=max_population,
    initial_population=initial_population,
    shift=1,
    noise_level=0.03,
    random_seed=42,
)
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)

# %%
pop_noisy, pop_clean = generate_growth_curve(
    time_points=time,
    lag_duration=lag_duration,
    growth_rate=growth_rate,
    max_population=max_population,
    initial_population=initial_population,
    shift=1,
    noise_level=0.03,
    random_seed=42,
    log_transform=True,
)
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)


# %%
# %%
pop_noisy, pop_clean = generate_growth_curve(
    time_points=time,
    lag_duration=lag_duration,
    growth_rate=growth_rate,
    max_population=max_population,
    initial_population=initial_population,
    noise_level=0.03,
    random_seed=42,
    log_transform=True,
)
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)

# %%
pop_noisy, pop_clean = generate_growth_curve(
    time_points=time,
    lag_duration=lag_duration,
    growth_rate=growth_rate,
    max_population=max_population,
    initial_population=initial_population,
    noise_level=0.03,
    shift=1,
    random_seed=42,
    log_transform=True,
)
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)

# %%
pop_noisy

# %% [markdown]
# Convert hours (range of 24h) to timedelta and move to start data
#

# %%
start_datetime = pd.Timestamp("2025-11-21 08:00")
time_series = pd.Series(pd.to_timedelta(time, unit="h")) + start_datetime
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

# %%
df_rolling.plot(figsize=(7.4, 4))

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
# recalculate the OD value based on the estimation

# %%
max_population / (
    1
    # scaling term is interesting
    + ((max_population - initial_population) / initial_population)
    * np.exp(-growth_rate * (t_to_max_in_h - lag_duration))
)

# %% [markdown]
# - lag-phase duration should be estimated (maybe from first derivative plot?)
# - max population can be measured using OD
# - min population can be measured using OD
#
# use `od_max` from `df_rolling`

# %%
t_max_corrected = t_to_max_in_h - lag_duration
od_max = df_rolling.loc[derivatives.idxmax()].squeeze()
factor = (max_population - initial_population) / initial_population
max_population / (
    1
    # scaling term is interesting
    + factor * np.exp(-growth_rate * (t_max_corrected))
)

# %% [markdown]
# Apporximate growth rate assuming exponential growth up to max growth time point
# - this needs to correctly estimate the end of the lag phase
#

# %%
np.exp((np.log(od_max / initial_population) / t_max_corrected)) - 1

# %%
np.exp2((np.log2(od_max) - np.log2(initial_population)) / t_max_corrected) - 1


# %%
calculate_growth_rate(
    od_max=od_max,
    initial_population=initial_population,
    t_max=t_to_max_in_h,
    lag_duration=3,
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
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)
add_region_high_growth(ax, max_range, use_elapsed_time=True)

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
t_in_h_exponential_growth = (max_range.end - max_range.start).total_seconds() / 3600
od_max = df_rolling.loc[max_range.end].squeeze()
initial_population = df_rolling.loc[max_range.start].squeeze()
calculate_growth_rate(
    od_max=od_max,
    initial_population=initial_population,
    t_max=t_in_h_exponential_growth,
    lag_duration=0,
)

# %% [markdown]
# ## Log-Transfrom data

# %%
# ToDo
time = np.linspace(0, 24, 2880)
# Generate growth curve
pop_noisy, pop_clean = generate_growth_curve(
    time_points=time,
    lag_duration=3.0,
    growth_rate=0.6,
    max_population=1.5,
    initial_population=0.05,
    noise_level=0.03,
    random_seed=42,
    non_negative=True,
    log_transform=True,
)

fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)
_ = ax.set_ylabel("Population log(OD or cells/mL)")

# %% [markdown]
# if the noise is evenly across all measurements, then small values
# get disproportionately magnified in the log space.
