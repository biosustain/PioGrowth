# %% [markdown]
# # Simulating Biological Growth Curves with Lag Phase

# %%
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# %%
def generate_growth_curve(
    time_points: np.array,
    lag_duration: float = 2.0,
    growth_rate: float = 0.5,
    max_population: float = 1.0,
    initial_population: float = 0.01,
    noise_level: float = 0.02,
    random_seed: int = None,
    non_negative: bool = False,
    log_transform: bool = False,
):
    """
    Generate biological growth curve with lag phase.

    Parameters:
    -----------
    time_points : array-like
        Time points at which to evaluate the growth curve
    lag_duration : float
        Duration of the lag phase (same units as time_points)
    growth_rate : float
        Maximum specific growth rate (1/time)
    max_population : float
        Carrying capacity / maximum population size
    initial_population : float
        Initial population size
    noise_level : float
        Standard deviation of Gaussian noise (relative to signal)
    random_seed : int, optional
        Random seed for reproducibility
    non_negative : bool
        Whether to enforce non-negative population values
    log_transform : bool
        Whether to apply log10 transform to the output

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

    # Modified logistic growth with lag phase
    # Shift time by lag duration
    adjusted_time = time_points - lag_duration

    # Logistic growth equation
    population_clean = max_population / (
        1
        + ((max_population - initial_population) / initial_population)
        * np.exp(-growth_rate * adjusted_time)
    )

    # During lag phase, keep population close to initial
    lag_mask = time_points < lag_duration
    population_clean[lag_mask] = initial_population * (
        1 + 0.1 * (time_points[lag_mask] / lag_duration)
    )

    # Add Gaussian noise
    noise = np.random.normal(
        0,
        noise_level * max_population,
        size=len(time_points),
    )
    population = population_clean + noise
    # Ensure non-negative
    if non_negative:
        population = np.maximum(population, 0.0001)

    if log_transform:
        population = np.log10(population)
        population_clean = np.log10(population_clean)

    return population, population_clean


# Plot results
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


# Call the function to plot
# %% [markdown]
# Generate time points:
# - 2,880 would we every 30seconds for a day
# - 17,280 would be every 5seconds for a day

# %%
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
)
fig, ax = plot_simulated_growth_curve(time, pop_clean, pop_noisy)

# %%
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
