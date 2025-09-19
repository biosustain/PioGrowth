import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter

st.cache_data()


def plot_growth_data(df_long: pd.DataFrame):
    """Plot optical density (OD) growth data."""
    units = df_long["pioreactor_unit"].nunique()
    fig, axes = plt.subplots(
        units, 1, figsize=(10, 2 * units), sharey=True, sharex=True, squeeze=False
    )
    axes = axes.flatten()
    # grid container (reactive to UI changes)
    for (label, group_df), ax in zip(df_long.groupby("pioreactor_unit"), axes):
        group_df.plot.scatter(
            x="timestamp",
            y="od_reading",
            rot=45,
            ax=ax,
            title=f"Reactor {label}",  # Customize legend text here
        )
    ax = axes[-1]
    fig = ax.get_figure()
    fig.tight_layout()

    date_form = DateFormatter("%Y-%m-%d %H:%M")
    _ = ax.xaxis.set_major_formatter(date_form)
    return fig


def plot_growth_data_w_mask(
    df_wide: pd.DataFrame,
    df_mask: pd.DataFrame,
) -> plt.Figure:
    """Plot optical density (OD) growth data."""
    # ?check that index is datetime and columns are numeric?

    units = df_wide.shape[1]
    fig, axes = plt.subplots(
        units, 1, figsize=(10, 2 * units), sharey=True, sharex=True, squeeze=False
    )
    axes = axes.flatten()
    df_columns = df_wide.columns
    index_name = df_wide.index.name
    df_wide = df_wide.loc[df_mask.index].reset_index()
    df_mask = df_mask.reset_index()
    # grid container (reactive to UI changes)
    for col, ax in zip(df_columns, axes):
        mask = df_mask[col]
        # plot kept values in blue
        df_wide.loc[~mask].plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c="blue",
            ax=ax,
            alpha=0.1,
            s=1,
            title=f"Reactor: {col}",  # Customize legend text here
        )
        # Plot removed values in red
        df_wide.loc[mask].plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c="red",
            ax=ax,
            alpha=1.0,
            s=2,
            title=f"Reactor: {col}",  # Customize legend text here
        )
    ax = axes[-1]
    fig = ax.get_figure()
    fig.tight_layout()

    date_form = DateFormatter("%Y-%m-%d %H:%M")
    _ = ax.xaxis.set_major_formatter(date_form)
    return fig


def plot_growth_data_w_peaks(
    df_wide: pd.DataFrame,
    peaks: pd.DataFrame,
) -> plt.Figure:
    """Plot optical density (OD) growth data."""
    # ?check that index is datetime and columns are numeric?

    units = df_wide.shape[1]
    fig, axes = plt.subplots(
        units, 1, figsize=(10, 2 * units), sharex=True, squeeze=False
    )
    axes = axes.flatten()
    df_columns = df_wide.columns
    index_name = df_wide.index.name
    # grid container (reactive to UI changes)
    df_wide = df_wide.reset_index()
    for i, (col, ax) in enumerate(zip(df_columns, axes)):
        # plot kept values in blue
        df_wide.plot.scatter(
            x=index_name,
            y=col,
            rot=45,
            c=f"C{i}",
            ax=ax,
            alpha=0.1,
            s=1,
            title=f"Reactor: {col}",  # Customize legend text here
        )
        # Plot removed values in red
        peak_times = peaks[col].dropna().index
        for timepoint in peak_times:
            ax.axvline(x=timepoint, color="red", alpha=0.5, linestyle="--")
    ax = axes[-1]
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    _ = ax.xaxis.set_major_formatter(date_form)
    fig = ax.get_figure()
    fig.tight_layout()

    return fig, axes


def plot_fitted_data(splines, titles=None, ylabel="OD readings"):
    rows = (splines.shape[-1] + 1) // 2
    axes = splines.plot.line(
        style=".",
        ms=2,
        subplots=True,
        layout=(-1, 2),
        sharex=True,
        sharey=False,
        title=titles,
        ylabel=ylabel,
        xlabel="timepoints (rounded)",
        legend=False,
        figsize=(10, rows * 2),
    )
    _axes = axes.flatten()
    fig = _axes[-1].get_figure()
    fig.tight_layout()
    return fig, axes


def plot_derivatives(derivatives: pd.DataFrame, titles=None) -> plt.Figure:
    rows = (derivatives.shape[-1] + 1) // 2
    axes = derivatives.plot.line(
        style=".",
        ms=2,
        subplots=True,
        layout=(-1, 2),
        title=titles,
        ylabel="1st derivative",
        xlabel="timepoints (rounded)",
        legend=False,
        sharex=True,
        sharey=False,
        figsize=(10, rows * 2),
    )
    _axes = axes.flatten()
    decimal_place = int(np.log10(derivatives.max().min()) - 1)
    if decimal_place < 1:
        decimal_place = f"%.{np.abs(decimal_place)}f"
    else:
        decimal_place = "%.1f"
    for ax in _axes:
        _ = ax.yaxis.set_major_formatter(FormatStrFormatter(decimal_place))
    ax = _axes[-1]
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, axes
