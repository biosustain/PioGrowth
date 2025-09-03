import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter

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
