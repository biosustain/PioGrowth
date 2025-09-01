import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter
import numpy as np

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
