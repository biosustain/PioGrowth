import numpy as np
import pandas as pd
import streamlit as st
from plots import plot_growth_data

import piogrowth

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
rolling_window = st.session_state.get("rolling_window", 31)
min_periods = st.session_state.get("min_periods", 5)

st.title("Upload Data")
container_download_example = st.empty()

########################################################################################
# Upload Form

with st.form("Upload_data_form", clear_on_submit=False):

    file = st.file_uploader(
        "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
        type=["csv"],
        # needs callback to clear session state
    )
    custom_id = st.text_input(
        "Enter custom ID for data",
        max_chars=30,
        value=custom_id,
    )
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    filter_option = col1.radio(
        "Select if selected reactors are to be kept or removed", ("Remove", "Keep")
    )
    if st.session_state.get("df_raw_od_data") is None:
        reactors_selected = col2.text_input(
            "Enter reactors to filter (comma separated)", ""
        )
    else:
        # update possible reactors in form with available reactors
        col2.empty()  # Clear previous text input
        reactors_selected = col2.multiselect(
            "Select reactors to filter",
            options=df_raw_od_data["pioreactor_unit"].unique(),
        )
    # Options for handeling negative OD readings
    remove_zero = st.checkbox("Remove zero OD readings", value=False)
    round_time = st.slider("Round time to nearest minute", 0, 15, 5, step=1)
    min_date, max_date = None, None
    if df_raw_od_data is not None:
        min_date, max_date = st.select_slider(
            "Select time window (inferred)",
            options=df_raw_od_data["timestamp_rounded"],
            value=(
                df_raw_od_data["timestamp_rounded"].min(),
                df_raw_od_data["timestamp_rounded"].max(),
            ),
        )
    if df_raw_od_data is None:
        st.form_submit_button("Process file", type="primary")
    else:
        st.form_submit_button("Re-process file", type="primary")
########################################################################################
# Raw data and plots

st.header("Raw OD data")
container_raw_data = st.empty()
container_figures = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id


if file is not None:
    df_raw_od_data = piogrowth.load.read_csv(file)
    msg = (
        f" - Loaded {df_raw_od_data.shape[0]:,d} rows "
        f"and {df_raw_od_data.shape[1]:,d} columns.\n"
    )
    # round timestamp data
    df_raw_od_data.insert(
        0,
        "timestamp_rounded",
        df_raw_od_data["timestamp"].dt.round(
            f"{round_time}s",
        ),
    )
    st.session_state["df_raw_od_data"] = df_raw_od_data

    # Filter reactors (all measurements from selected reactors)
    if reactors_selected:
        st.write(f"Filtering reactors: {reactors_selected}")
        if isinstance(reactors_selected, str):
            # make it a list
            reactors_selected = reactors_selected.split(",")
            st.write(f"Filtering reactors: {reactors_selected}")
        mask = df_raw_od_data["pioreactor_unit"].isin(reactors_selected)
        if filter_option == "Remove":
            df_raw_od_data = df_raw_od_data.loc[~mask]
        else:
            df_raw_od_data = df_raw_od_data.loc[mask]

    # Handle negative values
    if remove_zero:
        mask_negative = df_raw_od_data["od_reading"] < 0
        msg += f" - Setting {mask_negative.sum():,d} negative OD readings to zero.\n"
        df_raw_od_data.loc[mask_negative, "od_reading"] = 0

    st.write(msg)

    st.header(f"Wide OD data with rounded timestamps to {round_time} seconds")
    # wide data of raw data, maybe removing outliers?
    # can be used in plot for visualization,
    # and in curve fitting (where gaps would be interpolated)
    df_wide_raw_od_data = df_raw_od_data.pivot(
        index="timestamp_rounded",
        columns="pioreactor_unit",
        values="od_reading",
    )
    if min_date:
        df_wide_raw_od_data = df_wide_raw_od_data.loc[min_date:max_date]
        st.info(f"Time range: {min_date} to {max_date}")

    st.write(df_wide_raw_od_data)
    # st.write(df_wide_raw_od_data.describe())
    ax = df_wide_raw_od_data.plot.line(style=".", ms=2, title="OD readings")
    st.write(ax.get_figure())

    # outlier detection using IQR on rolling window: sets for center value of window a
    # true or false (this would be arguing maybe for long data format)
    # can be used in plot for visualization
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    st.header(f"Rolling median in window of {rolling_window}s of OD data")
    df_rolling = df_wide_raw_od_data.rolling(
        rolling_window,
        min_periods=min_periods,
        center=True,
    ).median()
    st.write(df_rolling)

    # skip first N seconds or measurements
    # set rolling median to x seconds

    mask_extreme_values = df_wide_raw_od_data > df_wide_raw_od_data.quantile(0.99)
    st.write(
        f"### Number of extreme values detected: {mask_extreme_values.sum().sum()}"
    )
    st.write(mask_extreme_values.sum())
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.mask(mask_extreme_values)

    mask_outliers = (
        df_wide_raw_od_data_filtered.rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
            closed="both",
        )
        .apply(piogrowth.filter.out_of_iqr)
        .astype(bool)
    )
    st.write(f"### Number of outliers detected: {mask_outliers.sum().sum()}")
    st.write(mask_outliers)

    # apply mask to entire dataframe

    df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_outliers)
    st.write("### df_wide_raw_od_data_filtered")
    st.write(df_wide_raw_od_data_filtered)
    st.write(df_wide_raw_od_data_filtered.describe())
    ax = df_wide_raw_od_data_filtered.plot.line(
        style=".",
        title="OD readings with outliers removed",
        ms=2,
    )
    st.write(ax.get_figure())

else:
    with container_download_example:
        columns = st.columns([1, 2])
        columns[0].warning("no data uploaded.")
        columns[1].download_button(
            label="Download example  pioreactor experiment in csv format.",
            data=pd.read_csv("data/example_batch_data_od_readings.csv").to_csv(
                index=False
            ),
            file_name="example_batch_data_od_readings.csv",
            key="download_example_csv",
            mime="text/csv",
        )


with container_raw_data:
    st.dataframe(df_raw_od_data, use_container_width=True)

if df_raw_od_data is not None:
    # Download options
    fig = plot_growth_data(df_raw_od_data)
    with container_figures:
        st.write(fig)

    st.markdown("### Plot smoothed data")
    ax = df_rolling.plot.line(
        style=".", ms=2, title="Smoothed OD readings for reactors"
    )
    st.write(ax.get_figure())

st.markdown("### Store in QuervE format")

convert = st.button("Store in QuervE format", key="store_in_querve")

if convert:
    st.warning("fct to convert to QuervE.")
    # store in state
