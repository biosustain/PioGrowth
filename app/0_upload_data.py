from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from plots import plot_growth_data_w_mask

import piogrowth

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
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
    round_time = st.slider("Round time to nearest minute", 0, 15, 5, step=1)
    # Options for handeling negative OD readings
    st.write("Data filtering options:")
    filter_columns = st.columns(3)
    remove_zero = filter_columns[0].checkbox(
        "Remove zero OD readings",
        value=False,
    )
    remove_max = filter_columns[1].checkbox(
        "Remove maximum OD readings by quantile",
        value=False,
    )
    quantile_max = filter_columns[1].slider(
        "Max quantile for maximum removal",
        0.9,
        1.0,
        0.99,
        step=0.01,
    )
    iqr_range_value = filter_columns[2].slider(
        "IQR range for outlier removal",
        1.0,
        3.0,
        1.5,
        step=0.1,
    )
    rolling_window = filter_columns[2].slider(
        "Rolling window (in seconds)",
        11,
        61,
        31,
        step=2,
    )

    min_date, max_date = None, None
    if df_raw_od_data is not None:
        min_date, max_date = st.select_slider(
            "Select time window (inferred). This won't be plotted as filtered data.",
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
        f"- Loaded {df_raw_od_data.shape[0]:,d} rows "
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
    rerun = st.session_state.get("df_raw_od_data") is None
    st.session_state["df_raw_od_data"] = df_raw_od_data
    # re-run now with data set
    if rerun:
        # ? replace with callback function that creates the input form?
        st.rerun()

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

    msg += f"- Wide OD data with rounded timestamps to {round_time} seconds.\n"
    # wide data of raw data
    # - can be used in plot for visualization,
    # - and in curve fitting (where gaps would be interpolated)
    df_wide_raw_od_data = df_raw_od_data.pivot(
        index="timestamp_rounded",
        columns="pioreactor_unit",
        values="od_reading",
    )

    # skip first or last measurements based on user input (after first loading the data)
    # ! won't be plotted as filtered data
    if min_date:
        df_wide_raw_od_data = df_wide_raw_od_data.loc[min_date:max_date]
        st.info(f"Time range: {min_date} to {max_date}")

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    # Handle negative values
    if remove_zero:
        mask_negative = df_wide_raw_od_data_filtered < 0
        msg += (
            f"- Setting {mask_negative.sum().sum():,d} negative OD readings to NaN.\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_negative)
        masked = masked | mask_negative

    st.title("Processing summary of OD readings")
    if remove_max:
        mask_extreme_values = (
            df_wide_raw_od_data_filtered
            > df_wide_raw_od_data_filtered.quantile(quantile_max)
        )
        msg += (
            f"- Number of extreme values detected: {mask_extreme_values.sum().sum()}\n"
        )
        msg += f"   - in detail: {mask_extreme_values.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            mask_extreme_values
        )
        masked = masked | mask_extreme_values

    # outlier detection using IQR on rolling window: sets for center value of window a
    # true or false (this would be arguing maybe for long data format)
    # can be used in plot for visualization
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

    mask_outliers = (
        df_wide_raw_od_data_filtered.rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
            closed="both",
        )
        .apply(piogrowth.filter.out_of_iqr, kwargs={"factor": iqr_range_value})
        .astype(bool)
    )
    # st.write(f"### Number of outliers detected: {mask_outliers.sum().sum()}")
    msg += f"- Number of outliers detected: {mask_outliers.sum().sum()}\n"
    msg += f"   - in detail: {mask_outliers.sum().to_dict()}\n"
    masked = masked | mask_outliers

    # apply mask to entire dataframe

    df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_outliers)

    masked = masked.convert_dtypes()
    st.write(msg)
    # fpath = Path(f"playground/data/{custom_id}_masked_values.csv")
    # fpath.parent.mkdir(exist_ok=True, parents=True)
    # masked.to_csv(fpath)
    # df_wide_raw_od_data.to_csv(Path(f"playground/data/{custom_id}_raw_wide_data.csv"))

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
    fig = plot_growth_data_w_mask(df_wide_raw_od_data, masked)
    with container_figures:
        st.write(fig)

    st.header(f"Rolling median in window of {rolling_window}s using filtered OD data")
    df_rolling = df_wide_raw_od_data_filtered.rolling(
        rolling_window,
        min_periods=min_periods,
        center=True,
    ).median()
    st.write(df_rolling)
    ax = df_rolling.plot.line(style=".", ms=2)
    st.write(ax.get_figure())

st.markdown("### Store in QuervE format")

convert = st.button("Store in QuervE format", key="store_in_querve")

if convert:
    st.warning("fct to convert to QuervE.")
    # store in state
