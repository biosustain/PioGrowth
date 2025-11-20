import pandas as pd
import streamlit as st
from buttons import download_data_button_in_sidebar
from plots import plot_growth_data_w_mask

import piogrowth

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_wide_raw_od_data_filtered = st.session_state.get("df_wide_raw_od_data_filtered")
df_rolling = st.session_state.get("df_rolling")
masked = st.session_state.get("masked")
min_periods = st.session_state.get("min_periods", 5)

st.title("Upload Data")
container_download_example = st.empty()

########################################################################################
# Upload File section
file = st.file_uploader(
    "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
    type=["csv", "txt"],
    # needs callback to clear session state
)
keep_core_data = st.checkbox(
    "Keep only core data columns (timestamp, pioreactor_unit, od_reading)?",
    value=True,
    help="If checked, only the essential columns will be kept from the uploaded file.",
)
if file is None:
    if df_raw_od_data is not None:
        st.info("Some data was uploaded before. Processing will apply to that data.")
    else:
        with container_download_example:
            col0, col1 = st.columns(2)
            # populate columns (could be outside of with statement)
            col0.warning("no data uploaded.")
            # clicking triggers a re-run, but that is fast if no data was previously uploaded
            col1.download_button(
                label="Download example  pioreactor experiment in csv format.",
                data=pd.read_csv("data/example_batch_data_od_readings.csv").to_csv(
                    index=False
                ),
                file_name="example_batch_data_od_readings.csv",
                key="download_example_csv",
                mime="text/csv",
            )
        st.info("Upload a comma-separated (csv) file to get started.")

########################################################################################
# Filtering Form
with st.form("Upload_data_form", clear_on_submit=False):

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
    round_time = st.slider(
        "Round time to nearest second (defining timesteps)", 0, 15, 5, step=1
    )
    # Options for handeling negative OD readings
    st.write("Data filtering options:")
    filter_columns = st.columns(3)
    remove_negative = filter_columns[0].checkbox(
        "Remove negative OD readings",
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
    filter_by_iqr_range = filter_columns[2].checkbox(
        "Remove outliers by IQR",
        value=False,
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
    st.divider()
    st.write(
        "Select time window for data to be processed. Dates are inferred from "
        "uploaded data. This won't be plotted in red as filtered data, but just "
        "cap the datapoints for reactors outside of the selected time window."
        "The overall time window bounds the selected time windows for the individual "
        "reactors."
    )
    min_date, max_date = None, None
    if df_raw_od_data is not None:
        min_date, max_date = st.select_slider(
            "Select overall time window (inferred).",
            options=df_raw_od_data["timestamp_rounded"],
            value=(
                df_raw_od_data["timestamp_rounded"].min(),
                df_raw_od_data["timestamp_rounded"].max(),
            ),
        )
    st.divider()
    if df_wide_raw_od_data is not None:
        with st.expander("Select time window per reactor"):
            st.info("Note: Minimum and maximum for slider are reactor specific!")
            # per reactor, get min and max timestamps
            time_ranges = dict()
            for reactor in df_wide_raw_od_data.columns:
                time_ranges[reactor] = st.select_slider(
                    f"Select time window (inferred) for {reactor}."
                    " Bounded by overall time window.",
                    options=df_wide_raw_od_data[reactor].dropna().index,
                    value=(
                        df_wide_raw_od_data[reactor].dropna().index.min(),
                        df_wide_raw_od_data[reactor].dropna().index.max(),
                    ),
                )
    st.divider()
    st.write("Plotting options:")
    use_same_yaxis_scale = st.checkbox(
        "Use same y-axis for all reactors?",
        value=False,
        key="yaxis_scale",
        help="Select plotting behaviour.",
    )
    st.divider()
    button_pressed = st.form_submit_button(
        "Apply options to uploaded data", type="primary"
    )

########################################################################################
# Raw data and plots

extra_warn = st.empty()

st.header("Raw OD data")
container_raw_data = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id

if button_pressed and file is None and df_raw_od_data is None:
    extra_warn.warning("No data uploaded.")
    st.stop()

msg = ""

# this runs wheather the button is pressed or not, but only if a file is uploaded?
if file is not None:
    df_raw_od_data = piogrowth.load.read_csv(file)

    # ! add check that required columns are in data and have correct dtypes (pandera)
    msg = (
        f"- Loaded {df_raw_od_data.shape[0]:,d} rows "
        f"and {df_raw_od_data.shape[1]:,d} columns.\n"
    )
    # round timestamp data
    # ! 'timestamp_localtime' must be in data (note down requirement)
    df_raw_od_data.insert(
        0,
        "timestamp_rounded",
        df_raw_od_data["timestamp_localtime"].dt.round(
            f"{round_time}s",
        ),
    )
    st.session_state["round_time"] = round_time
    rerun = st.session_state.get("df_raw_od_data") is None
    # only keep core data?
    if keep_core_data:
        try:
            df_raw_od_data = df_raw_od_data[
                [
                    "timestamp_rounded",
                    "timestamp_localtime",
                    "pioreactor_unit",
                    "od_reading",
                ]
            ]
            msg += "- Kept only core data columns.\n"
        except KeyError as e:
            st.error(
                "Could not keep only core data columns. "
                "Please check that the uploaded file contains "
                "the required columns: timestamp_localtime, pioreactor_unit, od_reading."
            )
            st.stop()
    st.session_state["df_raw_od_data"] = df_raw_od_data
    # re-run now with data set

    msg += f"- Wide OD data with rounded timestamps to {round_time} seconds.\n"
    # wide data of raw data
    # - can be used in plot for visualization,
    # - and in curve fitting (where gaps would be interpolated)
    N_before = df_raw_od_data.shape[0]
    df_raw_od_data = df_raw_od_data.dropna(
        subset=["timestamp_rounded", "pioreactor_unit", "od_reading"]
    )
    N_after = df_raw_od_data.shape[0]
    N_dropped = N_before - N_after
    if N_dropped > 0:
        msg += (
            f"- Dropped {N_dropped:,d} rows with missing values in core columns "
            "(timestamp_rounded, pioreactor_unit, od_reading).\n"
        )
    try:
        df_wide_raw_od_data = df_raw_od_data.pivot(
            index="timestamp_rounded",
            columns="pioreactor_unit",
            values="od_reading",
        )
    except ValueError as e:
        st.error(
            "Rounding produced duplicated timepoints in reactors,"
            f" please decrease below: {round_time} seconds."
        )
        with st.expander("Show error details"):
            st.write(e)
            st.write(df_raw_od_data)
        st.stop()
    st.session_state["df_wide_raw_od_data"] = df_wide_raw_od_data
    if rerun:
        # ? replace with callback function that creates the input form?
        st.rerun()


if button_pressed:
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
    # skip first or last measurements based on user input (after first loading the data)
    # ! won't be plotted in red as filtered data, but just not appear in the plots
    # ! applied to wide raw data
    if min_date:
        df_wide_raw_od_data = df_wide_raw_od_data.loc[min_date:max_date]
        st.info(f"Time range: {min_date} to {max_date}")

    for reactor, time_range in time_ranges.items():
        if reactor not in df_wide_raw_od_data.columns:
            continue
        _min_date, _max_date = time_range
        _min_date = max(_min_date, min_date)
        _max_date = min(_max_date, max_date)
        df_wide_raw_od_data[reactor] = df_wide_raw_od_data.loc[
            _min_date:_max_date, reactor
        ]

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    # Handle negative values
    if remove_negative:
        mask_negative = df_wide_raw_od_data_filtered < 0
        msg += (
            f"- Setting {mask_negative.sum().sum():,d} negative OD readings to NaN.\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_negative)
        masked = masked | mask_negative

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

    if filter_by_iqr_range:
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

    # from pathlib import Path
    # fpath = Path(f"playground/data/{custom_id}_masked_values.csv")
    # fpath.parent.mkdir(exist_ok=True, parents=True)
    # masked.to_csv(fpath)
    # df_wide_raw_od_data.to_csv(Path(f"playground/data/{custom_id}_raw_wide_data.csv"))
    st.session_state["df_wide_raw_od_data_filtered"] = df_wide_raw_od_data_filtered
    st.session_state["masked"] = masked

    df_rolling = df_wide_raw_od_data_filtered.rolling(
        rolling_window,
        min_periods=min_periods,
        center=True,
    ).median()
    st.session_state["df_rolling"] = df_rolling


with container_raw_data:
    st.dataframe(df_raw_od_data, width="content")

if df_wide_raw_od_data is not None and masked is not None:
    # Download options
    if not use_same_yaxis_scale:
        st.warning("Using different y-axis scale for each reactor.")
    fig = plot_growth_data_w_mask(
        df_wide_raw_od_data, masked, sharey=use_same_yaxis_scale
    )
    st.write(fig)

if msg:
    st.subheader("Processing summary of OD readings")
    st.markdown(msg)

if st.session_state.get("df_raw_od_data") is not None:
    download_data_button_in_sidebar(
        "df_raw_od_data",
        "Download raw data  \n(long format)",
        file_name="data_long_rounded_timestamps.csv",
    )

if st.session_state.get("df_wide_raw_od_data") is not None:
    download_data_button_in_sidebar(
        "df_wide_raw_od_data",
        "Download raw data  \n(wide format)",
        file_name="data_wide_rounded_timestamps.csv",
    )
if st.session_state.get("df_wide_raw_od_data_filtered") is not None:
    download_data_button_in_sidebar(
        "df_wide_raw_od_data_filtered",
        "Download filtered data",
        file_name="filtered_data_wide_rounded_timestamps.csv",
    )


if df_rolling is not None:
    st.header(f"Rolling median in window of {rolling_window}s using filtered OD data")
    st.write(df_rolling)
    ax = df_rolling.plot.line(style=".", ms=2)
    st.write(ax.get_figure())
    download_data_button_in_sidebar(
        "df_rolling",
        "Download rolling median data",
        file_name="rolling_median_on_filtered_wide_data_with_rounded_timestamps.csv",
    )

st.markdown("### Store in QurvE format")
st.info("This feature is not yet implemented.")
# convert = st.button("Store in QurvE format", key="store_in_QurvE")

# if convert:
#     st.warning("fct to convert to QurvE not implemented.")
# store in state
