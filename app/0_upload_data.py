import growthcurves as gc
import pandas as pd
import streamlit as st
from ui_components import page_header_with_help

import piogrowth

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_wide_raw_od_data_filtered = st.session_state.get("df_wide_raw_od_data_filtered")
min_periods = st.session_state.get("min_periods", 5)
# use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)

UPLOAD_HELP = """
This page loads and preprocesses a single PioReactor OD dataset.

Use this order:
1. Upload the OD data file
2. (Optional) Upload calibration/turbidostat metadata files
3. Configure and apply preprocessing options
4. Review tables and plots on the Data Dashboard page
5. Use the Downloads page for exports
"""

page_header_with_help("Upload Data", UPLOAD_HELP)


def callback_clear_raw_data():
    st.session_state["df_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data"] = None
    st.session_state["df_wide_raw_od_data_filtered"] = None
    st.session_state["masked"] = None
    st.session_state["upload_processing_summary_msg"] = None
    # reset time windows axis and data
    if "min_date" in st.session_state:
        del st.session_state["min_date"]
    if "max_date" in st.session_state:
        del st.session_state["max_date"]


########################################################################################
# Upload File section
with st.container(border=True):
    header_col, req_col = st.columns([4, 1], vertical_alignment="center")
    with header_col:
        st.header("Step 1. Upload PioReactor OD Data")
    with req_col:
        with st.popover("Requirements", width="stretch"):
            st.markdown("**Expected structure:**")
            st.markdown("- CSV/TXT file readable by `pandas.read_csv`")
            st.markdown(
                "- Required columns: `timestamp_localtime`, `pioreactor_unit`, `od_reading`"
            )
            st.markdown("- One row per measurement")
            st.divider()
            st.markdown("**Example file:**")
            example_data = pd.read_csv("data/example_batch_data_od_readings.csv")
            st.dataframe(example_data.head(10), hide_index=True, width="stretch")
            st.download_button(
                label="Download example CSV",
                data=example_data.to_csv(index=False),
                file_name="example_batch_data_od_readings.csv",
                key="download_example_csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )

    st.markdown("**Main OD Data**")
    file = st.file_uploader(
        "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
        type=["csv", "txt"],
        on_change=callback_clear_raw_data,
    )
    main_options_cols = st.columns([3, 2], gap="medium")
    with main_options_cols[0]:
        keep_core_data = st.checkbox(
            "Keep only core data columns (timestamp, pioreactor_unit, od_reading)?",
            value=True,
            help="If checked, only the essential columns are kept from the uploaded file.",
        )
    with main_options_cols[1]:
        custom_id = st.text_input(
            "Custom ID for data",
            max_chars=30,
            value=custom_id,
        )

    if file is None:
        if df_raw_od_data is None:
            st.warning("No data uploaded.")
            st.info("Upload a comma-separated (`.csv`) file to get started.")

with st.container(border=True):
    st.header("Step 2. Optional metadata uploads")
    optional_upload_cols = st.columns([2, 3], gap="small")
    with optional_upload_cols[0]:
        st.markdown("**OD Calibration Table**")
        od_adjustment_upload = st.file_uploader(
            "OD adjustment table",
            type=["csv"],
            key="upload_page_od_adjustment_table",
        )
    with optional_upload_cols[1]:
        st.markdown("**Turbidostat Metadata**")
        turbidostat_meta_upload = st.file_uploader(
            "Dilution metadata (for Turbidostat page)",
            type=["csv"],
            key="upload_page_turbidostat_meta",
        )
        st.session_state.setdefault("turbidostat_timestamp_col", "timestamp_localtime")
        st.session_state.setdefault("turbidostat_reactor_col", "pioreactor_unit")
        st.session_state.setdefault("turbidostat_message_col", "message")
        timestamp_options = ["timestamp", "timestamp_localtime"]
        if st.session_state.get("turbidostat_timestamp_col") not in timestamp_options:
            st.session_state["turbidostat_timestamp_col"] = "timestamp_localtime"
        turbi_cols = st.columns(3, gap="small")
        with turbi_cols[0]:
            st.selectbox(
                "Select timestamp column",
                options=timestamp_options,
                key="turbidostat_timestamp_col",
            )
        with turbi_cols[1]:
            st.text_input(
                "Select column with reactor information",
                key="turbidostat_reactor_col",
            )
        with turbi_cols[2]:
            st.text_input(
                "Select column with event description",
                key="turbidostat_message_col",
            )

    if od_adjustment_upload is not None:
        st.session_state["od_adjustment_upload_bytes"] = od_adjustment_upload.getvalue()
        st.session_state["od_adjustment_upload_name"] = od_adjustment_upload.name
    else:
        st.session_state["od_adjustment_upload_bytes"] = None
        st.session_state["od_adjustment_upload_name"] = None
    if turbidostat_meta_upload is not None:
        st.session_state["turbidostat_meta_upload_bytes"] = (
            turbidostat_meta_upload.getvalue()
        )
        st.session_state["turbidostat_meta_upload_name"] = turbidostat_meta_upload.name
    else:
        st.session_state["turbidostat_meta_upload_bytes"] = None
        st.session_state["turbidostat_meta_upload_name"] = None

### Form ##############################################################################
with st.container(border=True):
    st.header("Step 3. Configure Processing Options")

    with st.form("Upload_data_form", clear_on_submit=False):
        st.write("#### Data filtering options:")
        if st.session_state.get("df_raw_od_data") is None:
            available_reactors = []
            reactors_selected = st.multiselect(
                "Select reactors to include in analysis",
                options=available_reactors,
                default=available_reactors,
                help="Upload OD data to populate available reactors.",
            )
        else:
            available_reactors = sorted(
                df_raw_od_data["pioreactor_unit"].dropna().astype(str).unique().tolist()
            )
            reactors_selected = st.multiselect(
                "Select reactors to include in analysis",
                options=available_reactors,
                default=available_reactors,
                help=(
                    "All reactors are selected by default. Remove any reactors you do  "
                    "not want analyzed."
                ),
            )
        filter_columns = st.columns(2)
        with filter_columns[0]:
            negative_handling = st.radio(
                "How should negative OD readings be handled?",
                options=[
                    "Set negative OD readings to missing (NaN)",
                    "Impute negative values by moving average",
                ],
                index=(
                    1
                    # if st.session_state.get("remove_negative", False)
                    # and st.session_state.get("fill_na", False)
                    # else 0
                ),
                key="negative_handling",
                help=(
                    "Negative values distort curve fitting. Choose whether to convert "
                    "them to missing values or impute them."
                ),
            )
            remove_negative = True
            if negative_handling == "Impute negative values by moving average":
                remove_negative = False
            fill_na = st.checkbox(
                "Impute missing bioscatter readings using forward and backward filling",
                help=(
                    "If checked, missing values will be "
                    "imputed using forward fill and backward fill. This is recommended "
                    "if you expect only a few missing or negative values that are "
                    "likely due to measurement errors.  Note that this will include "
                    "negative zeros which were previously removed using the above "
                    "option."
                ),
                value=st.session_state.get("fill_na", False),
            )
            # ! move to after smoothing is applied?
            remove_downward_trending = st.checkbox(
                label="Remove downward trending data points (negative OD changes) "
                " globally after smoothing the data.",
                value=st.session_state.get("remove_downward_trending", False),
                help=(
                    "This can be used to remove data points that are smaller than a "
                    "previous one. Downward trends will be removed, but the upward "
                    "trend will be kept from a local minimum."
                ),
            )
            remove_max = st.checkbox(
                "Remove maximum OD readings by quantile",
                value=st.session_state.get("remove_max", False),
            )
            filter_by_iqr_range = st.checkbox(
                "Remove outliers by Inter-Quartile Range (IQR) in rolling window of"
                " timepoints",
                value=st.session_state.get("filter_by_iqr_range", False),
            )
        with filter_columns[1]:
            quantile_max = st.slider(
                "Max quantile for maximum removal",
                0.9,
                1.0,
                st.session_state.get("quantile_max", 0.99),
                step=0.01,
            )
            iqr_range_value = st.slider(
                "IQR range for outlier removal",
                1.0,
                3.0,
                st.session_state.get("iqr_range_value", 1.5),
                step=0.1,
            )
            rolling_window = st.slider(
                "Rolling window (of timepoints) for IQR outlier removal",
                11,
                61,
                st.session_state.get("rolling_window", 21),
                step=2,
            )

        st.divider()
        st.write(
            "#### Time options:\n"
            "Select time windows for data to be processed. Dates are inferred from "
            "uploaded data. This won't be plotted in red as filtered data, but just "
            "cap the datapoints for reactors outside of the selected windows."
            " The overall time window bounds the selected time windows for the individual"
            " reactors."
        )
        min_date, max_date = None, None
        time_window_cols = st.columns(
            [3, 4, 1], gap="large", vertical_alignment="bottom"
        )
        with time_window_cols[0]:
            round_time = st.slider(
                "Round time to nearest second (defining timesteps)."
                "Can be used to align timeseries "
                "with slight time offsets.",
                0,
                60,
                st.session_state.get("round_time", 5),
                step=1,
            )
        # ! move this to data_dashboard page.
        update_zero_timepoint = None
        time_ranges = {}
        if df_wide_raw_od_data is not None:
            with time_window_cols[1]:
                if df_raw_od_data is not None:
                    min_date, max_date = st.select_slider(
                        "Select overall time window (inferred).",
                        options=df_raw_od_data["timestamp_rounded"],
                        value=(
                            st.session_state.get(
                                "min_date", df_raw_od_data["timestamp_rounded"].min()
                            ),
                            st.session_state.get(
                                "max_date", df_raw_od_data["timestamp_rounded"].max()
                            ),
                        ),
                    )
                else:
                    st.empty()
            with time_window_cols[2]:
                update_zero_timepoint = st.checkbox(
                    "Reset T0",
                    value=st.session_state.get("update_zero_timepoint", False),
                    help=(
                        "If checked, a new zero time is set to the minimum timestamp of the"
                        " overall time window."
                    ),
                )
            with st.expander("Select time window per reactor"):
                st.info("Note: Minimum and maximum for slider are reactor specific!")
                # per reactor, get min and max timestamps
                for reactor in df_wide_raw_od_data.columns:
                    if st.session_state.get("time_ranges", {}).get(reactor) is not None:
                        _min_tp, _max_tp = st.session_state["time_ranges"][reactor]
                    else:
                        _options_timepoints = (
                            df_wide_raw_od_data[reactor].dropna().index
                        )
                        _min_tp, _max_tp = (
                            _options_timepoints.min(),
                            _options_timepoints.max(),
                        )
                    time_ranges[reactor] = st.select_slider(
                        f"Select time window (inferred) for {reactor}."
                        " Bounded by overall time window.",
                        options=df_wide_raw_od_data[reactor].dropna().index,
                        value=(
                            _min_tp,
                            _max_tp,
                        ),
                    )

        st.divider()
        button_pressed = st.form_submit_button(
            "Apply options to uploaded data", type="primary", width="stretch"
        )

### save form state
# remember form values for next time page is opened
st.session_state["keep_core_data"] = keep_core_data
st.session_state["custom_id"] = custom_id
st.session_state["reactors_selected"] = reactors_selected
st.session_state["remove_negative"] = remove_negative
st.session_state["fill_na"] = fill_na
st.session_state["remove_downward_trending"] = remove_downward_trending
st.session_state["remove_max"] = remove_max
st.session_state["filter_by_iqr_range"] = filter_by_iqr_range
st.session_state["quantile_max"] = quantile_max
st.session_state["iqr_range_value"] = iqr_range_value
st.session_state["rolling_window"] = rolling_window
st.session_state["round_time"] = round_time
st.session_state["update_zero_timepoint"] = update_zero_timepoint
st.session_state["time_ranges"] = time_ranges

if min_date is not None and max_date is not None:
    # update data specific options in session state
    st.session_state["min_date"] = min_date
    st.session_state["max_date"] = max_date
########################################################################################
# Process data

extra_warn = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id

if button_pressed and file is None and df_raw_od_data is None:
    extra_warn.warning("No data uploaded.")
    st.stop()

msg = ""

# File Uploaded ########################################################################
# this runs wheather the button is pressed or not, but only if a file is uploaded
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
    # use starttime to compute elapsed time
    start_time = df_raw_od_data["timestamp_rounded"].min()
    st.session_state["start_time"] = start_time
    df_raw_od_data["elapsed_time_in_seconds"] = (
        df_raw_od_data["timestamp_rounded"] - start_time
    ).dt.total_seconds()
    msg += f"- Added elapsed time in seconds since start ({start_time}).\n"
    st.session_state["round_time"] = round_time
    rerun = st.session_state.get("df_raw_od_data") is None
    # only keep core data?
    if keep_core_data:
        try:
            df_raw_od_data = df_raw_od_data[
                [
                    "timestamp_rounded",
                    "timestamp_localtime",
                    "elapsed_time_in_seconds",
                    "pioreactor_unit",
                    "od_reading",
                ]
            ]
            msg += "- Kept only core data columns.\n"
        except KeyError:
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
    st.session_state["upload_processing_summary_msg"] = msg
    if rerun:
        # ? replace with callback function that creates the input form?
        st.rerun()

### Apply option from form #############################################################
if button_pressed:
    # Keep only reactors selected for analysis
    if not reactors_selected:
        st.warning("No reactors selected. Select at least one reactor to continue.")
        st.stop()
    st.write(f"Reactors included in analysis: {reactors_selected}")
    df_raw_od_data = df_raw_od_data.loc[
        df_raw_od_data["pioreactor_unit"].astype(str).isin(reactors_selected)
    ]

    # skip first or last measurements based on user input (after first loading the data)
    # ! won't be plotted in red as filtered data, but just not appear in the plots
    # ! applied to wide raw data
    if min_date:
        df_wide_raw_od_data = df_wide_raw_od_data.loc[min_date:max_date]
        st.info(f"Time range: {min_date} to {max_date}")

    if update_zero_timepoint:
        start_time = min_date

    for reactor, time_range in time_ranges.items():
        if reactor not in df_wide_raw_od_data.columns:
            continue
        _min_date, _max_date = time_range
        _min_date = max(_min_date, min_date)
        _max_date = min(_max_date, max_date)
        df_wide_raw_od_data[reactor] = df_wide_raw_od_data.loc[
            _min_date:_max_date, reactor
        ]

        if update_zero_timepoint and start_time < _min_date:
            # update start time if new zero time is after current start time
            start_time = _min_date

    # initalize masked here
    masked = pd.DataFrame(
        False,
        index=df_wide_raw_od_data.index,
        columns=df_wide_raw_od_data.columns,
    )
    df_wide_raw_od_data_filtered = df_wide_raw_od_data.copy()

    #### Apply Data Filtering options ##################################################
    # Handle negative values
    n_negative = (df_wide_raw_od_data_filtered < 0).sum().sum()
    if n_negative > 0:
        st.warning(f"Found {n_negative:,d} negative OD readings.")
        msg += f"- Found {n_negative:,d} negative OD readings.\n"
    if remove_negative:
        mask_negative = df_wide_raw_od_data_filtered < 0
        msg += (
            f"- Setting {mask_negative.sum().sum():,d} negative OD readings to NaN.\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(mask_negative)
        masked = masked | mask_negative
    else:
        mask_negative = df_wide_raw_od_data_filtered < 0
        window = 31
        # Replace negatives with NaN,
        # then compute centered rolling mean over non-missing values
        temp = df_wide_raw_od_data_filtered.mask(mask_negative)
        rolling_mean = temp.rolling(window=window, min_periods=1, center=True).mean()
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            mask_negative, rolling_mean
        )
        n_imputed = mask_negative.sum().sum()
        msg += (
            f"- Imputed {n_imputed:,d} negative OD readings using"
            f" centered rolling mean (window={window}).\n"
        )
        msg += f"   - in detail: {mask_negative.sum().to_dict()}\n"
        masked = masked | mask_negative
        del temp, rolling_mean, mask_negative
    if fill_na:
        mask_na = df_wide_raw_od_data_filtered.isna()
        msg += f"- Filling {mask_na.sum().sum():,d} missing OD readings.\n"
        msg += f"   - in detail: {mask_na.sum().to_dict()}\n"
        # ! should I visualize the values differently?
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.fillna(
            method="ffill"
        ).fillna(method="bfill")

    # remove quantiles
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
        with st.spinner("Applying IQR outlier removal..."):
            mask_outliers = df_wide_raw_od_data_filtered.apply(
                gc.preprocessing.detect_outliers,
                raw=False,
                method="iqr",
                factor=iqr_range_value,
                window_size=rolling_window,
            ).astype(bool)
            # st.write(f"### Number of outliers detected: {mask_outliers.sum().sum()}")
            msg += f"- Number of outliers detected: {mask_outliers.sum().sum()}\n"
            msg += f"   - in detail: {mask_outliers.sum().to_dict()}\n"
            masked = masked | mask_outliers

            # apply mask to entire dataframe

            df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
                mask_outliers
            )

    masked = masked.convert_dtypes()

    st.session_state["df_wide_raw_od_data_filtered"] = df_wide_raw_od_data_filtered
    st.session_state["masked"] = masked

    df_rolling = (
        df_wide_raw_od_data_filtered.rolling(
            rolling_window,
            min_periods=min_periods,
            center=True,
        )
        .median()
        .sort_index()
    )
    # ! check if overwriting start time has consequences

    if remove_downward_trending:
        # Remove downward trending data globally on averaged data
        df_wide_raw_od_data_filtered = df_wide_raw_od_data_filtered.mask(
            df_wide_raw_od_data_filtered.diff().le(0)
        )
        msg += (
            "- Downward trending data points (negative OD changes) were "
            "removed globally."
        )
    #### switch wide data to time eplased in hours #####################################
    df_rolling = piogrowth.reindex_w_relative_time(
        df=df_rolling,
        start_time=st.session_state["start_time"],
    )
    st.session_state["df_rolling"] = df_rolling

    st.session_state["USE_ELAPSED_TIME_FOR_PLOTS"] = bool(
        st.session_state.get("elapsed_time_option", True)
    )
    st.session_state["rolling_window"] = int(rolling_window)

    df_time_map = (
        df_raw_od_data[["timestamp_rounded", "elapsed_time_in_seconds"]]
        .drop_duplicates()
        .set_index("timestamp_rounded")
    )
    df_time_map["elapsed_time_in_hours"] = (
        df_time_map["elapsed_time_in_seconds"] / 3600.0
    )
    st.session_state["df_time_map"] = df_time_map
    st.session_state["upload_processing_summary_msg"] = msg
    st.write("### Data processing summary:")
    st.write(msg)
