import functools

import pandas as pd
import streamlit as st
from buttons import create_download_button, download_data_button_in_sidebar
from names import summary_mapping
from plots import (
    create_figure_bytes_to_download,
    plot_derivatives,
    plot_fitted_data,
    plot_growth_data_w_peaks,
    reindex_w_relative_time,
)
from ui_components import show_warning_to_upload_data

from piogrowth.durations import find_max_range
from piogrowth.fit import fit_growth_data_w_peaks
from piogrowth.turbistat import detect_peaks


## Logic and PLOTTING
def create_summary(maxima: dict[str, pd.Series]) -> pd.DataFrame:
    """Create a summary DataFrame from the maxima dictionary."""
    df_summary = pd.DataFrame(maxima).stack()
    df_summary.index.names = ["timestamp", "pioreactor_unit"]
    df_summary.name = "OD_value"
    df_summary = df_summary.to_frame()
    return df_summary


def get_values_from_df(df_wide: pd.DataFrame, indices: pd.MultiIndex) -> pd.DataFrame:
    """Get values from the wide DataFrame based on the index of the summary DataFrame."""
    return df_wide.loc[indices.get_level_values("timestamp")].stack().loc[indices]


def reset_metadata():
    st.session_state["df_meta"] = None


########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")
df_meta = st.session_state.get("df_meta")
round_time = st.session_state.get("round_time", 60)
########################################################################################
# UI

st.title("Growth Analysis of turbidostat mode")

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

st.markdown(
    "Analyse pioreactor OD600 measurements when running in turbidostat mode. "
    "In turbidostat mode, the growth is diluted to enable continuous growth state "
    "of microorganisms in the reactors."
)
st.info(
    "Data is plotted using measured timepoints (in seconds), and the modeling is done "
    "using elapsed seconds since the initial timepoint."
)
### Form
with st.form(key="turbidostat_form"):
    turbiostat_meta = st.file_uploader(
        (
            "Upload metadata of dilution events. Optional, but recommended. "
            "If provided the peaks will be assigned based on the dilution events."
        ),
        type=["csv"],
    )
    # ! pick out names of columns in form
    meta_data_options = st.columns(3)
    if df_meta is None:
        col_timestamp = meta_data_options[0].selectbox(
            "Select timestamp column",
            options=["timestamp", "timestamp_localtime"],
            index=1,
        )
        col_reactors = meta_data_options[1].text_input(
            "Select column with reactor information",
            value="pioreactor_unit",
        )
        col_message = meta_data_options[2].text_input(
            "Select column with event description",
            value="message",
        )
    else:
        col_timestamp = meta_data_options[0].selectbox(
            "Select timestamp column",
            options=df_meta.columns.tolist(),
            index=(
                df_meta.columns.get_loc(st.session_state.turbidostat_timestamp_col)
                if st.session_state.get("turbidostat_timestamp_col") in df_meta.columns
                else 0
            ),
        )
        col_reactors = meta_data_options[1].selectbox(
            "Select column with reactor information",
            options=df_meta.columns.tolist(),
            index=(
                df_meta.columns.get_loc(st.session_state.turbidostat_reactor_col)
                if st.session_state.get("turbidostat_reactor_col") in df_meta.columns
                else 0
            ),
        )
        col_message = meta_data_options[2].selectbox(
            "Select column with event description",
            options=df_meta.columns.tolist(),
            index=(
                df_meta.columns.get_loc(st.session_state.turbidostat_message_col)
                if st.session_state.get("turbidostat_message_col") in df_meta.columns
                else 0
            ),
        )
    st.divider()
    with st.expander(
        "Peak detection settings if no dilution event data is available"
        " (or should not be used)",
        expanded=False,
    ):
        minimum_peak_height = st.number_input(
            label=(
                "Minimum peak height (in OD units) - used only if no metadata provided. "
                "No values uses adaptive thresholding based on the maximum of a OD curve."
                "The default is one-fifth of the maximum OD value in a time series."
            ),
            min_value=0.0,
            value=None,
        )
        minimum_distance = st.number_input(
            label="Minimum distance between peaks (in number of measurement timepoints)",
            min_value=3,
            value=300,
            step=1,
            key="turbiostat_distance",
        )
    st.divider()
    remove_downward_trending = st.checkbox(
        label="Remove downward trending data points (negative OD changes) globally",
        value=True,
        key="remove_downward_trending",
    )
    smoothing_factor = st.slider(
        label="Smoothing factor for spline fitting",
        min_value=1.0,
        value=1000.0,
        step=1.0,
        key="smoothing_factor",
    )
    high_percentage_threshold = st.slider(
        "Define percentage of µmax considered as high",
        min_value=0,
        max_value=100,
        value=90,
        step=1,
        key="high_percentage_threshold",
    )
    submitted = st.form_submit_button("Analyse")

with st.sidebar:
    st.button("Reset uploaded metadata", on_click=reset_metadata)

### Error messages
if st.session_state.get("show_error"):
    st.error(
        "Could not find column in metadata. Please check the column names."
        " The selection was adjusted to the available columns."
    )

container_metadata = st.empty()
if df_meta is not None:
    with container_metadata:
        st.write(df_meta)

########################################################################################
### On Submission of form parameters
if not submitted:
    st.stop()

st.session_state["show_error"] = False

if turbiostat_meta is None and df_meta is not None:
    st.warning(
        "Using previously uploaded metadata of dilution events."
        " Reset app to use automatic peak picking."
    )

if turbiostat_meta is not None:
    st.subheader("Uploaded metadata of dilution events (optional)")
    df_meta = pd.read_csv(
        turbiostat_meta, parse_dates=["timestamp_localtime"]
    ).convert_dtypes()
    df_meta.insert(
        0,
        "timestamp_rounded",
        df_meta["timestamp_localtime"].dt.round(
            f"{round_time}s",
        ),
    )
    mask_dilution_events = df_meta["event_name"] == "DilutionEvent"
    if not mask_dilution_events.all():
        st.info('Showing only rows with "DilutionEvent" in column "event_name".')
        df_meta = df_meta.loc[mask_dilution_events]
    st.session_state["df_meta"] = df_meta
    # ! check that format is as expected
    with container_metadata:
        st.write(df_meta)

# Peak detection: Based on metadata or using scipy.signal.find_peaks
if df_meta is not None:
    st.subheader("Reading peaks from provided metadata")
    st.write("Data is rounded to match OD data timepoints.")
    # if this fails user needs to pick out names of columns in form
    if not (len(set((col_timestamp, col_reactors, col_message))) == 3):
        st.error(
            "Selected columns from uploaded dilution metadata cannot overlap."
            " Use for each a unique column."
        )
        st.stop()
    try:
        peaks = df_meta.pivot(
            index="timestamp_rounded",
            columns=col_reactors,
            values=col_message,
        )
        st.session_state["turbidostat_timestamp_col"] = col_timestamp
        st.session_state["turbidostat_reactor_col"] = col_reactors
        st.session_state["turbidostat_message_col"] = col_message
    except KeyError:
        st.session_state["show_error"] = True
        st.rerun()

    st.dataframe(peaks, width="stretch")
else:
    st.subheader("Detected peaks")
    st.write(
        "Note: Peaks are detected using "
        "[`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)"
    )
    if minimum_peak_height is not None:
        st.write(
            "Minimum distance between peaks: "
            f"{minimum_peak_height} number of measured timepoints"
        )
    _detect_peaks = functools.partial(
        detect_peaks,
        distance=minimum_distance,
        prominence=minimum_peak_height,
    )
    peaks = df_rolling.apply(_detect_peaks)
    st.dataframe(peaks)

if remove_downward_trending:
    # Remove downward trending data globally on averaged data
    df_rolling = df_rolling.mask(df_rolling.diff().le(0))
    st.info(
        "Downward trending data points (negative OD changes) were removed globally."
    )

# views for plotting to allow for elapsed time option
df_rolling_view = df_rolling
peaks_view = peaks.copy()  # ! pointer only works if passed to a function
xlabel = "timepoints (rounded)"
if use_elapsed_time:
    # reindex all data to elapsed time for plotting
    xlabel = "elapsed time (in hours)"
    df_rolling_view = reindex_w_relative_time(df_rolling)
    # alternative for reindex_w_relative_time
    peaks_view.index = (
        pd.to_datetime(peaks.index) - st.session_state["start_time"]
    ).total_seconds() / 3_600
    peaks_view.index.name = "elapsed time in hours"
fig, axes = plot_growth_data_w_peaks(
    df_rolling_view, peaks_view, is_data_index=not use_elapsed_time
)
st.pyplot(fig)

with st.sidebar:
    create_download_button(
        label="Download figure for growth data with peaks as PDF",
        data=create_figure_bytes_to_download(fig, fmt="pdf"),
        file_name="growth_data_with_peaks.pdf",
        disabled=False,
        mime="application/pdf",
    )

# ? should the one with negative values removed stored globally?
st.session_state["df_rolling_turbidostat"] = df_rolling
download_data_button_in_sidebar(
    "df_rolling_turbidostat",
    label="Download data used for growth analysis",
    file_name="df_rolling_turbidostat.csv",
)
splines, df_first_derivative, d_maxima = fit_growth_data_w_peaks(
    df_rolling, peaks, smoothing_factor=smoothing_factor
)
st.session_state["df_splines_turbidostat"] = splines
st.session_state["df_derivatives_turbidostat"] = df_first_derivative
download_data_button_in_sidebar(
    "df_splines_turbidostat",
    label="Download data of fitted splines",
    file_name="df_splines_turbidostat.csv",
)
download_data_button_in_sidebar(
    "df_derivatives_turbidostat",
    label="Download data of derivatives",
    file_name="df_derivatives_turbidostat.csv",
)

prop_high = high_percentage_threshold / 100
cutoffs = df_first_derivative.max() * prop_high
in_high_growth = df_first_derivative.ge(cutoffs, axis=1)
# ToDo: do this for each segment between peaks
max_time_range = in_high_growth.apply(find_max_range, axis=0).T.convert_dtypes()
# st.write(max_time_range)

splines_view = splines
d_maxima_view = pd.DataFrame(d_maxima)
if use_elapsed_time:
    # reindex all data to elapsed time for plotting
    splines_view = reindex_w_relative_time(splines)
    d_maxima_view = reindex_w_relative_time(d_maxima_view)

fig, axes = plot_fitted_data(
    splines_view,
    xlabel=xlabel,
)
axes = axes.flatten()
for ax, _col in zip(axes, d_maxima_view.columns):
    s_maxima = d_maxima_view[_col].dropna()
    for x in s_maxima.index:
        ax.axvline(x=x, color="red", linestyle="--")
for ax, col in zip(axes, df_first_derivative.columns):
    row = max_time_range.loc[col]
    _start = row.start
    _end = row.end
    if use_elapsed_time:
        _start = (row.start - st.session_state["start_time"]).total_seconds() / 3600
        _end = (row.end - st.session_state["start_time"]).total_seconds() / 3600
    if row.is_continues:
        # only plot span if the time range is continuous (no jumps)
        ax.axvspan(_start, _end, color="gray", alpha=0.2)
st.subheader("Fitted splines per segment")
st.pyplot(fig)

with st.sidebar:
    create_download_button(
        label="Download figure for fitted splines as PDF",
        data=create_figure_bytes_to_download(fig, fmt="pdf"),
        file_name="fitted_splines.pdf",
        disabled=False,
        mime="application/pdf",
    )

st.subheader("First Derivative of fitted splines per segment")

df_first_derivative_view = df_first_derivative
if use_elapsed_time:
    # reindex all data to elapsed time for plotting
    df_first_derivative_view = reindex_w_relative_time(df_first_derivative)

fig, axes = plot_derivatives(df_first_derivative_view)
st.pyplot(fig)

with st.sidebar:
    create_download_button(
        label="Download figure for fitted derivatives as PDF",
        data=create_figure_bytes_to_download(fig, fmt="pdf"),
        file_name="fitted_derivatives.pdf",
        disabled=False,
        mime="application/pdf",
    )

# Summary table
st.subheader("Summary of high growth periods")

# Sidebar Download buttons
df_summary = create_summary(d_maxima)
# ! to fix for automatic peak picking
try:
    df_summary["OD_median"] = get_values_from_df(df_rolling, df_summary.index)
    df_summary["OD_spline"] = get_values_from_df(splines, df_summary.index)
    df_summary["OD_derivative"] = get_values_from_df(
        df_first_derivative, df_summary.index
    )
except ValueError as e:
    st.error(f"Error occurred while creating summary - ValueError: {e}")
df_summary = df_summary.swaplevel(0, 1).sort_index().rename(columns=summary_mapping)
st.dataframe(df_summary)
st.session_state["df_summary"] = df_summary
download_data_button_in_sidebar(
    "df_summary",
    label="Download summary of high growth periods",
    file_name="summary_turbidostat_periods.csv",
)
