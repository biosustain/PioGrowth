import functools

import pandas as pd
import streamlit as st
from buttons import create_download_button, download_data_button_in_sidebar
from growthcurves_options import render_options_for_growthcurve_fitting
from plots import create_figure_bytes_to_download, plot_growth_data_w_peaks
from ui_components import page_header_with_help, show_warning_to_upload_data

from piogrowth.fit_growthcurves import run_model_fitting_on_df_with_peaks
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


def render_metadata_preview(df_meta_preview: pd.DataFrame):
    """Render uploaded dilution metadata."""
    with st.container(border=True):
        st.subheader("Uploaded metadata of dilution events (optional)")
        st.dataframe(df_meta_preview, width="stretch")


########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")
df_meta = st.session_state.get("df_meta")
round_time = st.session_state.get("round_time", 60)

DEFAULT_XLABEL_TPS = st.session_state.get("DEFAULT_XLABEL_TPS", "Timepoints (rounded)")
DEFAULT_XLABEL_REL = st.session_state.get("DEFAULT_XLABEL_REL", "Elapsed time (hours)")
########################################################################################
# UI

TURBIDOSTAT_HELP = """
Analyse OD600 measurements in turbidostat mode and identify high-growth periods.

Workflow:
1. Configure model and peak settings (optionally upload dilution metadata)
2. Run analysis and inspect peaks/fit visualizations
3. Review and download summary outputs
"""

page_header_with_help("Turbidostat Growth Analysis", TURBIDOSTAT_HELP)
if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

with st.container(border=True):
    st.markdown(
        "Analyse pioreactor OD600 measurements when running in turbidostat mode. "
        "In turbidostat mode, the growth is diluted to enable continuous growth state "
        "of microorganisms in the reactors."
    )
    st.info(
        "Data is plotted using measured timepoints (in seconds), and the modeling is done "
        "using elapsed seconds since the initial timepoint."
    )

### Form ###############################################################################
with st.container(border=True):
    st.header("Step 1. Configure and Run Analysis")
    with st.form(key="turbidostat_form"):
        # Model selection
        (
            selected_model,
            spline_smoothing_value,
            n_fits_sliding_window,
            n_window_size,
            phase_boundary_method,
            exp_frac,
        ) = render_options_for_growthcurve_fitting(s_min=3, s_max=1000)

        meta_col, req_col = st.columns([4, 1], vertical_alignment="center")
        with meta_col:
            st.markdown("#### Dilution Metadata (Optional)")
        with req_col:
            with st.popover("Requirements", width="stretch"):
                st.markdown("Expected CSV with event records.")
                st.markdown("Columns should include:")
                st.markdown("- timestamp column")
                st.markdown("- reactor identifier")
                st.markdown("- event/message column")
                st.markdown("Rows labeled `DilutionEvent` are used when available.")

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
        submitted = st.form_submit_button("Analyse", type="primary")

with st.sidebar:
    st.button("Reset uploaded metadata", on_click=reset_metadata)

### Error messages
if st.session_state.get("show_error"):
    with st.container(border=True):
        st.error(
            "Could not find column in metadata. Please check the column names."
            " The selection was adjusted to the available columns."
        )

if df_meta is not None:
    render_metadata_preview(df_meta)

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
    # st.subheader("Uploaded metadata of dilution events (optional)")
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
    df_meta["elapsed_time_in_seconds"] = (
        df_meta["timestamp_localtime"] - start_time
    ).dt.total_seconds()
    df_meta["elapsed_time_in_hours"] = df_meta["elapsed_time_in_seconds"] / 3600.0

    # ! check that format is as expected
    render_metadata_preview(df_meta)

# Peak detection: Based on metadata or using scipy.signal.find_peaks
if df_meta is not None:
    with st.container(border=True):
        st.subheader("Step 2. Detect Peaks from Uploaded Metadata")
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
                index="elapsed_time_in_hours",
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
    with st.container(border=True):
        st.subheader("Step 2. Detect Peaks Automatically")
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
    with st.container(border=True):
        st.dataframe(peaks, width="stretch")
    st.session_state["peaks"] = peaks
    download_data_button_in_sidebar(
        "peaks",
        label="Download peaks in format used for growth analysis",
        file_name="peaks.csv",
    )

if remove_downward_trending:
    # Remove downward trending data globally on averaged data
    df_rolling = df_rolling.mask(df_rolling.diff().le(0))
    st.info(
        "Downward trending data points (negative OD changes) were removed globally."
    )

if phase_boundary_method == "default":
    phase_boundary_method = None

# views for plotting to allow for elapsed time option
xlabel = DEFAULT_XLABEL_REL


# ? should the one with negative values removed stored globally?
st.session_state["df_rolling_turbidostat"] = df_rolling

download_data_button_in_sidebar(
    "df_rolling_turbidostat",
    label="Download data used for growth analysis",
    file_name="df_rolling_turbidostat.csv",
)

stats_df = run_model_fitting_on_df_with_peaks(
    df_rolling,
    peaks,
    model_name=selected_model,
    n_fits=n_fits_sliding_window,
    spline_s=spline_smoothing_value,
    window_points=n_window_size,
    phase_boundary_method=phase_boundary_method,
    exp_frac=exp_frac,
    lag_frac=exp_frac,
)

fig, axes = plot_growth_data_w_peaks(df_rolling, peaks, is_data_index=False)

time_at_mu_max = stats_df["time_at_umax"]

axes = axes.flatten()
for ax, _col in zip(axes, df_rolling.columns):
    s_maxima = time_at_mu_max.loc[_col]
    for x in s_maxima:
        ax.axvline(x=x, color="red", linestyle="--")
for ax, col in zip(axes, df_rolling.columns):
    sub_df = stats_df.loc[col]
    range_exp_phase = list(zip(sub_df["exp_phase_start"], sub_df["exp_phase_end"]))
    for _start, _end in range_exp_phase:
        ax.axvspan(_start, _end, color="gray", alpha=0.2)
with st.container(border=True):
    st.subheader("Step 3. Review Fitted Curves and Peaks")
    st.pyplot(fig)

with st.sidebar:
    create_download_button(
        label="Download figure for fitted splines as PDF",
        data=create_figure_bytes_to_download(fig, fmt="pdf"),
        file_name="data_with_peaks_and_mu_max.pdf",
        disabled=False,
        mime="application/pdf",
    )


# Summary table
### Summary Table ##################################################################
with st.container(border=True):
    st.subheader("Step 4. Summary of High Growth Periods")
    st.write(
        f"The start time was {start_time}. Timepoints are relative to this start time."
    )
    st.dataframe(stats_df, width="content")
st.session_state["batch_analysis_summary_df"] = stats_df
download_data_button_in_sidebar(
    "batch_analysis_summary_df",
    label="Download summary",
    file_name="batch_analysis_summary_df.csv",
)
