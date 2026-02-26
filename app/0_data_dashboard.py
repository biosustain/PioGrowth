import pandas as pd
import streamlit as st
from buttons import download_data_button_in_sidebar
from plots import plot_growth_data_w_mask
from ui_components import page_header_with_help, show_warning_to_upload_data

import piogrowth

DATA_DASHBOARD_HELP = """
Review processed upload outputs in one place:

1. Raw and filtered data tables
2. Filtered-data plot with removed-point overlay
3. Rolling-median table and line plot
"""

page_header_with_help("Data Dashboard", DATA_DASHBOARD_HELP)

df_raw_od_data = st.session_state.get("df_raw_od_data")
df_wide_raw_od_data = st.session_state.get("df_wide_raw_od_data")
df_rolling = st.session_state.get("df_rolling")
df_time_map = st.session_state.get("df_time_map")
masked = st.session_state.get("masked")
start_time = st.session_state.get("start_time")
processing_summary = st.session_state.get("upload_processing_summary_msg")
rolling_window = st.session_state.get("rolling_window")
st.session_state.setdefault("yaxis_scale", False)
st.session_state.setdefault("elapsed_time_option", True)
use_same_yaxis_scale = bool(st.session_state.get("yaxis_scale", False))
use_elapsed_time = bool(st.session_state.get("elapsed_time_option", True))

if df_raw_od_data is None and df_rolling is None:
    show_warning_to_upload_data()
    st.stop()

with st.container(border=True):
    st.header("Summary Tables")
    raw_col, time_col = st.columns(2, gap="large")
    with raw_col:
        st.subheader("Raw OD data")
        if df_raw_od_data is None:
            st.info("Raw OD data preview appears after data is loaded.")
        else:
            st.dataframe(df_raw_od_data, width="stretch")
    with time_col:
        st.subheader("Timestamp to elapsed-time map")
        if df_time_map is None:
            st.info("Timestamp map is generated after preprocessing.")
        else:
            st.dataframe(df_time_map, width="stretch")

if df_wide_raw_od_data is not None and masked is not None:
    with st.container(border=True):
        st.header("Filtered Data Plot")
        plot_option_cols = st.columns(2, gap="large")
        with plot_option_cols[0]:
            use_same_yaxis_scale = st.checkbox(
                "Use same y-axis for all reactors?",
                key="yaxis_scale",
                help="Select plotting behaviour.",
            )
        with plot_option_cols[1]:
            use_elapsed_time = st.checkbox(
                "Use elapsed time (since start) as x-axis on plots?",
                key="elapsed_time_option",
                help="If checked, elapsed time will be used as x-axis in plots.",
            )
        st.session_state["USE_ELAPSED_TIME_FOR_PLOTS"] = bool(use_elapsed_time)
        if not use_same_yaxis_scale:
            st.warning("Using different y-axis scale for each reactor.")
        df_plot = df_wide_raw_od_data
        mask_plot = masked
        if use_elapsed_time:
            df_plot = piogrowth.reindex_w_relative_time(
                df=df_plot,
                start_time=start_time,
            )
            mask_plot = piogrowth.reindex_w_relative_time(
                df=mask_plot,
                start_time=start_time,
            )
        fig = plot_growth_data_w_mask(
            df_plot,
            mask_plot,
            sharey=use_same_yaxis_scale,
            is_data_index=not use_elapsed_time,
        )
        st.write(fig)

if processing_summary:
    with st.container(border=True):
        st.subheader("Processing summary of OD readings")
        st.markdown(processing_summary)

if df_rolling is not None:
    with st.container(border=True):
        st.header("Rolling Median")
        if rolling_window is not None:
            st.subheader(
                f"Rolling median in window of {rolling_window}s using filtered OD data"
            )
        else:
            st.subheader("Rolling median using filtered OD data")
        st.write(df_rolling)

        if not use_elapsed_time and start_time is not None:
            view = df_rolling.copy()
            view.index = start_time + pd.to_timedelta(view.index, unit="h")
        else:
            view = df_rolling

        ax = view.plot.line(style=".", ms=2)
        st.write(ax.get_figure())

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
    download_data_button_in_sidebar(
        "df_rolling",
        "Download rolling median data",
        file_name="rolling_median_on_filtered_wide_data_with_rounded_timestamps.csv",
    )
