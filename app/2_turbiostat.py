import functools

import pandas as pd
import streamlit as st
from plots import plot_derivatives, plot_fitted_data, plot_growth_data_w_peaks
from ui_components import show_warning_to_upload_data

from piogrowth.durations import find_max_range
from piogrowth.fit import fit_growth_data_w_peaks
from piogrowth.turbistat import detect_peaks

## Logic and PLOTTING


## UI
st.title("Growth Analysis of turbidostat mode")

no_data_uploaded = st.session_state.get("df_rolling") is None

if no_data_uploaded:
    show_warning_to_upload_data()

st.markdown(
    "Analyse pioreactor OD600 measurements when running in turbidostat mode. "
    "In turbistat mode, the growth is diluted to enable continuous growth state "
    "of microorganisms in the reactors."
)


with st.form(key="turbidostat_form"):
    turbiostat_meta = st.file_uploader(
        (
            "Upload metadata of dilution events. Optional, but recommended. "
            "If provided the peaks will be assigned based on the dilution events."
        ),
        type=["csv"],
    )

    minimum_distance = st.number_input(
        label="Minimum distance between peaks (in number of samples)",
        min_value=3,
        value=300,
        step=1,
        key="turbiostat_distance",
    )
    smoothing_factor = st.slider(
        label="Smoothing factor for spline fitting",
        min_value=1.0,
        value=1000.0,
        step=1.0,
        key="smoothing_factor",
    )
    high_percentage_treshold = st.slider(
        "Define percentage of µmax considered as high", 0, 100, 90, step=1
    )
    submitted = st.form_submit_button("Analyse")

if submitted:
    round_time = st.session_state.get("round_time", 60)
    df_rolling = st.session_state.get("df_rolling")

    df_meta = None
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
        # ! check that format is as expected
        st.write(df_meta)

    # Peak detection: Based on metadata or using scipy.signal.find_peaks
    if df_meta is not None:
        st.subheader("Reading peaks from provided metadata")
        st.write("Data is rounded to match OD data timepoints.")
        peaks = df_meta.pivot(
            index="timestamp_localtime",
            columns="pioreactor_unit",
            values="message",
        )
        st.dataframe(peaks, use_container_width=True)
    else:
        st.subheader("Detected peaks")
        st.write(
            "Note: Peaks are detected using "
            "[`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)"
        )

        _detect_peaks = functools.partial(detect_peaks, distance=minimum_distance)
        peaks = df_rolling.apply(_detect_peaks)
        st.dataframe(peaks)

    fig, axes = plot_growth_data_w_peaks(df_rolling, peaks)
    st.pyplot(fig)

    splines, df_first_derivative, d_maxima = fit_growth_data_w_peaks(
        df_rolling, peaks, smoothing_factor=smoothing_factor
    )

    prop_high = high_percentage_treshold / 100
    cutoffs = df_first_derivative.max() * prop_high
    in_high_growth = df_first_derivative.ge(cutoffs, axis=1)
    max_time_range = in_high_growth.apply(find_max_range, axis=0).T.convert_dtypes()

    fig, axes = plot_fitted_data(
        splines,
    )
    axes = axes.flatten()
    for ax, s_maxima in zip(axes, d_maxima.values()):
        for x in s_maxima.index:
            ax.axvline(x=x, color="red", linestyle="--")
    for ax, col in zip(axes, df_first_derivative.columns):
        row = max_time_range.loc[col]
        if row.is_continues:
            # only plot span if the time range is continous (no jumps)
            ax.axvspan(row.start, row.end, color="gray", alpha=0.2)
    st.subheader("Fitted splines per segment")
    st.pyplot(fig)

    st.subheader("First Derivative of fitted splines per segment")

    fig, axes = plot_derivatives(df_first_derivative)
    st.pyplot(fig)
