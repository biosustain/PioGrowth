import functools

import pandas as pd
import streamlit as st
from plots import plot_growth_data_w_peaks
from ui_components import show_warning_to_upload_data

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
            "Upload metadata of dilution events."
            "Optional and only used for verification of peaks detected at the moment."
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
    submitted = st.form_submit_button("Analyse")

if submitted:
    if turbiostat_meta is not None:
        df_meta = pd.read_csv(turbiostat_meta)
        st.write(df_meta)

    df_rolling = st.session_state.get("df_rolling")

    _detect_peaks = functools.partial(detect_peaks, distance=minimum_distance)
    peaks = df_rolling.apply(_detect_peaks)

    st.write("## Detected Peaks")
    st.write(peaks)

    fig, axes = plot_growth_data_w_peaks(df_rolling, peaks)
    st.pyplot(fig)
