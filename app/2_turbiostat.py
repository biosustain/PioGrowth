import pandas as pd
import streamlit as st
from ui_components import show_warning_to_upload_data

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
        "Upload metadata of dilution events.", type=["csv"]
    )
    st.text_input(
        label=(
            "Reactor Group - Members of a group are comma separated (`,`) and groups "
            "are semicolon (`;`) separated: e.g. `P01,P02,P03;P04,P05,P06` "
            "is two groups with each three unique reactors associated."
        ),
        value=st.session_state.get("turbiostat_text_input", ""),
    )
    submitted = st.form_submit_button("Analyse")

if submitted:
    df_meta = pd.read_csv(turbiostat_meta)
    st.write(df_meta)
