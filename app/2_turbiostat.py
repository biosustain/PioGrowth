import streamlit as st

st.title("Growth Analysis of turbidostat mode")

st.markdown("Analyse pioreactor OD600 measurements when running in turbidostat mode.")

with st.form(key="turbidostat_form"):
    st.radio(label="Filtering Strategy for x values", options=["Remove", "Keep"])
    st.text_input(
        label=(
            "Reactor Group - Members of a group are comma separated (`,`) and groups "
            "are semicolon (`;`) separated: e.g. `P01,P02,P03;P04,P05,P06` "
            "is two groups with each three unique reactors associated."
        ),
        value=st.session_state.get("turbiostat_text_input", ""),
    )
    st.form_submit_button("Analyse")
