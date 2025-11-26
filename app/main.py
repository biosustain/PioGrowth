import streamlit as st
from ui_components import render_markdown

import piogrowth

# General configurations
st.set_page_config(page_title="PioGrowth", layout="wide")

# Initialize constants
DEFAULT_CUSTOM_ID = "pioreactor_experiment"
if st.session_state.get("custom_id") is None:
    st.session_state["custom_id"] = DEFAULT_CUSTOM_ID
if st.session_state.get("df_raw_od_data") is None:
    st.session_state["df_raw_od_data"] = None

st.session_state["DEFAULT_XLABEL_TPS"] = "Timepoints (rounded)"
st.session_state["DEFAULT_XLABEL_REL"] = "Elapsed time (hours)"


# function creating the about page from a markdown file
def render_about():
    render_markdown("app/markdowns/about.md")


# Navigation
raw_data = st.Page("0_upload_data.py", title="Upload Data")
batch_analysis = st.Page("1_batch_analysis.py", title="Analyse growth experiment")
turbistat_modus = st.Page(
    "2_turbiostat.py", title="Analyse growth experiment in turbidostat mode"
)
about_page = st.Page(render_about, title="About")

# Sidebar
st.sidebar.info("Info: To reset the app, reload the page.")
# st.sidebar.button("Reset session", on_click=st.session_state.clear)
st.sidebar.write(f"version: {piogrowth.__version__}")
st.sidebar.write("Buttons activate if associated data is available:")
# st.sidebar.write(st.session_state)

# build multi-page app
pg = st.navigation([raw_data, batch_analysis, turbistat_modus, about_page])
pg.run()
