import streamlit as st

# General configurations
st.set_page_config(page_title="PioGrowth", layout="wide")

# Initialize constants
DEFAULT_CUSTOM_ID = "pioreactor_experiment"
if not st.session_state.get("custom_id"):
    st.session_state["custom_id"] = DEFAULT_CUSTOM_ID
if st.session_state.get("df_raw_od_data") is None:
    st.session_state["df_raw_od_data"] = None


# function creating the about page from a markdown file
def render_about():
    with open("app/markdowns/about.md", "r") as f:
        about_content = f.read()
    st.write(about_content)


# Navigation
raw_data = st.Page("0_upload_data.py", title="Upload Data")
batch_analysis = st.Page("1_batch_analysis.py", title="Analyse batch growth experiment")
turbistat_modus = st.Page(
    "2_turbiostat.py", title="Analyse batch growth experiment in turbidostat mode"
)
about_page = st.Page(render_about, title="About")

# Sidebar
st.sidebar.header("Info:")
st.sidebar.write("version: 0.0.1")
st.sidebar.button(
    "Download selected raw data",
    on_click=lambda: print("Download functionality not implemented yet."),
)
st.sidebar.button(
    "Download growth rate plots",
    on_click=lambda: print("Download functionality not implemented yet."),
)
st.sidebar.button(
    "Download summarized data",
    on_click=lambda: print("Download functionality not implemented yet."),
)
st.sidebar.button("Reset session", on_click=st.session_state.clear)

# build multi-page app
pg = st.navigation([raw_data, batch_analysis, turbistat_modus, about_page])
pg.run()
