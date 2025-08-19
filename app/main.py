import streamlit as st

# General configurations 
st.set_page_config(page_title="PioGrowth", layout="wide")

# Initialize constants
DEFAULT_CUSTOM_ID = "pioreactor_experiment"
if not st.session_state.get('custom_id'):
    st.session_state['custom_id'] = DEFAULT_CUSTOM_ID
if st.session_state.get('df_raw_od_data') is None:
    st.session_state['df_raw_od_data'] = None

# function creating the about page from a markdown file
def render_about():
    with open("app/markdowns/about.md", "r") as f:
        about_content = f.read()
    st.write(about_content)

# Navigation
raw_data = st.Page('0_upload_data.py', title="Upload Data")
batch_analysis = st.Page('1_batch_analysis.py', title="Batch Analysis")
about_page = st.Page(render_about, title="About")

# Sidebar
st.sidebar.header("Download:")
st.sidebar.write("Plot and other things (buttons)")

# build multi-page app
pg = st.navigation([raw_data, batch_analysis, about_page])
pg.run()
