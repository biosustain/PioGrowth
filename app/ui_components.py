import streamlit as st


def is_data_available(key):
    """Check that pioreactor data was uploaded."""
    ret = st.session_state.get(key) is not None
    return ret


def show_warning_to_upload_data():
    """Show a warning message to upload data."""
    with st.container():
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            st.warning("No data available for analysis. Please upload first:")
        with col2:
            st.page_link(
                "0_upload_data.py",
                icon=":material/upload:",
                label="Upload Data",
                help="Go to upload data page.",
            )


def render_markdown(fpath: str):
    """Open and write markdown content from file."""
    with open(fpath, "r") as f:
        about_content = f.read()
    st.write(about_content)
