import streamlit as st


def is_data_available():
    """Check that pioreactor data was uploaded."""
    ret = st.session_state.get("df_raw_od_data") is not None
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
