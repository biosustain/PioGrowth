import streamlit as st


def convert_data(df):
    return df.to_csv(index=True).encode("utf-8")


def download_data_button_in_sidebar(session_key: str, label: str = "Download data"):
    """Create a download button associated with a key in session state
    in the sidebar.

    - nested keys not possible
    - session state must be a DataFrame (which we do not check yet)
    """
    if st.session_state.get(session_key) is not None:
        disabled = False
        data = convert_data(st.session_state[session_key])
    else:
        disabled = True
        data = ""
    st.sidebar.download_button(
        label,
        data=data,
        file_name="filtered_data.csv",
        mime="text/csv",
        disabled=disabled,
    )
