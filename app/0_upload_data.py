import pandas as pd
import streamlit as st

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]

with st.form("Upload_data_form", clear_on_submit=False):

    file = st.file_uploader(
        "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
        type=["csv"],
    )
    custom_id = st.text_input(
        "Enter custom ID for data",
        max_chars=30,
        value=custom_id,
    )
    st.form_submit_button("Submit", type="primary")

raw_data = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id


if file is not None:
    df_raw_od_data = pd.read_csv(file).convert_dtypes()
    df_raw_od_data = df_raw_od_data.assign(
        timestamp=pd.to_datetime(df_raw_od_data["timestamp"])
    )
    st.session_state["df_raw_od_data"] = df_raw_od_data

with raw_data:
    st.dataframe(df_raw_od_data, use_container_width=True)

if df_raw_od_data is not None:
    # Download options
    st.download_button(
        "Download Raw Data",
        data=df_raw_od_data.to_csv(),
        file_name=f"{custom_id}_quervE_data.csv",
        mime="text/csv",
    )

st.markdown("### Store in QuervE format")

convert = st.button("Store in QuervE format", key="store_in_querve")

if convert:
    st.warning("fct to convert to QuervE.")
    # store in state
