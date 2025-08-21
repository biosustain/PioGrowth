import pandas as pd
import streamlit as st

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]

st.title("Upload Data")

with st.form("Upload_data_form", clear_on_submit=False):

    file = st.file_uploader(
        "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
        type=["csv"],
        # needs callback to clear session state
    )
    custom_id = st.text_input(
        "Enter custom ID for data",
        max_chars=30,
        value=custom_id,
    )
    col1, col2 = st.columns([1,3], vertical_alignment="center")
    filter_option = col1.radio("Select if selected reactors are to be kept or removed", ("Remove", "Keep"))
    if st.session_state.get("df_raw_od_data") is None:
        reactors_to_filter = col2.text_input("Enter reactors to filter (comma separated)", "")
    round_time = st.slider("Round time to nearest minute", 0, 15, 5, step=1)
    st.form_submit_button("Submit", type="primary")

container_raw_data = st.empty()
container_figures = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id


if file is not None:
    df_raw_od_data = pd.read_csv(file).convert_dtypes()
    df_raw_od_data = df_raw_od_data.assign(
        timestamp=pd.to_datetime(df_raw_od_data["timestamp"])
    )
    st.session_state["df_raw_od_data"] = df_raw_od_data
    col2.empty()  # Clear previous input
    reactors_selected = col2.multiselect("Select reactors to filter", options=df_raw_od_data["pioreactor_unit"].unique())
    if reactors_selected:
        mask = df_raw_od_data["pioreactor_unit"].isin(reactors_selected)
        if filter_option == "Remove":
            df_raw_od_data = df_raw_od_data.loc[~mask]
        else:
            df_raw_od_data = df_raw_od_data.loc[mask]

with container_raw_data:
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
