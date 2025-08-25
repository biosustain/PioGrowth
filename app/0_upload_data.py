import pandas as pd
import streamlit as st
from plots import plot_growth_data

custom_id = st.session_state["custom_id"]
df_raw_od_data = st.session_state["df_raw_od_data"]

st.title("Upload Data")
container_download_example = st.empty()

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
    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    filter_option = col1.radio(
        "Select if selected reactors are to be kept or removed", ("Remove", "Keep")
    )
    if st.session_state.get("df_raw_od_data") is None:
        reactors_to_filter = col2.text_input(
            "Enter reactors to filter (comma separated)", ""
        )
    # Options for handeling negative OD readings
    remove_zero = st.checkbox("Remove zero OD readings", value=False)
    round_time = st.slider("Round time to nearest minute", 0, 15, 5, step=1)
    st.form_submit_button("Submit", type="primary")

st.header("Raw OD data")
container_raw_data = st.empty()
container_figures = st.empty()

if custom_id:
    st.session_state["custom_id"] = custom_id


if file is not None:
    df_raw_od_data = pd.read_csv(file).convert_dtypes()
    df_raw_od_data = df_raw_od_data.assign(
        timestamp=pd.to_datetime(df_raw_od_data["timestamp"])
    )
    msg = f" - Loaded {df_raw_od_data.shape[0]:,d} rows and {df_raw_od_data.shape[1]:,d} columns.\n"
    df_raw_od_data["timestamp_rounded"] = df_raw_od_data["timestamp"].dt.round(
        f"{round_time}s"
    )
    st.session_state["df_raw_od_data"] = df_raw_od_data
    col2.empty()  # Clear previous input
    reactors_selected = col2.multiselect(
        "Select reactors to filter", options=df_raw_od_data["pioreactor_unit"].unique()
    )
    if reactors_selected:
        mask = df_raw_od_data["pioreactor_unit"].isin(reactors_selected)
        if filter_option == "Remove":
            df_raw_od_data = df_raw_od_data.loc[~mask]
        else:
            df_raw_od_data = df_raw_od_data.loc[mask]

    # Handle negative values
    if remove_zero:
        mask_negative = df_raw_od_data["od_reading"] < 0
        msg += f" - Setting {mask_negative.sum():,d} negative OD readings to zero.\n"
        df_raw_od_data.loc[mask_negative, "od_reading"] = 0

    st.write(msg)

    # wide data of raw data, maybe removing outliers?
    # can be used in plot for visualization,
    # and in curve fitting (where gaps would be interpolated)
    df_wide_raw_od_data = df_raw_od_data.pivot(
        index="timestamp_rounded", columns="pioreactor_unit", values="od_reading"
    )
    st.header(f"Wide OD data with rounded timestamps to {round_time} seconds")
    st.write(df_wide_raw_od_data)
    st.write(df_wide_raw_od_data.describe())

    # outlier detection using IQR on rolling window: sets for center value of window a
    # true or false (this would be arguing maybe for long data format)
    # can be used in plot for visualization
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    st.header("Rolling median in window of 240s of OD data")
    df_rolling = df_wide_raw_od_data.rolling(
        "240s", min_periods=5, center=True
    ).median()
    st.write(df_rolling)
    # skip first N seconds or measurments
    # set rolling median to x seconds


else:
    with container_download_example:
        columns = st.columns([1, 2])
        columns[0].warning("no data uploaded.")
        columns[1].download_button(
            label="Download example  pioreactor experiment in csv format.",
            data=pd.read_csv("data/example_batch_data_od_readings.csv").to_csv(
                index=False
            ),
            file_name="example_batch_data_od_readings.csv",
            key="download_example_csv",
            mime="text/csv",
        )


with container_raw_data:
    st.dataframe(df_raw_od_data, use_container_width=True)

if df_raw_od_data is not None:
    # Download options
    fig = plot_growth_data(df_raw_od_data)
    with container_figures:
        st.write(fig)


st.markdown("### Store in QuervE format")

convert = st.button("Store in QuervE format", key="store_in_querve")

if convert:
    st.warning("fct to convert to QuervE.")
    # store in state
