import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter


def load_data(data_source):
    # Placeholder function to load data based on the selected source
    if data_source == "Calibrated":
        # Load calibrated data
        return pd.DataFrame()  # Replace with actual data loading logic
    else:
        # Load raw data
        return pd.DataFrame()  # Replace with actual data loading logic


def perform_analysis(od_data_list, high_mu_percentage, spline_smoothing):
    # Placeholder function to perform batch analysis
    return od_data_list  # Replace with actual analysis logic


def plot_growth_data(df: pd.DataFrame):
    units = df["pioreactor_unit"].nunique()
    fig, axes = plt.subplots(units, figsize=(10, 2*units), sharey=True, sharex=True)
    # grid container (reactive to UI changes)
    for (label, group_df), ax in zip(df.groupby("pioreactor_unit"), axes):
        group_df.plot.scatter(
            x="timestamp",
            y="od_reading",
            rot=45,
            ax=ax,
            title=f"Reactor {label}",  # Customize legend text here
        )
    ax = axes[-1]
    fig = ax.get_figure()
    fig.tight_layout()

    date_form = DateFormatter("%Y-%m-%d %H:%M")
    _ = ax.xaxis.set_major_formatter(date_form)
    st.write(fig)

def plot_growth_rate_data(fitted_spline_data):
    # Placeholder function to plot growth rate data
    plt.figure()
    plt.plot(fitted_spline_data)  # Replace with actual plotting logic
    st.pyplot()


def summarize_growth_data(fitted_spline_data):
    # Placeholder function to summarize growth data
    return pd.DataFrame()  # Replace with actual summary logic


########################################################################################
# page

st.header("Batch Growth Analysis")

# Data source selection
# data_source = st.radio("Select data source", ("Calibrated", "Raw"))

file = st.file_uploader(
    "PioReactor OD table. Upload a single CSV file with PioReactor recordings.",
    type=["csv"],
)
raw_data = st.empty()

with st.form("Batch_processing_options", enter_to_submit=True):
    filter_option = st.radio("Select filter option", ("Option 1", "Option 2"))
    # User inputs for analysis
    high_mu_percentage = st.slider("Define high mu percentage", 0, 100, 50, step=1)
    spline_smoothing = st.slider("Define spline smoothing", 0.1, 10.0, 1.0, step=0.1)

    form_submit = st.form_submit_button("Submit", type="primary")

# Process button
if form_submit:
    st.session_state.process_batch = True
    df_raw_od_data = pd.read_csv(file).convert_dtypes()
    df_raw_od_data = df_raw_od_data.assign(
        timestamp=pd.to_datetime(df_raw_od_data["timestamp"])
    )
    with raw_data:
        st.dataframe(df_raw_od_data, use_container_width=True)

    # # Placeholder for plots
    # if "process_batch" in st.session_state and st.session_state.process_batch:
    #     od_data_list = load_data(
    #         data_source
    #     )  # Function to load data based on selection
    #     fitted_spline_data = perform_analysis(
    #         od_data_list, high_mu_percentage, spline_smoothing
    #     )  # Function to perform analysis

    #     # Plot growth data
    plot_growth_data(df_raw_od_data)

    #     # Plot growth rate data
    #     plot_growth_rate_data(fitted_spline_data)

    #     # Summary table
    #     summary_data = summarize_growth_data(fitted_spline_data)
    #     st.write("Summary Data", summary_data)

    # Download options
    st.download_button(
        "Download Raw Data",
        data=df_raw_od_data.to_csv(),
        file_name="raw_od_data.csv",
        mime="text/csv",
    )
