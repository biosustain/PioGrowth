import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.dates import DateFormatter
from ui_components import is_data_available, show_warning_to_upload_data


def plot_growth_data(df: pd.DataFrame):
    """Plot optical density (OD) growth data."""
    units = df["pioreactor_unit"].nunique()
    fig, axes = plt.subplots(units, figsize=(10, 2 * units), sharey=True, sharex=True)
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


########################################################################################
# page

st.header("Batch Growth Analysis")

no_data_uploaded = not is_data_available()

if no_data_uploaded:
    show_warning_to_upload_data()

raw_data = st.empty()

with st.form("Batch_processing_options", enter_to_submit=True):
    correction_strategy = st.radio(
        "Negative correction strategy", ("Median", "Qurve(OD + |min(OD)|)")
    )
    spline_smoothing_values = st.slider(
        "Smoothing of the spline fitted to OD values (zero means no smoothing)",
        0.0,
        1.0,
        1.0,
        step=0.05,
    )
    high_percentage_treshold = st.slider(
        "Define percentage of µmax considered as high", 0, 100, 90, step=1
    )
    # User inputs for analysis
    st.write("#### Plotting options:")
    remove_raw_data = st.checkbox("Remove raw data from plots", value=False)
    add_tangent_of_mu_max = st.checkbox(
        "Add tangent of µmax to growth plots", value=False
    )
    form_submit = st.form_submit_button("Submit", type="primary")

# Process button
if form_submit and not no_data_uploaded:
    st.write("Filter option:", filter_option)
    st.session_state.process_batch = True
    df_raw_od_data = pd.read_csv(file).convert_dtypes()
    df_raw_od_data = df_raw_od_data.assign(
        timestamp=pd.to_datetime(df_raw_od_data["timestamp"])
    )
    with raw_data:
        st.dataframe(df_raw_od_data, use_container_width=True)

    plot_growth_data(df_raw_od_data)

    # Download options
    st.download_button(
        "Download Raw Data",
        data=df_raw_od_data.to_csv(),
        file_name="raw_od_data.csv",
        mime="text/csv",
    )
