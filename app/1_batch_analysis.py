import streamlit as st
from plots import plot_growth_data
from ui_components import is_data_available, show_warning_to_upload_data

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
    st.session_state.process_batch = True
    df_raw_od_data = st.session_state["df_raw_od_data"]
    with raw_data:
        st.dataframe(df_raw_od_data, use_container_width=True)

    fig = plot_growth_data(df_raw_od_data)
    st.write(fig)
