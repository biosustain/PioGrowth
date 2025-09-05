import streamlit as st
from plots import plot_derivatives, plot_fitted_data
from ui_components import render_markdown, show_warning_to_upload_data

from piogrowth.fit import fit_spline_and_derivatives_no_nan, get_smoothing_range

########################################################################################
# page

st.header("Batch Growth Analysis")

no_data_uploaded = st.session_state.get("df_rolling") is None

if no_data_uploaded:
    show_warning_to_upload_data()

st.title("Data used for analysis (rolling median data):")
view_data_module = st.empty()
with view_data_module:
    st.write("No data available for analysis. Please upload first.")

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
    remove_raw_data = st.checkbox("Remove underlying data from plots", value=False)
    add_tangent_of_mu_max = st.checkbox(
        "Add tangent of µmax to growth plots", value=False
    )
    form_submit = st.form_submit_button("Run Analysis", type="primary")

if not no_data_uploaded:
    with view_data_module:
        st.dataframe(st.session_state["df_rolling"], use_container_width=True)

# Process button
if form_submit and not no_data_uploaded:
    st.session_state.process_batch = True
    df_rolling = st.session_state["df_rolling"].interpolate()
    smoothing_range = get_smoothing_range(len(df_rolling))
    st.write("Smoothing parameter range as suggested by scipy:")
    st.dataframe(smoothing_range)

    splines, derivatives = fit_spline_and_derivatives_no_nan(
        df_rolling,
        smoothing_factor=smoothing_range.iloc[0],
    )

    titles = [
        f"{col} - max $\\mu$ {mu:<.5f} at {idx}"
        for col, mu, idx in zip(
            df_rolling.columns, derivatives.max(), derivatives.idxmax()
        )
    ]

    st.title("Fitted splines")
    st.dataframe(splines, use_container_width=True)
    fig, axes = plot_fitted_data(splines, titles=titles)
    st.write(fig)

    st.title("First order derivatives")
    st.dataframe(derivatives, use_container_width=True)
    fig, axes = plot_derivatives(derivatives=derivatives, titles=titles)
    st.write(fig)

# info on used methods
render_markdown("app/markdowns/curve_fitting.md")
