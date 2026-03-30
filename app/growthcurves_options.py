import growthcurves as gc
import streamlit as st


def render_options_for_growthcurve_fitting(s_min=3, s_max=1000, s_default=1000):
    st.write("### Model selection")
    selected_model = st.selectbox(
        "See the differences between the options in the"
        " [growthcurves package](https://growthcurves.readthedocs.io)",
        gc.get_all_models(),
        index=7,
    )
    st.session_state["selected_model"] = selected_model
    st.write("#### Spline fitting options:")
    spline_smoothing_value = st.slider(
        "Smoothing of the spline fitted to OD values (zero means no smoothing). "
        "Range suggested using scipy, see "
        "[docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html)",
        1,
        s_max,
        s_default,
        step=1,
    )
    st.write("#### Sliding window options:")
    n_fits_sliding_window = st.slider(
        "Number of fits used for sliding window to calculate derivatives",
        5,
        200,
        50,
        step=5,
    )
    n_window_size = st.slider(
        "Window size for sliding window method (in hours)",
        3,
        1000,
        300,
        step=3,
    )
    # ! Add tangent and threshold method options here
    tangent_cols = st.columns(2)
    phase_boundary_method = tangent_cols[0].radio(
        "Select method for exponential phase detection (default recommended):",
        ["default", "tangent", "threshold"],
        help=("""
            Default picks for parametric models the threshold method and
            for phenomenological models, the sliding window and spline method the 
            tangent method.

            In short:

            - "threshold": Threshold-based method using fractions of μ_max
            - "tangent": Tangent line method at point of maximum growth rate
            """),
        index=0,
    )
    exp_frac = tangent_cols[1].slider(
        "Define percentage of µmax considered as high for threshold method",
        0,
        100,
        90,
        step=1,
    )
    return (
        selected_model,
        spline_smoothing_value,
        n_fits_sliding_window,
        n_window_size,
        phase_boundary_method,
        exp_frac,
    )
