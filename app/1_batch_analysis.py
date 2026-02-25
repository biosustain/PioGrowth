import numpy as np
import pandas as pd
import streamlit as st
from buttons import download_data_button_in_sidebar
from growthcurves_options import render_options_for_growthcurve_fitting

# from names import summary_mapping
from plots import plot_growth_data
from ui_components import (
    page_header_with_help,
    render_markdown,
    show_warning_to_upload_data,
)

# from piogrowth.durations import find_max_range
from piogrowth.fit_growthcurves import (  # datetimeindex_to_elapsed_hours,
    run_model_fitting_on_df,
)
from piogrowth.fit_spline import (  # fit_spline_and_derivatives_one_batch,
    get_smoothing_range,
)


def get_timestamps_from_elapsed_hours(
    elapsed_hours, start_time, elapsed_time_unit="h", round_to="s"
):
    return start_time + pd.to_timedelta(elapsed_hours, unit=elapsed_time_unit).dt.round(
        round_to
    )


########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")

DEFAULT_XLABEL_TPS = st.session_state.get("DEFAULT_XLABEL_TPS", "Timepoints (rounded)")
DEFAULT_XLABEL_REL = st.session_state.get("DEFAULT_XLABEL_REL", "Elapsed time (hours)")
########################################################################################
# page

BATCH_HELP = """
Run growth-model analysis on the uploaded rolling-median OD data.

Workflow:
1. Configure model and plotting options
2. Run analysis and review linear/log plots
3. Review the summary table and download results from the sidebar
"""

page_header_with_help("Batch Growth Analysis", BATCH_HELP)

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

smoothing_range = get_smoothing_range(len(df_rolling))

### Form ###############################################################################
with st.container(border=True):
    st.header("Step 1. Configure and Run Analysis")
    with st.form("Batch_processing_options", enter_to_submit=True):
        # Model selection
        (
            selected_model,
            spline_smoothing_value,
            n_fits_sliding_window,
            n_window_size,
            phase_boundary_method,
            exp_frac,
        ) = render_options_for_growthcurve_fitting(
            s_min=smoothing_range.s_min, s_max=smoothing_range.s_max
        )
        # if method == "Threshold method":
        #     high_percentage_threshold = st.slider(
        #         "Define percentage of µmax considered as high", 0, 100, 90, step=1
        #     )
        # User inputs for analysis
        st.write("#### Plotting options:")
        remove_raw_data = st.checkbox("Remove underlying data from plots", value=True)
        add_tangent_of_mu_max = st.checkbox(
            "Add tangent of µmax to growth plots of fitted splines", value=False
        )
        form_submit = st.form_submit_button("Run Analysis", type="primary")

        with st.expander("Data used for analysis (rolling median data):"):
            st.dataframe(st.session_state["df_rolling"], width="content")

### Render after from submission    ####################################################
if form_submit and not no_data_uploaded:
    Y_LABEL = "OD readings"
    exp_frac = exp_frac / 100
    if phase_boundary_method == "default":
        method = None
    stats_df = run_model_fitting_on_df(
        df_rolling,
        model_name=selected_model,
        n_fits=n_fits_sliding_window,
        spline_s=spline_smoothing_value,
        window_points=n_window_size,
        phase_boundary_phase_boundary_method=phase_boundary_method,
        exp_frac=exp_frac,
        lag_frac=exp_frac,
    )

    msg = """
    In plots the maximum change in OD (fitted) is indicated by the red dashed lines.
    The maximum change in OD (fitted) and it's timepoint is mentioned in the title of
    each plot. The selected range within the **gray shaded area** indicates the time
    period where the growth rate was in the exponential phase. The definiton depends
    on the method choosen:

    - [ ] fetch and plot example here
    """
    with st.container(border=True):
        st.header("Step 2. Review Results")
        st.markdown(msg)
        with st.expander("Show data used for model fitting"):
            st.dataframe(df_rolling, width="content")

    xlabel = DEFAULT_XLABEL_REL + f" since start at {start_time}"

    # ? allow users to plot data using TimeStamps?
    # Use starttime for timestamp calculations using elapsed time

    mu_max = stats_df["mu_max"]
    time_at_mu_max = stats_df["time_at_umax"]
    range_exp_phase = list(zip(stats_df["exp_phase_start"], stats_df["exp_phase_end"]))

    titles = [
        f"{col} - $\\mu$ max {mu:<.5f} at {idx:<.3f} hours"
        for col, mu, idx in zip(df_rolling.columns, mu_max, time_at_mu_max)
    ]

    fig, axes = plot_growth_data(
        df_rolling, titles=titles, ylabel=Y_LABEL, xlabel=xlabel
    )
    axes = axes.flatten()
    for ax, x in zip(axes, time_at_mu_max):
        ax.axvline(x=x, color="red", linestyle="--")
    for ax, (_start, _end) in zip(axes, range_exp_phase):
        ax.axvspan(_start, _end, color="gray", alpha=0.2)
    if add_tangent_of_mu_max:
        pass
        # ! decide how to plot this on the log or non-log data
        # for ax, col in zip(axes, derivatives.columns):
        #     b = maxima.loc[col]
        #     x_center = maxima_idx.loc[col]
        #     y_center = splines.loc[x_center, col]
        #     x = (derivatives.index - x_center).total_seconds().to_numpy()
        #     y = b * x + y_center
        #     mask = (y < splines_view[col].max()) & (y > splines_view[col].min())
        #     # only plot tangent if the time range is continuous (no jumps)
        #     ax.plot(derivatives_view.index[mask], y[mask], color="blue", linestyle="--")
        # del x, y, b, x_center, y_center, mask
    with st.container(border=True):
        st.subheader("Growth Curves (Linear Scale)")
        st.write(fig)

    ## Plot on log scale
    fig_log, axes = plot_growth_data(
        np.log((df_rolling + 0.01)),
        titles=titles,
        ylabel="ln(OD readings)",
        xlabel=xlabel,
    )
    # ! duplicated code, could be refactored
    axes = axes.flatten()
    for ax, x in zip(axes, time_at_mu_max):
        ax.axvline(x=x, color="red", linestyle="--")
    for ax, (_start, _end) in zip(axes, range_exp_phase):
        ax.axvspan(_start, _end, color="gray", alpha=0.2)

    with st.container(border=True):
        st.subheader("Growth Curves (Log Scale)")
        st.write(fig_log)

    ### Summary Table ##################################################################
    with st.container(border=True):
        st.subheader("Summary of batch analysis")
        st.write(
            f"The start time was {start_time}. Timepoints are relative to this start time."
        )
        st.dataframe(stats_df, width="content")
    st.session_state["batch_analysis_summary_df"] = stats_df
    download_data_button_in_sidebar(
        "batch_analysis_summary_df",
        label="Download summary",
        file_name="batch_analysis_summary_df.csv",
    )

# info on used methods
with st.container(border=True):
    st.header("Step 3. Method Notes")
    render_markdown("app/markdowns/curve_fitting.md")
