import numpy as np
import pandas as pd
import streamlit as st
from buttons import download_data_button_in_sidebar
from names import summary_mapping
from plots import plot_derivatives, plot_fitted_data, reindex_w_relative_time
from ui_components import render_markdown, show_warning_to_upload_data

from piogrowth.durations import find_max_range
from piogrowth.fit import fit_spline_and_derivatives_one_batch, get_smoothing_range

########################################################################################
# state

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_time_map = st.session_state.get("df_time_map")
no_data_uploaded = st.session_state.get("df_rolling") is None
df_rolling = st.session_state.get("df_rolling")


########################################################################################
# page

st.header("Batch Growth Analysis")

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

smoothing_range = get_smoothing_range(len(df_rolling))

view_data_module = st.empty()
with view_data_module:
    st.write("No data available for analysis. Please upload first.")


with st.form("Batch_processing_options", enter_to_submit=True):
    apply_log = st.checkbox(
        "Apply shift to minimum value from above zero and log transformation to data before fitting splines: $\\ln(y - \\max(\\min(\\text{OD}_{\\text{reactor}}), 0) + 0.001)$",
        value=False,
    )
    spline_smoothing_value = st.slider(
        "Smoothing of the spline fitted to OD values (zero means no smoothing). "
        "Range suggested using scipy, see "
        "[docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html)",
        1,
        smoothing_range.s_max,
        smoothing_range.s_min,
        step=1,
    )
    high_percentage_treshold = st.slider(
        "Define percentage of µmax considered as high", 0, 100, 90, step=1
    )
    # User inputs for analysis
    st.write("#### Plotting options:")
    remove_raw_data = st.checkbox("Remove underlying data from plots", value=True)
    add_tangent_of_mu_max = st.checkbox(
        "Add tangent of µmax to growth plots of fitted splines", value=False
    )
    form_submit = st.form_submit_button("Run Analysis", type="primary")

if not no_data_uploaded:
    with view_data_module:
        with st.expander("Data used for analysis (rolling median data):"):
            st.dataframe(st.session_state["df_rolling"], width="content")

# Process button
if form_submit and not no_data_uploaded:
    Y_LABEL = "OD readings"
    if apply_log:
        Y_LABEL = "ln(OD readings)"

        def log_transform(s):
            s_min = s.min()
            if s_min < 0:
                s = s - s_min

            return np.log(s + 0.001)

        df_rolling = df_rolling.apply(log_transform)
    splines, derivatives = fit_spline_and_derivatives_one_batch(
        df_rolling,
        smoothing_factor=spline_smoothing_value,
    )
    prop_high = high_percentage_treshold / 100
    cutoffs = derivatives.max() * prop_high
    in_high_growth = derivatives.ge(cutoffs, axis=1)
    max_time_range = in_high_growth.apply(find_max_range, axis=0).T.convert_dtypes()
    st.session_state["splines"] = splines
    st.session_state["derivatives"] = derivatives

    download_data_button_in_sidebar(
        "derivatives",
        label="Download derivatives",
        file_name="derivatives.csv",
    )
    download_data_button_in_sidebar(
        "splines",
        label="Download fitted splines",
        file_name="splines.csv",
    )

    maxima = derivatives.max()
    maxima_idx = derivatives.idxmax()

    titles = [
        f"{col} - max $\\mu$ {mu:<.5f} at {idx}"
        for col, mu, idx in zip(df_rolling.columns, maxima, maxima_idx)
    ]

    msg = f"""
    In plots the maximum change in OD (fitted) is indicated by the red dashed lines.
    The maximum change in OD (fitted) and it's timepoint is mentioned in the title of
    each plot. The selected range within the **gray shaded area** indicates the time
    period where the growth rate was above {prop_high:.0%} of the maximum growth rate - if
    this range was continuous and had no spikes.
    """
    st.markdown(msg)
    st.title("Fitted splines")
    with st.expander("Show fitted splines data:"):
        st.dataframe(splines, width="content")
    # views for plotting to allow for elapsed time option
    splines_view = splines
    derivatives_view = derivatives
    df_rolling_view = df_rolling
    maxima_idx_view = maxima_idx
    xlabel = "timepoints (rounded)"
    if use_elapsed_time:
        # reindex all data to elapsed time for plotting
        splines_view = reindex_w_relative_time(splines)
        derivatives_view = reindex_w_relative_time(derivatives)
        xlabel = "elapsed time (in hours)"
        df_rolling_view = reindex_w_relative_time(df_rolling)
        maxima_idx_view = derivatives_view.idxmax()
    fig, axes = plot_fitted_data(
        splines_view, titles=titles, ylabel=Y_LABEL, xlabel=xlabel
    )
    axes = axes.flatten()
    if not remove_raw_data:
        for col, ax in zip(df_rolling_view.columns, axes):
            df_rolling_view[col].plot(
                ax=ax, c="black", style=".", alpha=0.3, ms=1, label="Raw data"
            )
    for ax, x in zip(axes, maxima_idx_view):
        ax.axvline(x=x, color="red", linestyle="--")
    for ax, col in zip(axes, derivatives.columns):
        row = max_time_range.loc[col]
        if row.is_continues:
            # only plot span if the time range is continuous (no jumps)
            _start = row.start
            _end = row.end
            if use_elapsed_time:
                _start = (row.start - df_rolling.index[0]).total_seconds() / 3600
                _end = (row.end - df_rolling.index[0]).total_seconds() / 3600
            ax.axvspan(_start, _end, color="gray", alpha=0.2)
    if add_tangent_of_mu_max:
        for ax, col in zip(axes, derivatives.columns):
            b = maxima.loc[col]
            x_center = maxima_idx.loc[col]
            y_center = splines.loc[x_center, col]
            x = (derivatives.index - x_center).total_seconds().to_numpy()
            y = b * x + y_center
            mask = (y < splines_view[col].max()) & (y > splines_view[col].min())
            # only plot tangent if the time range is continuous (no jumps)
            ax.plot(derivatives_view.index[mask], y[mask], color="blue", linestyle="--")
        del x, y, b, x_center, y_center, mask
    st.write(fig)

    st.title("First order derivatives")
    with st.expander("Show first derivative data:"):
        st.dataframe(derivatives, width="content")
    fig, axes = plot_derivatives(
        derivatives=derivatives_view, titles=titles, xlabel=xlabel
    )
    axes = axes.flatten()
    for ax, x in zip(axes, maxima_idx_view):
        ax.axvline(x=x, color="red", linestyle="--")
    for ax, col in zip(axes, derivatives.columns):
        row = max_time_range.loc[col]
        if row.is_continues:
            # only plot span if the time range is continuous (no jumps)
            _start = row.start
            _end = row.end
            if use_elapsed_time:
                _start = (row.start - df_rolling.index[0]).total_seconds() / 3600
                _end = (row.end - df_rolling.index[0]).total_seconds() / 3600
            ax.axvspan(_start, _end, color="gray", alpha=0.2)
    st.write(fig)

    batch_analysis_summary_df = pd.DataFrame(
        {
            "timpoint": maxima_idx,
            "max_growth_rate": maxima,
            "OD_median": [
                df_rolling.loc[idx, col]
                for idx, col in zip(maxima_idx, maxima_idx.index)
            ],
            "OD_in_filtered_data": [
                st.session_state["df_wide_raw_od_data_filtered"].loc[idx, col]
                for idx, col in zip(maxima_idx, maxima_idx.index)
            ],
            "OD_spline": [
                splines.loc[idx, col] for idx, col in zip(maxima_idx, maxima_idx.index)
            ],
        }
    )
    # rename maximum range columns and add to summary table
    max_time_range = max_time_range.rename(
        columns={
            "start": f"max_{prop_high:.0%}_growth_start",
            "end": f"max_{prop_high:.0%}_growth_end",
            "duration": f"max_{prop_high:.0%}_growth_duration",
            "is_continues": f"max_{prop_high:.0%}_growth_is_continues",
        }
    )
    batch_analysis_summary_df = pd.concat(
        [batch_analysis_summary_df, max_time_range], axis=1
    ).rename(columns=summary_mapping)
    st.subheader("Summary of batch analysis")
    st.dataframe(batch_analysis_summary_df, width="content")
    st.session_state["batch_analysis_summary_df"] = batch_analysis_summary_df
    download_data_button_in_sidebar(
        "batch_analysis_summary_df",
        label="Download summary",
        file_name="batch_analysis_summary_df.csv",
    )

# info on used methods
render_markdown("app/markdowns/curve_fitting.md")
