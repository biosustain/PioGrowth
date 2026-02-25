import inspect
import numpy as np
import pandas as pd
import streamlit as st
import growthcurves as gc
import growthcurves.plot as gc_plot
import plotly.graph_objects as go
import time
from buttons import download_data_button_in_sidebar
from growthcurves_options import (
    render_parameter_calculation_table_upload_style,
    render_upload_style_analysis_options,
)

# from names import summary_mapping
from ui_components import (
    page_header_with_help,
    show_warning_to_upload_data,
)

# from piogrowth.durations import find_max_range
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
NON_PARAMETRIC_FIT_PARAMS = set(
    inspect.signature(gc.non_parametric.fit_non_parametric).parameters
)
########################################################################################
# page

BATCH_HELP = """
Run growth-model analysis on the uploaded rolling-median OD data.

Workflow:
1. Configure analysis options and run analysis
2. Review linear/log plots and summary outputs
"""


def _load_method_notes_markdown() -> str:
    """Load method notes markdown shown in the Help popover."""
    try:
        with open("app/markdowns/curve_fitting.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "_Method notes file not found._"


BATCH_HELP = f"{BATCH_HELP}\n\n---\n\n{_load_method_notes_markdown()}"


def _cycle(items, current, step):
    """Cycle forward/backward through a list."""
    if not items:
        return current
    try:
        idx = items.index(current)
    except ValueError:
        idx = 0
    return items[(idx + step) % len(items)]


def _run_model_fitting_on_df_compat(
    df: pd.DataFrame,
    model_name: str,
    n_fits: int,
    spline_s: int,
    smooth_mode: str,
    window_points: int,
    phase_boundary_method: str,
    lag_cutoff: float,
    exp_cutoff: float,
) -> tuple[pd.DataFrame, dict]:
    """Run fitting across reactors using gc.fit_model in a version-compatible way."""
    stats_df = {}
    fit_cache = {}

    for col in df.columns:
        s = df[col].dropna()
        t_start = time.time()
        fit_kwargs = _build_fit_kwargs(
            model_name=model_name,
            n_fits=n_fits,
            window_points=window_points,
            spline_s=spline_s,
            smooth_mode=smooth_mode,
        )

        fit_result, stats = gc.fit_model(
            t=s.index.to_numpy(),
            N=s.to_numpy(),
            model_name=model_name,
            lag_threshold=lag_cutoff,
            exp_threshold=exp_cutoff,
            phase_boundary_method=phase_boundary_method,
            **fit_kwargs,
        )
        stats["elapsed_time"] = time.time() - t_start
        stats["model_name"] = model_name
        stats_df[col] = stats
        fit_cache[col] = fit_result

    return pd.DataFrame(stats_df).T, fit_cache


def _build_fit_kwargs(
    model_name: str,
    n_fits: int,
    window_points: int,
    spline_s: int,
    smooth_mode: str,
) -> dict:
    """Map UI options to growthcurves fit kwargs across API versions."""
    fit_kwargs = {}
    if model_name == "sliding_window":
        fit_kwargs["n_fits"] = n_fits
        fit_kwargs["window_points"] = window_points
    elif model_name == "spline":
        fit_kwargs["window_points"] = window_points
        if "smooth" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["smooth"] = smooth_mode
        if "spline_s" in NON_PARAMETRIC_FIT_PARAMS:
            fit_kwargs["spline_s"] = spline_s
    return fit_kwargs


def _plot_instantaneous_mu_trace(
    t_values: pd.Index | pd.Series | list | tuple,
    n_values: pd.Index | pd.Series | list | tuple,
    fit_result: dict | None,
    stats: dict,
    title: str,
) -> go.Figure:
    """Plot model-based instantaneous specific growth rate μ(t)."""
    t = pd.Series(t_values, dtype="float64").to_numpy()
    n = pd.Series(n_values, dtype="float64").to_numpy()
    mask = np.isfinite(t) & np.isfinite(n) & (n > 0)
    t, n = t[mask], n[mask]

    fig = go.Figure()
    if len(t) < 3:
        fig.update_layout(title=title, height=400, template="plotly_white")
        return fig

    fit_result = fit_result or {}
    params = fit_result.get("params", {})
    model_type = fit_result.get("model_type", "")

    fit_t_min = params.get("fit_t_min", stats.get("fit_t_min"))
    fit_t_max = params.get("fit_t_max", stats.get("fit_t_max"))

    if pd.notna(fit_t_min) and pd.notna(fit_t_max):
        fit_mask = (t >= float(fit_t_min)) & (t <= float(fit_t_max))
        t_model = t[fit_mask]
        n_model = n[fit_mask]
    else:
        t_model = t
        n_model = n

    if len(t_model) < 3:
        t_model = t
        n_model = n

    mu_model = None
    try:
        if model_type == "sliding_window":
            window_points = int(params.get("window_points", 15))
            _, mu_model = gc.inference.compute_sliding_window_growth_rate(
                t_model, n_model, window_points=window_points
            )
        elif model_type in gc.get_all_parametric_models():
            n_fit = gc.models.evaluate_parametric_model(t_model, model_type, params)
            _, mu_model = gc.inference.compute_instantaneous_mu(t_model, n_fit)
        elif model_type == "spline":
            spline = gc.models.spline_from_params(params)
            n_fit = np.exp(spline(t_model))
            _, mu_model = gc.inference.compute_instantaneous_mu(t_model, n_fit)
    except Exception:
        mu_model = None

    if mu_model is not None:
        mu_model = np.asarray(mu_model, dtype=float)
        valid = np.isfinite(mu_model)
        if valid.any():
            fig.add_trace(
                go.Scatter(
                    x=t_model[valid],
                    y=mu_model[valid],
                    mode="lines",
                    line=dict(width=3, color="#8dcde0"),
                    hovertemplate="Time=%{x:.2f}<br>μ=%{y:.4f}<extra></extra>",
                    showlegend=False,
                    name="Fitted",
                )
            )

    fig.update_layout(
        title=title,
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(
        showgrid=False, title="Time (hours)", range=[float(t.min()), float(t.max())]
    )
    fig.update_yaxes(showgrid=False, title="μ (h⁻¹)")
    return fig


page_header_with_help("Batch Growth Analysis", BATCH_HELP)

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

smoothing_range = get_smoothing_range(len(df_rolling))

### Form ###############################################################################
with st.container(border=True):
    st.header("Step 1. Configure Analysis Options")
    with st.popover("Data used for analysis (rolling median data)", width="stretch"):
        st.dataframe(st.session_state["df_rolling"], width="content")
    analysis_options = render_upload_style_analysis_options(
        s_min=smoothing_range.s_min, s_max=smoothing_range.s_max
    )
    render_parameter_calculation_table_upload_style(analysis_options)
    run_analysis = st.button("Run Analysis", type="primary", width="stretch")

### Render after from submission    ####################################################
if run_analysis and not no_data_uploaded:
    selected_model = analysis_options["selected_model"]
    spline_smoothing_value = analysis_options["spline_smoothing_value"]
    smooth_mode = analysis_options.get("smooth_mode", "fast")
    n_fits_sliding_window = analysis_options["n_fits"]
    n_window_size = analysis_options["window_points"]
    phase_boundary_method = analysis_options["phase_boundary_method"]
    lag_cutoff = analysis_options["lag_cutoff"]
    exp_cutoff = analysis_options["exp_cutoff"]
    min_data_points = analysis_options["min_data_points"]
    min_signal_to_noise = analysis_options["min_signal_to_noise"]
    min_od_increase = analysis_options["min_od_increase"]
    min_growth_rate = analysis_options["min_growth_rate"]

    stats_df_new, fit_cache = _run_model_fitting_on_df_compat(
        df_rolling,
        model_name=selected_model,
        n_fits=n_fits_sliding_window,
        spline_s=spline_smoothing_value,
        smooth_mode=smooth_mode,
        window_points=n_window_size,
        phase_boundary_method=phase_boundary_method,
        lag_cutoff=lag_cutoff,
        exp_cutoff=exp_cutoff,
    )

    st.session_state["batch_analysis_summary_df"] = stats_df_new
    st.session_state["batch_analysis_options"] = {
        "selected_model": selected_model,
        "spline_smoothing_value": spline_smoothing_value,
        "smooth_mode": smooth_mode,
        "n_fits_sliding_window": n_fits_sliding_window,
        "n_window_size": n_window_size,
        "phase_boundary_method": phase_boundary_method,
        "lag_cutoff": lag_cutoff,
        "exp_cutoff": exp_cutoff,
        "min_data_points": min_data_points,
        "min_signal_to_noise": min_signal_to_noise,
        "min_od_increase": min_od_increase,
        "min_growth_rate": min_growth_rate,
    }
    st.session_state["batch_analysis_fit_cache"] = fit_cache


stats_df = st.session_state.get("batch_analysis_summary_df")
batch_options = st.session_state.get("batch_analysis_options")

if stats_df is not None and batch_options is not None:
    msg = """
    Review one reactor at a time with the arrow buttons.
    The red marker shows μmax time and the shaded region marks the exponential phase.
    """
    with st.container(border=True):
        st.header("Step 2. Review Results")
        st.markdown(msg)
        with st.expander("Show data used for model fitting"):
            st.dataframe(df_rolling, width="content")

        reactors = [col for col in df_rolling.columns if col in stats_df.index]
        if not reactors:
            st.warning("No reactors available to display.")
            st.stop()

        if st.session_state.get("batch_selected_reactor") not in reactors:
            st.session_state["batch_selected_reactor"] = reactors[0]

        def _move_reactor(step: int):
            st.session_state["batch_selected_reactor"] = _cycle(
                reactors, st.session_state.get("batch_selected_reactor", reactors[0]), step
            )

        nav_col, popover_col, toggle_col = st.columns(
            [3, 1.2, 1], gap="small", vertical_alignment="bottom"
        )
        with popover_col:
            with st.popover("Annotations", width="stretch"):
                show_phase_boundaries = st.toggle(
                    "Phase boundaries",
                    value=st.session_state.get("batch_show_phase_boundaries", True),
                    key="batch_show_phase_boundaries",
                )
                show_umax_point = st.toggle(
                    "Max growth rate point",
                    value=st.session_state.get("batch_show_umax_point", True),
                    key="batch_show_umax_point",
                )
                show_max_od = st.toggle(
                    "Max OD",
                    value=st.session_state.get("batch_show_max_od", True),
                    key="batch_show_max_od",
                )
                show_baseline_od = st.toggle(
                    "Baseline OD",
                    value=st.session_state.get("batch_show_baseline_od", True),
                    key="batch_show_baseline_od",
                )
                show_fitted_model = st.toggle(
                    "Fitted model curve",
                    value=st.session_state.get("batch_show_fitted_model", True),
                    key="batch_show_fitted_model",
                )
                show_tangent = st.toggle(
                    "Tangent line at max growth",
                    value=st.session_state.get(
                        "batch_show_tangent",
                        False,
                    ),
                    key="batch_show_tangent",
                )
        with toggle_col:
            log_scale = st.toggle(
                "Log scale",
                value=st.session_state.get("batch_log_scale", False),
                key="batch_log_scale",
            )

        with nav_col:
            prev_col, sel_col, next_col = st.columns([1, 4, 1], vertical_alignment="bottom")
            with prev_col:
                st.button(
                    "◀",
                    on_click=_move_reactor,
                    args=(-1,),
                    key="batch_reactor_prev",
                    type="primary",
                    width="stretch",
                )
            with sel_col:
                selected_reactor = st.selectbox(
                    "Reactor",
                    reactors,
                    key="batch_selected_reactor",
                    index=reactors.index(st.session_state["batch_selected_reactor"]),
                )
            with next_col:
                st.button(
                    "▶",
                    on_click=_move_reactor,
                    args=(+1,),
                    key="batch_reactor_next",
                    type="primary",
                    width="stretch",
                )

        fit_cache = st.session_state.setdefault("batch_analysis_fit_cache", {})
        reactor_fit = fit_cache.get(selected_reactor)
        if reactor_fit is None:
            s = df_rolling[selected_reactor].dropna()
            fit_result, _ = gc.fit_model(
                t=s.index.to_numpy(),
                N=s.to_numpy(),
                model_name=batch_options["selected_model"],
                lag_threshold=batch_options["lag_cutoff"],
                exp_threshold=batch_options["exp_cutoff"],
                phase_boundary_method=batch_options["phase_boundary_method"],
                **_build_fit_kwargs(
                    model_name=batch_options["selected_model"],
                    n_fits=batch_options["n_fits_sliding_window"],
                    window_points=batch_options["n_window_size"],
                    spline_s=batch_options["spline_smoothing_value"],
                    smooth_mode=batch_options.get("smooth_mode", "fast"),
                ),
            )
            fit_cache[selected_reactor] = fit_result
            reactor_fit = fit_result

        stats = stats_df.loc[selected_reactor].to_dict()
        s = df_rolling[selected_reactor].dropna()
        scale = "log" if log_scale else "linear"

        fig = gc_plot.create_base_plot(
            s.index.to_numpy(),
            s.to_numpy(),
            scale=scale,
            xlabel=DEFAULT_XLABEL_REL + f" since start at {start_time}",
            marker_opacity=0.3,
        )
        fig = gc_plot.annotate_plot(
            fig,
            fit_result=reactor_fit,
            stats=stats,
            show_fitted_curve=show_fitted_model,
            show_phase_boundaries=show_phase_boundaries,
            show_crosshairs=show_umax_point,
            show_od_max_line=show_max_od,
            show_n0_line=show_baseline_od,
            show_umax_marker=show_umax_point,
            show_tangent=show_tangent,
            scale=scale,
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")

        phase_boundaries = (
            (stats.get("exp_phase_start"), stats.get("exp_phase_end"))
            if pd.notna(stats.get("exp_phase_start"))
            and pd.notna(stats.get("exp_phase_end"))
            else None
        )
        fig_dndt = gc_plot.plot_derivative_metric(
            s.index.to_numpy(),
            s.to_numpy(),
            metric="dndt",
            fit_result=reactor_fit,
            phase_boundaries=phase_boundaries,
            title=f"{selected_reactor} - dN/dt",
            raw_line_width=0,
            smoothed_line_width=0,
            fitted_line_width=3,
        )
        fig_mu = _plot_instantaneous_mu_trace(
            s.index.to_numpy(),
            s.to_numpy(),
            reactor_fit,
            stats,
            title=f"{selected_reactor} - Instantaneous μ",
        )
        fig_dndt.data = tuple(
            trace for trace in fig_dndt.data if getattr(trace, "name", None) == "Fitted"
        )
        st.plotly_chart(fig_dndt, width="stretch")
        st.plotly_chart(fig_mu, width="stretch")

    with st.container(border=True):
        st.subheader("Summary of batch analysis")
        st.write(
            f"The start time was {start_time}. Timepoints are relative to this start time."
        )
        st.dataframe(stats_df, width="content")
    download_data_button_in_sidebar(
        "batch_analysis_summary_df",
        label="Download summary",
        file_name="batch_analysis_summary_df.csv",
    )
