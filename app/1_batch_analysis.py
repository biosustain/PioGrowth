import inspect
import numpy as np
import pandas as pd
import streamlit as st
import growthcurves as gc
import growthcurves.plot as gc_plot
import plotly.graph_objects as go
import time
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
2. Review linear/log plots and optionally lasso-select points to re-fit
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


def _fit_single_series(
    t_values: np.ndarray,
    n_values: np.ndarray,
    batch_options: dict,
) -> tuple[dict | None, dict]:
    """Fit one reactor series using current batch options."""
    t_arr = np.asarray(t_values, dtype=float)
    n_arr = np.asarray(n_values, dtype=float)
    fit_kwargs = _build_fit_kwargs(
        model_name=batch_options["selected_model"],
        n_fits=batch_options["n_fits_sliding_window"],
        window_points=batch_options["n_window_size"],
        spline_s=batch_options["spline_smoothing_value"],
        smooth_mode=batch_options.get("smooth_mode", "fast"),
    )

    t_start = time.time()
    fit_result, stats = gc.fit_model(
        t=t_arr,
        N=n_arr,
        model_name=batch_options["selected_model"],
        lag_threshold=batch_options["lag_cutoff"],
        exp_threshold=batch_options["exp_cutoff"],
        phase_boundary_method=batch_options["phase_boundary_method"],
        **fit_kwargs,
    )
    stats["elapsed_time"] = time.time() - t_start
    stats["model_name"] = batch_options["selected_model"]
    return fit_result, stats


def _get_selected_times_from_event(event) -> np.ndarray:
    """Extract selected x-values from Streamlit Plotly selection event."""
    if event is None:
        return np.array([], dtype=float)

    selection = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not selection:
        return np.array([], dtype=float)

    points = (
        selection.get("points")
        if isinstance(selection, dict)
        else getattr(selection, "points", None)
    )
    if not points:
        return np.array([], dtype=float)

    x_values = []
    for point in points:
        try:
            x_values.append(float(point["x"]))
        except (TypeError, ValueError, KeyError):
            continue
    return np.asarray(x_values, dtype=float)


def _match_selected_times(
    all_t: np.ndarray,
    selected_times: np.ndarray,
    *,
    time_tolerance: float = 0.01,
) -> np.ndarray:
    """Return indices in all_t matching selected_times within tolerance."""
    if all_t.size == 0 or selected_times.size == 0:
        return np.array([], dtype=int)

    matched = []
    seen = set()
    for sel_t in selected_times:
        hits = np.where(np.abs(all_t - sel_t) < time_tolerance)[0]
        if len(hits) == 0:
            continue
        idx = int(hits[0])
        if idx not in seen:
            matched.append(idx)
            seen.add(idx)
    return np.asarray(matched, dtype=int)


def _collect_selected_series(
    series: pd.Series,
    selected_times: np.ndarray,
    *,
    time_tolerance: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect selected (t, y) points from a reactor series."""
    s = series.dropna()
    if s.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_all = s.index.to_numpy(dtype=float)
    y_all = s.to_numpy(dtype=float)
    idx = _match_selected_times(t_all, selected_times, time_tolerance=time_tolerance)
    if idx.size < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_refit = t_all[idx]
    y_refit = y_all[idx]
    order = np.argsort(t_refit)
    return t_refit[order], y_refit[order]


def _overlay_selected_points(
    fig: go.Figure,
    t_values: np.ndarray,
    y_values: np.ndarray,
    selected_times: list[float] | None,
    *,
    scale: str,
    time_tolerance: float = 0.01,
) -> go.Figure:
    """Overlay included (red) and excluded (gray) points on the growth plot."""
    t_arr = np.asarray(t_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.isfinite(t_arr) & np.isfinite(y_arr) & (y_arr > 0)
    t_arr = t_arr[valid]
    y_arr = y_arr[valid]
    if t_arr.size == 0:
        return fig

    y_plot = np.log(y_arr) if scale == "log" else y_arr
    included = np.ones_like(t_arr, dtype=bool)
    if selected_times:
        sel = np.asarray(selected_times, dtype=float)
        sel = sel[np.isfinite(sel)]
        if sel.size > 0:
            included = np.zeros_like(t_arr, dtype=bool)
            idx = _match_selected_times(t_arr, sel, time_tolerance=time_tolerance)
            included[idx] = True

    # Remove default mono-color points before custom overlays.
    fig.data = tuple(
        trace for trace in fig.data if getattr(trace, "name", None) != "Data"
    )

    excluded = ~included
    if excluded.any():
        fig.add_trace(
            go.Scatter(
                x=t_arr[excluded],
                y=y_plot[excluded],
                mode="markers",
                marker=dict(size=7, color="gray", opacity=0.55),
                hovertemplate="Time=%{x:.2f}<br>OD=%{y:.4f}<extra></extra>",
                showlegend=False,
                name="Excluded",
            )
        )

    if included.any():
        fig.add_trace(
            go.Scatter(
                x=t_arr[included],
                y=y_plot[included],
                mode="markers",
                marker=dict(size=8, color="#ef5350", opacity=0.95),
                hovertemplate="Time=%{x:.2f}<br>OD=%{y:.4f}<extra></extra>",
                showlegend=False,
                name="Included",
            )
        )

    return fig


def _is_bad_fit(gs: dict) -> bool:
    """Return True when stats indicate no growth or failed fit."""
    return gc.inference.is_no_growth(gs or {})


def _get_reactor_stat(stats_df: pd.DataFrame, reactor: str, key: str):
    """Get a scalar stat value even if reactor labels are duplicated."""
    if key not in stats_df.columns or reactor not in stats_df.index:
        return np.nan
    value = stats_df.loc[reactor, key]
    if isinstance(value, pd.Series):
        value = value.dropna()
        return value.iloc[0] if not value.empty else np.nan
    return value


def _get_reactor_stats_dict(stats_df: pd.DataFrame, reactor: str) -> dict:
    """Get one reactor stats row as a plain dictionary."""
    if reactor not in stats_df.index:
        return {}
    row = stats_df.loc[reactor]
    if isinstance(row, pd.DataFrame):
        if row.empty:
            return {}
        row = row.iloc[0]
    return row.to_dict()


def _normalize_smooth(value) -> str:
    """Normalize spline mode to fast/slow with legacy compatibility."""
    mode = str(value).strip().lower() if value is not None else ""
    if mode == "auto":
        return "slow"
    if mode in {"fast", "slow"}:
        return mode
    return "fast"


def _format_smooth(value) -> str:
    """Format spline mode for display."""
    return _normalize_smooth(value).capitalize()


def _growth_method_from_model(model_name: str) -> str:
    """Map selected model name to growth method label."""
    if model_name == "sliding_window":
        return "Sliding Window"
    if model_name == "spline":
        return "Spline"
    return "Model Fitting"


def _default_analysis_params(batch_options: dict) -> dict:
    """Build default analysis parameter dict for readout table."""
    params = {
        "min_od_increase": batch_options.get("min_od_increase"),
        "min_growth_rate": batch_options.get("min_growth_rate"),
        "min_signal_to_noise": batch_options.get("min_signal_to_noise"),
        "min_data_points": batch_options.get("min_data_points"),
    }
    method = _growth_method_from_model(batch_options.get("selected_model", ""))
    if method == "Sliding Window":
        params["window_points"] = batch_options.get("n_window_size")
    elif method == "Spline":
        params["smooth"] = _normalize_smooth(batch_options.get("smooth_mode", "fast"))
    return params


def _format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    gs = gs or {}
    metrics = [
        ("fit_method", "Fit Method", lambda x: str(x) if x else "sliding_window"),
        ("model_rmse", "RMSE", lambda x: f"{float(x):.5f}" if pd.notna(x) else "--"),
        ("max_od", "Maximum OD", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        (
            "mu_max",
            "Maximum Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "intrinsic_growth_rate",
            "Intrinsic Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "time_at_umax",
            "Time at Max Growth (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "od_at_umax",
            "OD at Max Growth",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_start",
            "Lag Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_end",
            "Exponential Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
    ]

    rows = []
    if _is_bad_fit(gs):
        reason = gs.get("no_growth_reason", "--")
        rows.append({"Metric": "No growth reason", "Value": reason})

    for key, label, formatter in metrics:
        value = gs.get(key)
        if key == "mu_max" and value is None:
            value = gs.get("specific_growth_rate")
        try:
            formatted_value = formatter(value) if value is not None else "--"
        except (ValueError, TypeError):
            formatted_value = "--"
        rows.append({"Metric": label, "Value": formatted_value})

    return pd.DataFrame(rows)


def _format_analysis_params_table(
    gs: dict,
    batch_options: dict,
    analysis_params: dict,
    n_total: int | None = None,
    n_selected: int | None = None,
) -> pd.DataFrame:
    """Format analysis parameters into a displayable table."""
    rows = []
    total_str = str(n_total) if n_total is not None else "?"
    selected_str = str(n_selected) if n_selected is not None else total_str
    rows.append(
        {"Parameter": "Data subset (points)", "Value": f"{selected_str}/{total_str}"}
    )

    growth_method = _growth_method_from_model(batch_options.get("selected_model", ""))
    common_params = [
        ("min_od_increase", "Min OD increase", lambda x: f"{float(x):.4f}"),
        ("min_growth_rate", "Min growth rate (1/h)", lambda x: f"{float(x):.5f}"),
        ("min_signal_to_noise", "Min signal-to-noise", lambda x: f"{float(x):.2f}"),
        ("min_data_points", "Min data points", lambda x: str(int(x))),
    ]
    method_params = []
    if growth_method == "Sliding Window":
        method_params = [
            ("window_points", "Window size (points)", lambda x: str(int(x)))
        ]
    elif growth_method == "Spline":
        method_params = [("smooth", "Spline mode", _format_smooth)]

    defaults = _default_analysis_params(batch_options)
    for param_name, plabel, formatter in common_params + method_params:
        value = analysis_params.get(param_name)
        if value is None:
            value = defaults.get(param_name)
        if value is not None:
            try:
                rows.append({"Parameter": plabel, "Value": formatter(value)})
            except (ValueError, TypeError):
                pass

    fit_metrics = [
        (
            "fit_t_min",
            "Analysis window start (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "fit_t_max",
            "Analysis window end (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        ("phase_boundary_method", "Phase boundary method", lambda x: str(x)),
    ]
    for key, plabel, formatter in fit_metrics:
        value = gs.get(key)
        if value is not None:
            try:
                rows.append({"Parameter": plabel, "Value": formatter(value)})
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(rows)


def _update_reactor_stats(
    stats_df: pd.DataFrame,
    reactor: str,
    stats: dict,
):
    """Write a reactor stats dict into the summary DataFrame."""
    for k, v in stats.items():
        stats_df.loc[reactor, k] = v


def _build_analysis_params_per_sample_table(
    stats_df: pd.DataFrame,
    df_rolling: pd.DataFrame,
    batch_options: dict,
    used_params_map: dict | None,
    selected_fit_times_map: dict | None,
) -> pd.DataFrame:
    """Build a per-sample table of analysis parameters used."""
    used_params_map = used_params_map or {}
    selected_fit_times_map = selected_fit_times_map or {}

    method = _growth_method_from_model(batch_options.get("selected_model", ""))
    base_params = _default_analysis_params(batch_options)
    rows = []

    for sample in stats_df.index:
        sample_params = dict(base_params)
        sample_params.update(used_params_map.get(sample, {}))

        total_points = (
            int(df_rolling[sample].dropna().shape[0]) if sample in df_rolling else 0
        )
        selected_times = selected_fit_times_map.get(sample)
        selected_points = len(selected_times) if selected_times else total_points

        row = {
            "sample": sample,
            "model": batch_options.get("selected_model"),
            "growth_method": method,
            "phase_boundary_method": batch_options.get("phase_boundary_method"),
            "min_od_increase": sample_params.get("min_od_increase"),
            "min_growth_rate": sample_params.get("min_growth_rate"),
            "min_signal_to_noise": sample_params.get("min_signal_to_noise"),
            "min_data_points": sample_params.get("min_data_points"),
            "window_points": sample_params.get("window_points"),
            "smooth_mode": sample_params.get("smooth"),
            "selected_points": selected_points,
            "total_points": total_points,
        }

        if sample in stats_df.index:
            sample_stats = _get_reactor_stats_dict(stats_df, sample)
            row["fit_t_min"] = sample_stats.get("fit_t_min")
            row["fit_t_max"] = sample_stats.get("fit_t_max")
        rows.append(row)

    return pd.DataFrame(rows)


page_header_with_help("Batch Growth Analysis", BATCH_HELP)

if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

smoothing_range = get_smoothing_range(len(df_rolling))

### Form ###############################################################################
with st.container(border=True):
    st.header("Step 1. Configure Analysis Options")
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
    st.session_state["batch_selected_fit_times"] = {}
    st.session_state["batch_analysis_used_params"] = {}
    st.session_state.pop("batch_selection_status", None)


stats_df = st.session_state.get("batch_analysis_summary_df")
batch_options = st.session_state.get("batch_analysis_options")

if stats_df is not None and batch_options is not None:
    with st.container(border=True):
        st.header("Step 2. Review Results")

        reactors = [col for col in df_rolling.columns if col in stats_df.index]
        if not reactors:
            st.warning("No reactors available to display.")
            st.stop()

        if st.session_state.get("batch_selected_reactor") not in reactors:
            st.session_state["batch_selected_reactor"] = reactors[0]

        selected_fit_times_map = st.session_state.setdefault(
            "batch_selected_fit_times", {}
        )
        fit_cache = st.session_state.setdefault("batch_analysis_fit_cache", {})
        used_params_map = st.session_state.setdefault("batch_analysis_used_params", {})

        def _move_reactor(step: int):
            st.session_state["batch_selected_reactor"] = _cycle(
                reactors,
                st.session_state.get("batch_selected_reactor", reactors[0]),
                step,
            )

        control_col, phase_col = st.columns(2, gap="large")
        with control_col:
            with st.container(border=True):
                reactor_col, popover_col, toggle_col = st.columns(
                    [2, 0.9, 0.9], vertical_alignment="bottom", gap="small"
                )
                with reactor_col:
                    sample_label = st.session_state.get(
                        "batch_selected_reactor", reactors[0]
                    )
                    st.caption(f"Sample: {sample_label}")
                with popover_col:
                    with st.popover("Annotations", width="stretch"):
                        show_phase_boundaries = st.toggle(
                            "Phase boundaries",
                            value=st.session_state.get(
                                "batch_show_phase_boundaries", True
                            ),
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
                        show_tangent = st.toggle(
                            "Tangent line at max growth",
                            value=st.session_state.get("batch_show_tangent", False),
                            key="batch_show_tangent",
                        )
                        show_fitted_model = st.toggle(
                            "Fitted model curve",
                            value=st.session_state.get("batch_show_fitted_model", True),
                            key="batch_show_fitted_model",
                        )
                with toggle_col:
                    log_scale = st.toggle(
                        "Log scale",
                        value=st.session_state.get("batch_log_scale", False),
                        key="batch_log_scale",
                    )

                prev_col, sel_col, next_col = st.columns(
                    [2, 4, 2], vertical_alignment="bottom"
                )
                with prev_col:
                    st.button(
                        "",
                        width="stretch",
                        on_click=_move_reactor,
                        args=(-1,),
                        key="batch_reactor_prev",
                        shortcut="Left",
                        type="primary",
                    )
                with sel_col:
                    selected_reactor = st.selectbox(
                        "Reactor",
                        reactors,
                        key="batch_selected_reactor",
                        index=reactors.index(
                            st.session_state["batch_selected_reactor"]
                        ),
                    )
                with next_col:
                    st.button(
                        "",
                        width="stretch",
                        on_click=_move_reactor,
                        args=(+1,),
                        key="batch_reactor_next",
                        shortcut="Right",
                        type="primary",
                    )

        s = df_rolling[selected_reactor].dropna()
        if s.empty:
            st.warning(f"No valid data points for {selected_reactor}.")
            st.stop()
        t_all = s.index.to_numpy(dtype=float)
        y_all = s.to_numpy(dtype=float)
        actual_max_od = float(np.nanmax(y_all)) if y_all.size else 0.0

        phase_key = f"batch_phase__{selected_reactor}"
        maxod_key = f"batch_maxod__{selected_reactor}"
        rp_min_od_key = f"batch_rp_min_od__{selected_reactor}"
        rp_min_gr_key = f"batch_rp_min_gr__{selected_reactor}"
        rp_min_snr_key = f"batch_rp_min_snr__{selected_reactor}"
        rp_min_dp_key = f"batch_rp_min_dp__{selected_reactor}"
        rp_window_key = f"batch_rp_window__{selected_reactor}"
        rp_smooth_key = f"batch_rp_smooth__{selected_reactor}"

        if phase_key not in st.session_state:
            exp_phase_start = _get_reactor_stat(
                stats_df, selected_reactor, "exp_phase_start"
            )
            exp_phase_end = _get_reactor_stat(stats_df, selected_reactor, "exp_phase_end")
            lag0 = (
                float(exp_phase_start)
                if pd.notna(exp_phase_start)
                else float(t_all.min())
            )
            exp0 = float(exp_phase_end) if pd.notna(exp_phase_end) else float(t_all.max())
            st.session_state[phase_key] = (lag0, exp0)
        if maxod_key not in st.session_state:
            max_od_stat = _get_reactor_stat(stats_df, selected_reactor, "max_od")
            default_max_od = float(max_od_stat) if pd.notna(max_od_stat) else actual_max_od
            st.session_state[maxod_key] = (
                min(default_max_od, actual_max_od) if actual_max_od > 0 else 0.0
            )

        if rp_min_od_key not in st.session_state:
            st.session_state[rp_min_od_key] = float(
                batch_options.get("min_od_increase", 0.05)
            )
        if rp_min_gr_key not in st.session_state:
            st.session_state[rp_min_gr_key] = float(
                batch_options.get("min_growth_rate", 0.001)
            )
        if rp_min_snr_key not in st.session_state:
            st.session_state[rp_min_snr_key] = float(
                batch_options.get("min_signal_to_noise", 1.0)
            )
        if rp_min_dp_key not in st.session_state:
            st.session_state[rp_min_dp_key] = int(
                batch_options.get("min_data_points", 5)
            )
        if rp_window_key not in st.session_state:
            st.session_state[rp_window_key] = int(
                batch_options.get("n_window_size", 10)
            )
        if rp_smooth_key not in st.session_state:
            st.session_state[rp_smooth_key] = _normalize_smooth(
                batch_options.get("smooth_mode", "fast")
            )

        def _build_effective_options_from_widgets() -> tuple[dict, dict]:
            options_refit = dict(batch_options)
            options_refit["min_od_increase"] = float(st.session_state[rp_min_od_key])
            options_refit["min_growth_rate"] = float(st.session_state[rp_min_gr_key])
            options_refit["min_signal_to_noise"] = float(
                st.session_state[rp_min_snr_key]
            )
            options_refit["min_data_points"] = int(st.session_state[rp_min_dp_key])

            analysis_params = {
                "min_od_increase": options_refit["min_od_increase"],
                "min_growth_rate": options_refit["min_growth_rate"],
                "min_signal_to_noise": options_refit["min_signal_to_noise"],
                "min_data_points": options_refit["min_data_points"],
            }
            method = _growth_method_from_model(batch_options["selected_model"])
            if method == "Sliding Window":
                options_refit["n_window_size"] = int(st.session_state[rp_window_key])
                analysis_params["window_points"] = int(st.session_state[rp_window_key])
            elif method == "Spline":
                options_refit["smooth_mode"] = _normalize_smooth(
                    st.session_state[rp_smooth_key]
                )
                analysis_params["smooth"] = options_refit["smooth_mode"]
            return options_refit, analysis_params

        with phase_col:
            with st.container(border=True):
                t_min, t_max = float(t_all.min()), float(t_all.max())
                step = float(max((t_max - t_min) / 200.0, 0.01))
                slider_col1, slider_col2 = st.columns(2)
                with slider_col1:
                    lag_end, exp_end = st.slider(
                        "Set phase boundaries (hours)",
                        t_min,
                        t_max,
                        step=step,
                        key=phase_key,
                    )
                with slider_col2:
                    if actual_max_od <= 0:
                        st.warning("All OD values are ≤ 0 - no growth detected")
                        max_od = 0.0
                    else:
                        max_od = st.slider(
                            "Set maximum OD",
                            0.0,
                            actual_max_od,
                            step=float(max(actual_max_od / 120, 1e-6)),
                            key=maxod_key,
                        )

                stats_df.loc[selected_reactor, "exp_phase_start"] = float(lag_end)
                stats_df.loc[selected_reactor, "exp_phase_end"] = float(exp_end)
                stats_df.loc[selected_reactor, "max_od"] = float(max_od)

                action_col1, action_col2, action_col3 = st.columns(3)

                def _on_no_growth():
                    new_stats = gc.inference.bad_fit_stats()
                    new_stats["no_growth_reason"] = "manually assigned"
                    new_stats["model_name"] = batch_options["selected_model"]
                    _update_reactor_stats(stats_df, selected_reactor, new_stats)
                    fit_cache.pop(selected_reactor, None)
                    selected_fit_times_map[selected_reactor] = t_all.tolist()
                    used_params_map[selected_reactor] = _default_analysis_params(
                        batch_options
                    )
                    st.session_state["batch_analysis_summary_df"] = stats_df
                    st.session_state["batch_analysis_fit_cache"] = fit_cache
                    st.session_state["batch_selected_fit_times"] = (
                        selected_fit_times_map
                    )
                    st.session_state["batch_analysis_used_params"] = used_params_map

                def _on_reanalyse():
                    options_refit, analysis_params = (
                        _build_effective_options_from_widgets()
                    )
                    used_times = selected_fit_times_map.get(selected_reactor)
                    if used_times:
                        t_refit, y_refit = _collect_selected_series(
                            s, np.asarray(used_times, dtype=float)
                        )
                        if t_refit.size < 2:
                            t_refit, y_refit = t_all, y_all
                            used_times = t_all.tolist()
                    else:
                        t_refit, y_refit = t_all, y_all
                        used_times = t_all.tolist()

                    fit_result_new, stats_new = _fit_single_series(
                        t_refit, y_refit, options_refit
                    )
                    stats_new["exp_phase_start"] = float(lag_end)
                    stats_new["exp_phase_end"] = float(exp_end)
                    stats_new["max_od"] = float(max_od)
                    fit_cache[selected_reactor] = fit_result_new
                    _update_reactor_stats(stats_df, selected_reactor, stats_new)
                    selected_fit_times_map[selected_reactor] = used_times
                    used_params_map[selected_reactor] = analysis_params
                    st.session_state["batch_analysis_summary_df"] = stats_df
                    st.session_state["batch_analysis_fit_cache"] = fit_cache
                    st.session_state["batch_selected_fit_times"] = (
                        selected_fit_times_map
                    )
                    st.session_state["batch_analysis_used_params"] = used_params_map

                def _on_defaults():
                    st.session_state[rp_min_od_key] = float(
                        batch_options.get("min_od_increase", 0.05)
                    )
                    st.session_state[rp_min_gr_key] = float(
                        batch_options.get("min_growth_rate", 0.001)
                    )
                    st.session_state[rp_min_snr_key] = float(
                        batch_options.get("min_signal_to_noise", 1.0)
                    )
                    st.session_state[rp_min_dp_key] = int(
                        batch_options.get("min_data_points", 5)
                    )
                    st.session_state[rp_window_key] = int(
                        batch_options.get("n_window_size", 10)
                    )
                    st.session_state[rp_smooth_key] = _normalize_smooth(
                        batch_options.get("smooth_mode", "fast")
                    )

                    fit_result_new, stats_new = _fit_single_series(
                        t_all, y_all, batch_options
                    )
                    stats_new["exp_phase_start"] = float(lag_end)
                    stats_new["exp_phase_end"] = float(exp_end)
                    stats_new["max_od"] = float(max_od)
                    fit_cache[selected_reactor] = fit_result_new
                    _update_reactor_stats(stats_df, selected_reactor, stats_new)
                    selected_fit_times_map[selected_reactor] = t_all.tolist()
                    used_params_map.pop(selected_reactor, None)
                    st.session_state["batch_analysis_summary_df"] = stats_df
                    st.session_state["batch_analysis_fit_cache"] = fit_cache
                    st.session_state["batch_selected_fit_times"] = (
                        selected_fit_times_map
                    )
                    st.session_state["batch_analysis_used_params"] = used_params_map

                def _on_exclude():
                    fit_cache.pop(selected_reactor, None)
                    selected_fit_times_map.pop(selected_reactor, None)
                    used_params_map.pop(selected_reactor, None)
                    if selected_reactor in stats_df.index:
                        stats_df.drop(index=selected_reactor, inplace=True)
                    remaining = [c for c in df_rolling.columns if c in stats_df.index]
                    if remaining:
                        st.session_state["batch_selected_reactor"] = remaining[0]
                    st.session_state["batch_analysis_summary_df"] = stats_df
                    st.session_state["batch_analysis_fit_cache"] = fit_cache
                    st.session_state["batch_selected_fit_times"] = (
                        selected_fit_times_map
                    )
                    st.session_state["batch_analysis_used_params"] = used_params_map
                    st.rerun()

                with action_col1:
                    st.button(
                        "No Growth",
                        width="stretch",
                        type="primary",
                        key=f"batch_nogrowth__{selected_reactor}",
                        on_click=_on_no_growth,
                    )
                with action_col2:
                    with st.popover("Re-analyse", width="stretch"):
                        st.markdown("**No-growth thresholds**")
                        st.number_input(
                            "Min OD increase",
                            min_value=0.0,
                            step=0.01,
                            format="%.3f",
                            key=rp_min_od_key,
                        )
                        st.number_input(
                            "Min growth rate (1/h)",
                            min_value=0.0,
                            step=0.0001,
                            format="%.4f",
                            key=rp_min_gr_key,
                        )
                        st.number_input(
                            "Min signal-to-noise",
                            min_value=0.0,
                            step=0.1,
                            format="%.2f",
                            key=rp_min_snr_key,
                        )
                        st.number_input(
                            "Min data points",
                            min_value=1,
                            step=1,
                            key=rp_min_dp_key,
                        )
                        method = _growth_method_from_model(
                            batch_options["selected_model"]
                        )
                        if method == "Sliding Window":
                            st.number_input(
                                "Window size (points)",
                                min_value=3,
                                step=1,
                                key=rp_window_key,
                            )
                        elif method == "Spline":
                            st.radio(
                                "Spline fitting mode",
                                options=["fast", "slow"],
                                key=rp_smooth_key,
                                horizontal=True,
                                format_func=lambda v: v.capitalize(),
                            )
                        btn_col, defaults_col = st.columns(2)
                        with btn_col:
                            st.button(
                                "Re-analyse",
                                type="primary",
                                width="stretch",
                                key=f"batch_reanalyse__{selected_reactor}",
                                on_click=_on_reanalyse,
                            )
                        with defaults_col:
                            st.button(
                                "Defaults",
                                width="stretch",
                                type="primary",
                                key=f"batch_restore_defaults__{selected_reactor}",
                                on_click=_on_defaults,
                            )
                with action_col3:
                    st.button(
                        "Exclude from analysis",
                        width="stretch",
                        type="tertiary",
                        key=f"batch_exclude__{selected_reactor}",
                        on_click=_on_exclude,
                    )

        selected_fit_times = selected_fit_times_map.get(selected_reactor)
        if not selected_fit_times:
            selected_fit_times = t_all.tolist()

        reactor_fit = fit_cache.get(selected_reactor)
        current_stats = _get_reactor_stats_dict(stats_df, selected_reactor)
        if reactor_fit is None and len(t_all) >= 2 and not _is_bad_fit(current_stats):
            fit_result, stats_new = _fit_single_series(t_all, y_all, batch_options)
            fit_cache[selected_reactor] = fit_result
            _update_reactor_stats(stats_df, selected_reactor, stats_new)
            st.session_state["batch_analysis_summary_df"] = stats_df
            st.session_state["batch_analysis_fit_cache"] = fit_cache
            reactor_fit = fit_result

        stats = _get_reactor_stats_dict(stats_df, selected_reactor)

        status_col, expander_col = st.columns([2, 5])
        with status_col:
            if _is_bad_fit(stats):
                reason = stats.get("no_growth_reason", "No growth detected")
                st.container(border=True).error(f"**No Growth:** {reason}")
            else:
                st.container(border=True).success("**Growth Detected**")
        with expander_col:
            stats_exp_col, params_exp_col = st.columns(2)
            table_key_base = (
                f"{selected_reactor}_"
                f"{stats.get('mu_max', stats.get('specific_growth_rate', 0))}_"
                f"{stats.get('max_od', 0)}_"
                f"{stats.get('exp_phase_start', 0)}_"
                f"{stats.get('exp_phase_end', 0)}_"
                f"{stats.get('model_rmse', 0)}_"
                f"{id(used_params_map.get(selected_reactor))}"
            )
            with stats_exp_col:
                with st.popover(
                    f"Growth Statistics — {selected_reactor}", width="stretch"
                ):
                    stats_table = _format_growth_stats_table(stats)
                    st.dataframe(
                        stats_table,
                        width="stretch",
                        hide_index=True,
                        key=f"batch_stats_{table_key_base}",
                    )
            with params_exp_col:
                with st.popover(
                    f"Analysis Parameters — {selected_reactor}", width="stretch"
                ):
                    params_table = _format_analysis_params_table(
                        stats,
                        batch_options,
                        used_params_map.get(selected_reactor, {}),
                        n_total=len(s),
                        n_selected=len(selected_fit_times),
                    )
                    st.dataframe(
                        params_table,
                        width="stretch",
                        hide_index=True,
                        key=f"batch_params_{table_key_base}",
                    )
            st.caption(
                "💡 **Tip:** Click and drag on the growth curve plot below to select a subset of data points. "
                "The analysis will be automatically rerun using only the selected points to recalculate growth parameters."
            )

        st.divider()

        scale = "log" if log_scale else "linear"
        fig = gc_plot.create_base_plot(
            t_all,
            y_all,
            scale=scale,
            xlabel=DEFAULT_XLABEL_REL + f" since start at {start_time}",
            marker_opacity=0.3,
        )
        fig = _overlay_selected_points(
            fig,
            t_all,
            y_all,
            selected_fit_times,
            scale=scale,
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
        y_label = "ln(OD600)" if log_scale else "OD600 (baseline-corrected)"
        fig.update_xaxes(
            title="Time (hours)",
            showgrid=False,
            type="linear",
            range=[float(t_all.min()), float(t_all.max())],
        )
        fig.update_yaxes(title=y_label, showgrid=False)
        fig.update_layout(
            uirevision="batch_lasso_keep",
            dragmode="lasso",
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=20),
            height=600,
        )
        chart_key = f"batch_lasso_fit_{selected_reactor}"

        def _on_lasso_select():
            xs = _get_selected_times_from_event(st.session_state.get(chart_key))
            if xs.size < 2:
                return
            refit_t, refit_y = _collect_selected_series(s, xs)
            if refit_t.size < 2:
                return
            options_refit, analysis_params = _build_effective_options_from_widgets()
            fit_result_new, stats_new = _fit_single_series(
                refit_t, refit_y, options_refit
            )
            fit_cache[selected_reactor] = fit_result_new
            _update_reactor_stats(stats_df, selected_reactor, stats_new)
            selected_fit_times_map[selected_reactor] = refit_t.tolist()
            used_params_map[selected_reactor] = analysis_params
            st.session_state["batch_analysis_summary_df"] = stats_df
            st.session_state["batch_analysis_fit_cache"] = fit_cache
            st.session_state["batch_selected_fit_times"] = selected_fit_times_map
            st.session_state["batch_analysis_used_params"] = used_params_map

        st.plotly_chart(
            fig,
            key=chart_key,
            selection_mode="lasso",
            on_select=_on_lasso_select,
            width="stretch",
        )

    with st.container(border=True):
        st.header("Step 3. Overview and Download Results")
        st.write(
            f"The start time was {start_time}. Timepoints are relative to this start time."
        )
        st.dataframe(stats_df, width="stretch")
        st.write("")

        used_params_map = st.session_state.get("batch_analysis_used_params", {})
        selected_fit_times_map = st.session_state.get("batch_selected_fit_times", {})
        params_table = _build_analysis_params_per_sample_table(
            stats_df=stats_df,
            df_rolling=df_rolling,
            batch_options=batch_options,
            used_params_map=used_params_map,
            selected_fit_times_map=selected_fit_times_map,
        )

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "Download rolling median data",
                data=df_rolling.to_csv(index=True).encode("utf-8"),
                file_name="batch_analysis_rolling_median_data.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        with dl_col2:
            st.download_button(
                "Download summary statistics",
                data=stats_df.to_csv(index=True).encode("utf-8"),
                file_name="batch_analysis_summary_stats.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        with dl_col3:
            st.download_button(
                "Download analysis parameters",
                data=params_table.to_csv(index=False).encode("utf-8"),
                file_name="batch_analysis_parameters_by_sample.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
