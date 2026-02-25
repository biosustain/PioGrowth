from io import BytesIO

import pandas as pd
import streamlit as st
from buttons import create_download_button, download_data_button_in_sidebar
from ui_components import page_header_with_help, show_warning_to_upload_data

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")
no_data_uploaded = st.session_state.get("df_rolling") is None

CALIBRATION_HELP = """
This page applies a linear OD calibration to `df_rolling`.

Workflow:
1. Upload a CSV adjustment table on the Upload Data page (Step 1)
2. Review the adjusted data and plot
3. Download calibrated data from the sidebar

Use with caution.

Calibration assumes a linear relationship between the original and target OD values.

For bioscatter values between zero and five, linear adjustment is often used in
practice. Detailed benchmarking should still be validated for your setup.
"""


def build_od_adjustment_template(df_rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reactor in df_rolling.columns:
        series = df_rolling[reactor].dropna()
        if series.empty:
            start_value = pd.NA
            end_value = pd.NA
        else:
            start_value = series.iloc[0]
            end_value = series.iloc[-1]
        rows.append({"reactor": reactor, "od": start_value})
        rows.append({"reactor": reactor, "od": end_value})
    return pd.DataFrame(rows)


def apply_linear_adjustments(
    df_rolling: pd.DataFrame, adjustment_table: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    required_columns = {"reactor", "od"}
    missing_columns = required_columns - set(adjustment_table.columns)
    if missing_columns:
        return df_rolling, [
            "Adjustment table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        ]

    warnings = []
    adjusted = df_rolling.copy()
    table = adjustment_table.loc[:, ["reactor", "od"]].dropna(subset=["reactor"])

    for reactor, group in table.groupby("reactor", sort=False):
        if reactor not in adjusted.columns:
            warnings.append(f"Reactor '{reactor}' not found in df_rolling columns.")
            continue
        od_values = group["od"].dropna().tolist()
        if len(od_values) < 2:
            warnings.append(
                f"Reactor '{reactor}' has fewer than two OD values in adjustment table."
            )
            continue

        target_start = od_values[0]
        target_end = od_values[-1]
        series = adjusted[reactor].dropna()
        if series.empty:
            warnings.append(f"Reactor '{reactor}' has no data in df_rolling.")
            continue

        original_start = series.iloc[0]
        original_end = series.iloc[-1]
        if original_end == original_start:
            warnings.append(
                f"Reactor '{reactor}' has identical start and end values in df_rolling."
            )
            continue

        slope = (target_end - target_start) / (original_end - original_start)
        intercept = target_start - slope * original_start
        adjusted[reactor] = adjusted[reactor] * slope + intercept

    return adjusted, warnings


page_header_with_help("Calibrate OD Data", CALIBRATION_HELP)

### Stop if df_rolling is not available
if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

with st.container(border=True):
    header_col, req_col = st.columns([4, 1], vertical_alignment="center")
    with header_col:
        st.header("Step 1. OD Adjustment Table (Upload Data page)")
    with req_col:
        with st.popover("Requirements", width="stretch"):
            st.markdown("**Expected CSV columns:**")
            st.markdown("- `reactor`")
            st.markdown("- `od`")
            st.markdown("")
            st.markdown(
                "Provide at least two `od` entries per reactor (start and end targets)."
            )
            st.divider()
            st.markdown("**Example template preview:**")
            template_preview = build_od_adjustment_template(df_rolling).head(10)
            st.dataframe(template_preview, hide_index=True, width="stretch")

    od_adjustment_bytes = st.session_state.get("od_adjustment_upload_bytes")
    od_adjustment_name = st.session_state.get("od_adjustment_upload_name")
    if od_adjustment_bytes is None:
        st.info("Upload the OD adjustment table on the Upload Data page (Step 1).")
        st.page_link(
            "0_upload_data.py",
            label="Go to Upload Data",
            icon=":material/upload:",
        )
    else:
        file_label = od_adjustment_name if od_adjustment_name else "uploaded_file.csv"
        st.success(f"Using uploaded OD adjustment table: `{file_label}`")

    od_adjustment_upload = (
        BytesIO(od_adjustment_bytes) if od_adjustment_bytes is not None else None
    )


if od_adjustment_upload is not None:
    with st.container(border=True):
        st.header("Step 2. Apply Adjustments")
        df_adjustments = pd.read_csv(od_adjustment_upload).convert_dtypes()
        df_rolling, adjustment_warnings = apply_linear_adjustments(
            df_rolling, df_adjustments
        )
        st.session_state["is_df_rolling_adjusted"] = True
        st.session_state["df_rolling"] = df_rolling
        st.subheader("Applied OD adjustments")
        st.dataframe(df_adjustments, width="stretch")
        for warning in adjustment_warnings:
            st.warning(warning)

if st.session_state.get("is_df_rolling_adjusted") not in (None, False):
    with st.container(border=True):
        st.header("Step 2. Review Adjusted Data")
        if od_adjustment_upload is None:
            st.info("Showing adjusted data that was previously uploaded.")
        st.dataframe(df_rolling, width="stretch")

        if not use_elapsed_time:
            view = df_rolling.copy()
            # Map the index (timestamp_rounded) to elapsed_time_in_hours
            print(st.session_state["start_time"])
            view.index = st.session_state["start_time"] + pd.to_timedelta(
                view.index, unit="h"
            )  # .to_timedelta(unit="s")
        else:
            # trigger no copy operation, as index is already in elapsed time
            view = df_rolling

        ax = view.plot.line(style=".", ms=2)
        st.write(ax.get_figure())
        download_data_button_in_sidebar(
            "df_rolling",
            "Download rolling median data",
            file_name="rolling_median_on_filtered_wide_data_with_rounded_timestamps.csv",
        )


### If no file is uploaded, show template and download button
if od_adjustment_upload is None:
    with st.container(border=True):
        st.header("Step 2. Generate Template (Optional)")
        st.markdown(
            """
Fill in the OD template and upload it on the Upload Data page (Step 1).

Template values are initialized from the first and last filtered OD values per reactor.
"""
        )
        adjustment_template = build_od_adjustment_template(df_rolling)
        create_download_button(
            label="Download OD adjustment template",
            data=adjustment_template.to_csv(index=False).encode("utf-8"),
            file_name="od_adjustment_template.csv",
            disabled=False,
            mime="text/csv",
        )
        st.dataframe(adjustment_template, width="stretch")
