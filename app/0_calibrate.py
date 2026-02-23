"""
- file uploader
- if no file is uploaded, a button should be displayed to manually add the od data

"""

import pandas as pd
import streamlit as st
from buttons import create_download_button, download_data_button_in_sidebar
from ui_components import show_warning_to_upload_data

use_elapsed_time = st.session_state.get("USE_ELAPSED_TIME_FOR_PLOTS", False)
df_rolling = st.session_state.get("df_rolling")
start_time = st.session_state.get("start_time")
no_data_uploaded = st.session_state.get("df_rolling") is None


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


### Start Page
st.title("Calibrate OD data")
st.error("⚠️ Use with caution and read comment:")
st.write(
    """
    Calibration using OD data assumes a linear relationship between the original and 
    target OD values.

    For bioscatter values between zero and five, the linear adjustment is typically 
    sufficient accoring to common community experience. Detailed data analysis has 
    to be published to suppport this claim.
    """
)

### Stop if df_rolling is not available
if no_data_uploaded:
    show_warning_to_upload_data()
    st.stop()

### File Uploader
st.subheader("Upload OD adjustment table")
od_adjustment_upload = st.file_uploader(
    "Upload OD adjustment table",
    type=["csv"],
    key="od_adjustment_table",
)


if od_adjustment_upload is not None:
    ### Show uploaded file and apply adjustments

    df_adjustments = pd.read_csv(od_adjustment_upload).convert_dtypes()
    df_rolling, adjustment_warnings = apply_linear_adjustments(
        df_rolling, df_adjustments
    )
    st.session_state["is_df_rolling_adjusted"] = True
    st.session_state["df_rolling"] = df_rolling
    st.subheader("Applied OD adjustments")
    st.dataframe(df_adjustments)
    for warning in adjustment_warnings:
        st.warning(warning)

if st.session_state.get("is_df_rolling_adjusted") not in (None, False):
    st.subheader("Adjusted OD data:")
    if od_adjustment_upload is None:
        st.info("Show adjusted data which was previously uploaded.")
    st.dataframe(df_rolling)

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
    st.subheader("OD linear adjustment template")
    st.write(
        """
        To adjust the OD values, please fill in the table below and upload it using the 
        uploader above

        The values were  based on the filtered data's first and last OD values.
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
    st.write(adjustment_template)
