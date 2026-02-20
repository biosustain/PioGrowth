import time

import growthcurves as gc
import pandas as pd


def datetimeindex_to_elapsed_hours(index):
    return (index - index[0]).total_seconds() / 3_600


def run_model_fitting_on_df(
    df,
    model_name="phenom_richards",
    window_points=500,
    spline_s=1000,
    n_fits=50,
    phase_boundary_method=None,
    **kwargs,
):
    stats_df = {}
    for col in df.columns:
        s = df[col].dropna()
        t = s.index
        start = time.time()
        _, stats_df[col] = gc.fit_model(
            t=t.to_numpy(),
            N=s.to_numpy(),
            model_name=model_name,
            window_points=window_points,
            spline_s=spline_s,
            n_fits=n_fits,
            phase_boundary_method=phase_boundary_method,
            **kwargs,
        )
        end_time = time.time()
        elapsed = end_time - start
        stats_df[col]["elapsed_time"] = elapsed
        stats_df[col]["model_name"] = model_name
        print(f"Finished fitting {model_name} on {col} in {elapsed:.2f} seconds.")

    stats_df = pd.DataFrame(stats_df).T
    return stats_df
