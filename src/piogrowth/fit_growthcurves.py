import time

import growthcurves as gc
import pandas as pd


def datetimeindex_to_elapsed_hours(index):
    return (index - index[0]).total_seconds() / 3_600


def run_model_fitting_on_df(df, model_name="phenom_richards"):
    stats_df = {}
    for col in df.columns:
        s = df[col].dropna()
        t = datetimeindex_to_elapsed_hours(s.index)
        start = time.time()
        _, stats_df[col] = gc.fit_model(
            time=t.to_numpy(),
            data=s.to_numpy(),
            model_name=model_name,
            window_points=500,
            spline_s=1000,
        )
        end_time = time.time()
        elapsed = end_time - start
        stats_df[col]["elapsed_time"] = elapsed
        stats_df[col]["model_name"] = model_name
        print(f"Finished fitting {model_name} on {col} in {elapsed:.2f} seconds.")

    stats_df = pd.DataFrame(stats_df).T
    return stats_df
