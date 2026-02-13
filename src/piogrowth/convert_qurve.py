import pandas as pd


def build_three_row_header(columns):
    """Take sample names from columns, and add replicate and concentration
    rows with default values of 1 and 0, respectively."""

    header = pd.MultiIndex.from_arrays(
        [list(columns), [1] * len(columns), [0] * len(columns)],
        names=["sample", "replicate", "concentration"],
    )
    return header
