import os
import datetime as dt
import numpy as np
import pandas as pd

import os
from os import path
import pathlib

# ###################################################################
# LOADING DATASETS FROM CSV FILES
# ###################################################################

# ##################################################################
# BASE PATH
# ##################################################################

# This is just for reference: gets us to project-level
PROJECT_DIR = pathlib.Path(__file__).parent.parent.resolve()


# ##################################
# PATHS
# ##################################

def get_data_file_path(currency: str):
    f = get_data_folder_path()
    return f"{f}{str(currency).upper()}.csv"


def get_data_folder_path():
    return os.path.join(PROJECT_DIR, "data")
    # return f"/home/affresco/code/affresco/deep_trading/data/"


# ##################################
# TRAIN & TEST SETS
# ##################################

def load_test_set(currency: str, max_rows=int(1e12)):
    p = os.path.join(get_data_folder_path(), f"{currency}_test.csv")
    return load_dataset(path=p, max_rows=max_rows)


def load_train_set(currency: str, max_rows=int(1e12)):
    p = os.path.join(get_data_folder_path(), f"{currency}_train.csv")
    return load_dataset(path=p, max_rows=max_rows)


# ##################################
# BASE LOADER USING PANDAS
# ##################################

def load_dataset(path: str, max_rows: int = None):
    #
    # Load
    if max_rows is not None:
        df = pd.read_csv(path, nrows=max_rows)
    else:
        df = pd.read_csv(path)
    print(f"Spot dataset loaded with shape {df.shape}")

    print(f"Creating index.")
    df["timestamp"] /= 1e9
    idx = pd.DatetimeIndex([dt.datetime.utcfromtimestamp(d) for d in df["timestamp"].to_numpy()])
    df.set_index(idx, inplace=True)

    df.drop(columns=["timestamp", ], inplace=True)

    # Setting the index
    # df.set_index("timestamp", inplace=True)
    print(f"Spot index set.")

    # Sorting the index in place
    df.sort_index(inplace=True)
    print(f"Spot index sorted.")

    df.index.drop_duplicates(keep="last")
    print(f"Spot (potential) duplicates dropped.")

    df = df.resample("1min").last()
    print(f"Spot re-sampled at 1 minute interval.")

    # Fill this in...
    OHLC = ["open", "high", "low", "close"]
    index_ohlc = [f"index_{ohlc}" for ohlc in OHLC]
    for col in index_ohlc:

        if col not in df.columns:
            continue

        print(f"Filling in column: {col}")
        df[col] = df[col].ffill().bfill()

    print("Dataset loaded.")

    col_mapping = {"perp_open": "open",
                   "perp_high": "high",
                   "perp_low": "low",
                   "perp_close": "close",
                   "quantity": "volume"}

    # df.rename(columns=col_mapping, inplace=True)

    for c in df.columns:

        if "iv_" not in c:
            continue

        df[c] = df[c].replace(np.inf, np.nan, inplace=False)
        df[c] = df[c].ffill().bfill()

    # rescaled_cols = ["open", "high", "low"]
    # for c in rescaled_cols:
    #     df[c] = df[c] / df["close"] - 1.0

    assert "perpetual" in df.columns

    if "spot" in df.columns:
        if np.mean(df.spot) > 10.0:
            print(f"Rescaling 'spot' time series.")
            df.spot = df.spot / df.perpetual - 1.0

    return df
