import logging
import datetime as dt

# Scientific
import numpy as np
import pandas as pd

# Local imports
from utilities.paths import *

# ##################################
# PARAMS
# ##################################

# Maximum number of rows to be read by default
MAX_ROWS: int = int(1e6)

# Column suffix for candles: e.g. 'index_close'
OHLC = ["open", "high", "low", "close"]


# ##################################
# TRAIN & TEST SETS
# ##################################

def load_test_set(currency: str, max_rows=MAX_ROWS):
    """
    Load the test set for a given currency (3-letter) symbol.

    :param currency: 3-letter string symbol (e.g. BTC)
    :param max_rows: Maximum number of rows to be read (int)
    :return: pandas DataFrame
    """
    logging.info(f"Loading test dataset in {currency} with a maximum of {max_rows} rows.")
    p = os.path.join(get_data_folder_path(), f"{str(currency).upper()}_test.csv")
    return load_dataset(path=p, max_rows=max_rows)


def load_train_set(currency: str, max_rows=MAX_ROWS):
    """
    Load the train set for a given currency (3-letter) symbol.

    :param currency: 3-letter string symbol (e.g. BTC)
    :param max_rows: Maximum number of rows to be read (int)
    :return: pandas DataFrame
    """
    logging.info(f"Loading training dataset in {currency} with a maximum of {max_rows} rows.")
    p = os.path.join(get_data_folder_path(), f"{str(currency).upper()}_train.csv")
    return load_dataset(path=p, max_rows=max_rows)


# ##################################
# BASIC LOADER USING PANDAS
# ##################################

def load_dataset(path: str, max_rows: int = None):
    """
    Load a dataset for a given currency (3-letter) symbol.

    :param path: Local dir with CSV file in (absolute).
    :param max_rows: Maximum number of rows to be read (int, optional)
    :return: pandas DataFrame
    """
    # Load the raw dataframe using pandas
    df = load_raw_dataframe(max_rows, path)

    # Clean-up the index and resample
    df = reindex_dataframe(df=df, resample=True)

    logging.info("Dataset loaded.")

    # Will fwd-fill spot price data (OHLC)
    # then back-fill the first row
    df = back_and_forward_fill_ohlc_index_columns(df.copy())
    df = clean_iv_columns(df)

    # This is added to prevent mistakes...
    # should be done more thoroughly
    assert "perpetual" in df.columns, "Required column 'perpetual' missing from dataset."
    if "spot" in df.columns:
        if np.mean(df.spot) > 10.0:
            logging.warning(f"Rescaling 'spot' time series.")
            df.spot = df.spot / df.perpetual - 1.0

    return df


# ##################################
# ANCILLARY METHODS: CLEANING
# ##################################

def back_and_forward_fill_ohlc_index_columns(df: pd.DataFrame):
    """
    Fill the DataFrame column forward, then backward.

    :param df: pandas DataFrame
    :return: pandas DataFrame (filled)
    """

    index_ohlc_cols = [f"index_{ohlc}" for ohlc in OHLC]
    for col in index_ohlc_cols:

        if col not in df.columns:
            continue

        logging.info(f"Filling dataframe column {col} forward and back.")
        df[col] = df[col].ffill().bfill()

    return df


def clean_iv_columns(df: pd.DataFrame):
    """
    Cleans the DataFrame columns representing implied vol data by
    replacing infinities by NaN, then filling it.

    :param df: pandas DataFrame
    :return: pandas DataFrame (filled)
    """
    for c in df.columns:
        if "iv_" not in c:
            continue
        df[c] = df[c].replace(np.inf, np.nan, inplace=False)
        df[c] = df[c].ffill().bfill()
    return df


# ##################################
# ANCILLARY METHODS: PANDAS WRAPPER
# ##################################

def load_raw_dataframe(max_rows: int, path: str):
    """
    Load a raw dataframe from a CSV file, basically a Pandas wrapper.

    :param max_rows: maximum number of rows to be read, as int
    :param path: path as string
    :return: pandas DataFrame
    """
    # Load data using pandas
    if max_rows is not None:
        logging.debug(f"Loading dataset with a maximum of {max_rows}.")
        df = pd.read_csv(path, nrows=max_rows)
    else:
        logging.debug(f"Loading dataset without a maximum row count.")
        df = pd.read_csv(path)
    logging.info(f"Dataset loaded with shape: {df.shape}.")
    return df


# ##################################
# ANCILLARY METHODS: INDEX CLEANING
# ##################################

def reindex_dataframe(df: pd.DataFrame, resample: bool = True):
    """
    Re-index the dataframe

    :param df: pandas dataframe
    :param resample: Bool, too make sure the interval is OK.
    :return:
    """

    logging.debug(f"Creating index.")
    df["timestamp"] /= 1e9
    idx = pd.DatetimeIndex([dt.datetime.utcfromtimestamp(d) for d in df["timestamp"].to_numpy()])

    # Set index
    df.set_index(idx, inplace=True)
    df.drop(columns=["timestamp", ], inplace=True)

    # Sort the index in place
    df.sort_index(inplace=True)
    logging.debug(f"Spot index sorted.")

    # Avoid potential duplicates
    df.index.drop_duplicates(keep="last")
    logging.debug(f"Spot (potential) duplicates dropped.")

    if resample:
        df = df.resample("1min").last()
        logging.info(f"Spot re-sampled at 1 minute interval.")

    return df
