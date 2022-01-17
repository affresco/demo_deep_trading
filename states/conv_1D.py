import datetime as dt
from typing import Optional

# Scientific
import math
import numpy as np
import pandas as pd

# Local
from actions.htm_with_hedge import Actions
from states.option import HedgedOptionPosition

# ####################################################################
# CONSTANTS
# ####################################################################

MINUTES_PER_YEAR = 365.2422 * 24 * 60

EPSILON_TTM = (1.0 / 60.0) / MINUTES_PER_YEAR  # 1 second
EPSILON_IV = 0.0001  # 1 bps


# ####################################################################
# AGENT STATE FOR 1D CONVOLUTIONAL NEURAL NETWORK
# ####################################################################

class State1D:
    #
    #
    def __init__(self,
                 currency: str,
                 features: pd.DataFrame,
                 conv_window_size: int,
                 name: str = "no_name", ):
        #
        #
        assert isinstance(conv_window_size, int)
        assert conv_window_size > 0

        # Give this state a name corresponding to its env
        self.name: str = str(name).upper()

        # Conv model backward-looking window size
        self.window_size: int = max(1, int(conv_window_size))

        # Our option position including pricing model
        self.__position: Optional[HedgedOptionPosition] = None

        # Only known at runtime
        self.__offset: int = 0
        self.__warmup_offset = 0  # matches the conv win size
        self.__expiry_offset = 0

        # The first offset provided by the env for every run
        self.__offset_reference: int = 0

        # For display purposes only
        self.__currency: str = currency

        assert isinstance(features, pd.DataFrame), "Feature sets must be passed as a DataFrame"
        self.__features: pd.DataFrame = features

        # We know how many (common) extrinsic
        # features have been passed to this state
        # About our dataset...
        self.__sample_size = self.__features.shape[0]
        self.__feature_count = self.__features.shape[1]

        # Features descriptors
        self.__features_ex_count: Optional[int] = self.__feature_count  # tbd
        self.__features_in_count: Optional[int] = 22  # fixed (based on our hard coded calculations in here)

        # How many samples in each feature set
        self.__features_time_steps = {}

        # Log for user
        self.log(f"Initialization completed.")

    # ####################################################################
    # PROPERTIES
    # ####################################################################

    @property
    def currency(self):
        return self.__currency

    @property
    def position(self):
        return self.__position

    @property
    def offset(self):
        return self.__offset

    @property
    def current_ts(self):
        return self.__features.index[self.__offset]

    @property
    def start_ts(self):
        return self.__features.index[self.__offset_reference]

    @property
    def duration(self):
        return self.__expiry_offset - self.__offset

    @property
    def intrinsic_features_count(self):
        return self.__features_in_count

    @property
    def extrinsic_features_count(self):
        return self.__features_ex_count

    @property
    def shape(self):
        features_in = self.intrinsic_features_count
        features_ex = self.extrinsic_features_count
        features_total = features_in + features_ex + 1
        return features_total, self.window_size

    # ####################################################################
    # DISPLAY
    # ####################################################################

    def log(self, message: str):
        print(f"[AGENT:{self.name}] {message}", flush=True)

    # ####################################################################
    # RESET STATE
    # ####################################################################

    def reset(self,
              offset: int,
              option_type: int,
              option_strike_moneyness: float,
              notional: float,
              max_duration: int = 1440,
              force_duration: bool = False,
              risk_aversion: float = 1.0,
              ):

        # Reset experiment context
        warmup_offset, offset, expiry_offset = self.reset_experiment(offset=offset,
                                                                     max_duration=max_duration,
                                                                     force_duration=force_duration)

        # Select the relevant features
        features = self.__features[warmup_offset:expiry_offset + 1]

        # Just to be safe
        del self.__position

        self.__position = HedgedOptionPosition(
            option_type=int(option_type),
            notional=float(notional),
            window_size=int(self.window_size),
            extrinsic_features=features,
            strike_moneyness=float(option_strike_moneyness),
            risk_aversion=risk_aversion,
        )

        self.__position.reset()

    def reset_experiment(self, offset: int,
                         max_duration: int,
                         force_duration: bool = False, ):

        # Compute the time-related parameters
        warmup_tuple, filtration_tuple, expiry_tuple = self.compute_time_offsets(offset=offset,
                                                                                 max_duration=max_duration,
                                                                                 force_duration=force_duration)

        # Completely dependent on external input
        self.__offset: int = int(filtration_tuple[1])
        self.__offset_reference: int = int(self.__offset)
        self.__warmup_offset = int(warmup_tuple[1])
        self.__expiry_offset: int = int(expiry_tuple[1])

        # All 'clean' offsets
        return self.__warmup_offset, self.__offset, self.__expiry_offset

    # ####################################################################
    # COMPUTES
    # ####################################################################

    def compute_time_offsets(self, offset: int, max_duration: int = 1440, force_duration: bool = False):

        if offset <= self.window_size:
            print(f"[STATE:{self.name}] Offset requested was too early, replaced by {self.window_size}.")
        offset = max(int(offset), int(self.window_size))

        # Get a filtration point for the observation time
        filtration_ts = self.__features.index[offset]

        # Start of the window (backward-looking)
        obs_win_start_offset = offset - self.window_size
        obs_win_start_ts = self.__features.index[obs_win_start_offset]

        if force_duration:
            expiry_offset = min(offset+max_duration, self.__sample_size-1)
            expiry_ts = self.__features.index[expiry_offset]
            return (obs_win_start_ts, obs_win_start_offset), (filtration_ts, offset), (expiry_ts, expiry_offset)

        # Get the corresponding expiry timestamp
        hrs = 8 if filtration_ts.hour < 8 else 32
        ts_shift = dt.timedelta(hours=hrs)
        expiry_ts = dt.datetime(year=filtration_ts.year, month=filtration_ts.month, day=filtration_ts.day) + ts_shift

        # Get a duration for this option contract
        duration_in_minutes = int((expiry_ts - filtration_ts).total_seconds() / 60.0)
        if max_duration is not None:
            duration_in_minutes = min(duration_in_minutes, int(max_duration))

        # Compute the expiry time
        expiry_offset = offset + duration_in_minutes
        max_offset = self.__sample_size - 1
        if expiry_offset > max_offset:
            print(f"[STATE:{self.name}] Expiry offset falling outside sample set, replaced by {max_offset}.")

        # Reset to account for sample set dimensions
        expiry_offset = min(max_offset, expiry_offset)
        expiry_ts = self.__features.index[expiry_offset]

        return (obs_win_start_ts, obs_win_start_offset), (filtration_ts, offset), (expiry_ts, expiry_offset)

    # ####################################################################
    # TAKE STEP
    # ####################################################################

    def step(self, action):

        time_step = self.__offset - self.__offset_reference

        # We've expiring, no actions to be taken
        if self.__offset >= self.__expiry_offset:
            # print(f"Expiry detected on {self.current_ts}")
            final_reward = self.__position.expire(time_step)
            return float(final_reward), True

        # Sanity check
        assert isinstance(action, Actions)

        # Mark-to-market
        reward = self.__position.mark_to_market(time_step)

        # For the reward
        fees = 0.0

        if action == Actions.Skip:
            pass

        elif action == Actions.Buy_Perpetual:
            fees = self.__position.buy_perpetual(offset=time_step)
            pass

        elif action == Actions.Sell_Perpetual:
            fees = self.__position.sell_perpetual(offset=time_step)
            pass

        self.__offset += 1
        return reward - math.fabs(fees), False

    # ####################################################################
    # ENCODE DATA
    # ####################################################################

    def encode(self):
        time_step = self.__offset - self.__offset_reference
        return self.__position.encode(offset=time_step)
