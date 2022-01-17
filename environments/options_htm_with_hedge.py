import random
from typing import Dict, Optional, List

import gym
import gym.spaces
import numpy.random
import pandas as pd
from gym.utils import seeding
from gym.envs.registration import EnvSpec

import numpy as np

# Actions that can be taken
from actions.htm_with_hedge import Actions

# Agent state
from states.conv_1D import State1D

from utilities import csv_loader

import time

# ###################################################################
# PARAMETERS
# ###################################################################


DEFAULT_CONV_WINDOW_SIZE = 60
RISK_AVERSION = 10.0


# ###################################################################
# OPEN AI GYM ENVIRONMENT
# ###################################################################

class OptionsHoldToMaturityEnv(gym.Env):
    #
    #
    #
    display_name = "Options-HTM-Env-v01"
    metadata = {'render.modes': ['human']}
    spec = EnvSpec(f"{display_name}")
    #
    # Total number of episodes generated
    episodes = 0
    episode_steps = 0
    #
    last_reset = 0.0
    #
    #
    available_data = {}

    def __init__(self,

                 # A currency to play with
                 currency: str = "BTC",

                 # Some parameters...
                 conv_window_size: int = DEFAULT_CONV_WINDOW_SIZE,

                 # Option duration
                 start_duration: int = 120,
                 target_duration: int = 1440,
                 reach_duration_after: int = 10000,

                 name: str = "Options-HTM-Env-v01"):

        # Give it a name for display
        self.name: str = str(name).upper()

        # Time series dataframes keyed by their instrument/currency as str
        assert isinstance(currency, str), "Currencies must be passed as string (e.g. BTC)"
        self.__currency: Optional[str] = currency.upper()

        # All datasets are stored in memory
        self.__feature_dataset: pd.DataFrame = self.load_dataset(currency=currency)

        # About our dataset...
        self.__sample_size = self.__feature_dataset.shape[0]
        self.__feature_count = self.__feature_dataset.shape[1]

        # Multiply by -1 each time will,
        # alternate between puts and calls
        self.__option_type: int = 1

        # Create the state
        self._state = State1D(
            currency=self.__currency,
            features=self.__feature_dataset,
            conv_window_size=conv_window_size,
            name=self.name
        )

        # Maximum option TTM in minutes
        self.__current_duration: int = int(start_duration)
        self.__initial_duration: int = int(start_duration)
        self.__target_duration: int = int(target_duration)
        self.__reach_duration_after: int = int(reach_duration_after)

        # Build action space
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        # Observation space request State object for dimensions...
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)

        # Conv 1D window size
        self.__window_size = conv_window_size

        self.seed()
        self.log(f"Initialization completed.")

    # ##################################
    # PROPERTIES
    # ##################################

    @property
    def global_epochs(self):
        return self.episodes

    @property
    def window_size(self):
        return self.__window_size

    @property
    def state(self):
        return self._state

    @property
    def option_position(self):
        return self._state.position

    # ##################################
    # DISPLAY
    # ##################################

    def log(self, message: str):
        print(f"[ENV:{self.name}] {message}", flush=True)

    # ##################################
    # DATA LOADING
    # ##################################

    def load_dataset(self, currency: str):

        if currency in self.available_data:
            self.log(f"Returning readily available dataset in {currency}.")
            return self.available_data[currency]

        self.log(f"Loading training dataset in {currency}.")
        df = csv_loader.load_train_set(currency, max_rows=250000)
        self.available_data[currency] = df

        return df

    # ##################################
    # PROPERTIES
    # ##################################

    def get_random_offset(self):
        ttm = self.__current_duration
        offset = self.np_random.randint(self._state.window_size + 1, self.__sample_size - ttm)
        return offset

    @classmethod
    def get_option_type(cls):
        # Select an option type (put: -1, call +1)
        return random.choice([-1, 1])

    def get_option_duration(self, min_duration: int = 360, max_duration: int = 1440):
        if min_duration == max_duration:
            return min_duration

        # Over 1k options
        x = self.__reach_duration_after
        span_from_epoch = (max_duration - min_duration) * min(1.0, self.episodes / x)
        return random.randint(min_duration, int(1 + min_duration + span_from_epoch))

    @classmethod
    def get_option_notional(cls):
        return random.choice([-1, 1])

    def __old__get_option_strike_moneyness(self, option_type: int, itm_k: float = 0.01, otm_k: float = 0.05):
        moneyness = self.np_random.uniform(-itm_k, otm_k)  # always close to OTM
        return 1.0 + (moneyness * option_type)

    def get_option_strike_moneyness(self, option_type: int, otm: float = 0.0, scale: float = 0.025):
        loc = otm * option_type
        return 1.0 + numpy.random.normal(loc=loc,
                                         scale=scale)

    def get_risk_aversion(self):
        floor = 1.0
        ra = floor + (RISK_AVERSION - floor) / (1.0 + self.episodes / self.__reach_duration_after)
        return ra

    # ##################################
    # INTERFACE: RESET
    # ##################################

    def reset(self):

        st = time.time()

        # Add to episode counter
        self.episodes += 1

        # Select an option type (put: -1, call +1)
        option_type = self.get_option_type()

        # Select a notional
        notional = self.get_option_notional()

        # Select an option strike: positive is OTM, negative is ITM
        option_strike_moneyness = self.get_option_strike_moneyness(option_type=self.__option_type,
                                                                   otm=0.0050,
                                                                   scale=0.025, )

        # Get a random starting point (in time)
        offset = self.get_random_offset()

        # Option duration in minutes
        duration = self.get_option_duration(min_duration=self.__initial_duration,
                                            max_duration=self.__target_duration)

        risk_aversion = 1.0  #self.get_risk_aversion()

        self._state.reset(
            offset=offset,
            notional=notional,
            option_type=int(option_type),
            option_strike_moneyness=option_strike_moneyness,
            max_duration=duration,
            force_duration=True,
            risk_aversion=risk_aversion,
        )

        just_now = time.time()

        print(
            f"Reset took: {1000.0 * (just_now - st)} ms, last reset occurred {1000.0 * (st - self.last_reset)} ms ago.")
        self.last_reset = st

        if self.episodes % 100 == 0:
            self.log(f"Resetting at episode number {self.episodes}.")

        return self._state.encode()

    # ##################################
    # INTERFACE: STEP
    # ##################################

    def step(self, action_idx):
        self.episode_steps += 1
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "currency": self._state.currency,
            "offset": self._state.offset
        }
        return obs, reward, done, info

    # ##################################
    # INTERFACE: ANCILLARY
    # ##################################

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


if __name__ == '__main__':
    # Data is loaded by the environment itself from the ../data/ folder
    env = OptionsHoldToMaturityEnv(currency="BTC", name="A3C-Train")
