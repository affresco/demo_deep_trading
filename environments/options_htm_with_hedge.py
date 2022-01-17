import random
import logging
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
    """
    Class representing the Environment for an option position
    held to maturity (HTM) where the delta exposure of the option
    can be hedged by the RL agent.
    """
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

                 # A currency default
                 currency: str = "BTC",

                 # Some parameters...
                 conv_window_size: int = DEFAULT_CONV_WINDOW_SIZE,

                 # Option duration
                 start_duration: int = 120,
                 target_duration: int = 1440,
                 reach_duration_after: int = 10000,

                 risk_aversion: float = 1.0,

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

        # Risk aversion factor to penalize
        # the AI's drift from the BSM delta
        self.__risk_aversion: float = float(risk_aversion)

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
        logging.info(f"[ENV:{self.name}] {message}")

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
        """
        Select a random offset in the dataset (i.e. a time) to start the experiment.
        :return: Offset as int
        """
        offset = self.np_random.randint(self._state.window_size + 1, self.__sample_size - self.__current_duration)
        return offset

    @classmethod
    def get_option_type(cls):
        """
        Select a put or a call randomly, returns their int equivalent (Put: -1, Call: +1)
        :return: Element from {-1, +1}
        """
        # Select an option type (put: -1, call +1)
        return random.choice([-1, 1])

    def get_option_duration(self, min_duration: int = 360, max_duration: int = 1440):
        """
        Select randomly an option duration in minutes. The option duration will be selected
        in the interval min_duration to max_duration.

        :param min_duration: Minimum duration in minutes (as int)
        :param max_duration: Maximum duration in minutes (as int)
        :return: Duration as int
        """

        # Avoids numerical issues
        if min_duration == max_duration:
            return min_duration

        # Over 1k options
        x = self.__reach_duration_after
        span_from_epoch = (max_duration - min_duration) * min(1.0, self.episodes / x)
        return random.randint(min_duration, int(1 + min_duration + span_from_epoch))

    @classmethod
    def get_option_notional(cls):
        """
        Provides a notional value, for the time being just long (+1) or short (-1).
        :return: Element from {-1, +1}
        """
        return random.choice([-1, 1])

    @classmethod
    def get_option_strike_moneyness(cls, option_type: int, otm: float = 0.0, scale: float = 0.025):
        loc = otm * option_type
        return 1.0 + numpy.random.normal(loc=loc, scale=scale)

    def get_risk_aversion(self, risk_aversion: float, floor_risk: float = 1.0):
        """
        Provides a numerical value for the risk aversion parameter, which can be
        lowered with the number of episodes to enforce a different behaviour.

        :param risk_aversion: Float
        :param floor_risk: Float
        :return:
        """
        if risk_aversion <= floor_risk:
            return floor_risk
        return floor_risk + (risk_aversion - floor_risk) / (1.0 + self.episodes / self.__reach_duration_after)

    # ##################################
    # INTERFACE: RESET
    # ##################################

    def reset(self):
        """
        Reset the episode. Required by the OpenAI Gym framework.

        :return: Initial state of the world
        """

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

        risk_aversion = self.get_risk_aversion(self.__risk_aversion)

        self._state.reset(
            offset=offset,
            notional=notional,
            option_type=int(option_type),
            option_strike_moneyness=option_strike_moneyness,
            max_duration=duration,
            force_duration=True,
            risk_aversion=risk_aversion,
        )

        # Take a note
        self.last_reset = time.time()

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
