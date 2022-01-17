#!/usr/bin/env python3
import argparse
import numpy as np

#from lib import environ, data, models

import torch

from rl_models.__DPQ_from_book import models
from environments.options_htm_v01 import OptionsHoldToMaturityEnv
from actions.options_htm_v01 import Actions

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


EPSILON = 0.02


if __name__ == "__main__":


    # Load data into the various environments
    BARS_COUNT = 360
    env = OptionsHoldToMaturityEnv(currency="ETH",
                                   conv_window_size=BARS_COUNT,
                                   start_duration=720,
                                   maximum_duration=1440,
                                   name="env")

    net = models.DQNConv1D(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load("./saves/conv-runs/val_reward--9.758.data", map_location=lambda storage, loc: storage))

    obs = env.reset()
    start_price = env._state._cur_close()

    total_reward = 0.0
    step_idx = 0
    rewards = []

    while True:
        step_idx += 1
        obs_v = torch.tensor([obs])
        out_v = net(obs_v)
        action_idx = out_v.max(dim=1)[1].item()
        if np.random.random() < EPSILON:
            action_idx = env.action_space.sample()

        # Replaced...
        # action = environ.Actions(action_idx)
        action = Actions(action_idx)

        obs, reward, done, _ = env.step(action_idx)
        total_reward += reward
        rewards.append(total_reward)
        if step_idx % 100 == 0:
            print("%d: reward=%.3f" % (step_idx, total_reward))
        if done:
            break

    plt.clf()
    plt.plot(rewards)
    plt.title("Total reward, data=%s" % args.name)
    plt.ylabel("Reward, %")
    plt.savefig("rewards-%s.png" % args.name)
