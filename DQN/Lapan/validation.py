import numpy as np

import torch

# from lib import environ
# from lib.environ import StocksEnv
import wandb

from environments.options_htm_v01 import OptionsHoldToMaturityEnv as StocksEnv

METRICS = (
    'hedge_increase_count',
    'hedge_decrease_count',
    'hedge_cashflows',
    'episode_reward',
    'final_account_mtm',
)

_OLD_METRICS = (
    'episode_reward',
    'episode_steps',
    'order_profits',
    'order_steps',
)


def validation_run(env: StocksEnv, net, episodes=250, device="cpu", epsilon=0.001,):

    stats = {metric: [] for metric in METRICS}

    for episode in range(episodes):

        obs = env.reset()
        total_reward = 0.0

        while True:

            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()

            # Compute step as usual
            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward

            if done:
                final_reward = env.state.position.cash_balance
                break

        stats['episode_reward'].append(total_reward)
        stats['final_account_mtm'].append(final_reward)
        stats['hedge_increase_count'].append(env.option_position.hedge_increase_count)
        stats['hedge_decrease_count'].append(env.option_position.hedge_decrease_count)
        stats['hedge_cashflows'].append(env.option_position.hedge_total_cashflow)

    res_mean = {key: np.mean(vals) for key, vals in stats.items()}
    wandb.log(res_mean)
    return res_mean

