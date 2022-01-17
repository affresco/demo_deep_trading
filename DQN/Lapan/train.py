#!/usr/bin/env python3
import os
import datetime as dt

import ptan
import pathlib
import argparse
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from environments.options_htm_v01 import OptionsHoldToMaturityEnv

from rl_models.__DPQ_from_book import models
from rl_models.__DPQ_from_book import common
from rl_models.__DPQ_from_book import validation

import wandb

run_time = dt.datetime.utcnow().strftime("%d%b%y_%H:%M").upper()
wandb.init(name=f"Option-HTM-{run_time}", project="deep-trading", entity="affresco")

# ###################################################################
# PARAMETERS
# ###################################################################

SAVES_DIR = pathlib.Path("saves")

STOCKS = "./data/BTC_train.csv"
VAL_STOCKS = "./data/BTC_test.csv"

BATCH_SIZE = 32
BARS_COUNT = 120

EPS_START = 1.0
EPS_FINAL = 0.15
EPS_STEPS = 100000

GAMMA = 0.99

REPLAY_SIZE = 1000000
REPLAY_INITIAL = 10000

REWARD_STEPS = 10
LEARNING_RATE = 0.0002
STATES_TO_EVALUATE = 10000

DEFAULT_CURRENCY = "BTC"
CURRENCIES = ["BTC", "ETH"]
MAX_ROWS = 1e6

SAVE_ID = 0

# ###################################################################
# ARG PARSER INPUTS
# ###################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", help="Enable cuda", default=False, action="store_true")
parser.add_argument("--data", default=STOCKS, help=f"Stocks file or dir, default={STOCKS}")
parser.add_argument("--year", type=int, help="Year to train on, overrides --data")
parser.add_argument("--val", default=VAL_STOCKS, help="Validation data, default=" + VAL_STOCKS)
parser.add_argument("-r", "--run", required=True, help="Run name")
parser.add_argument("-lr", help="Set learning rate", type=float, default=LEARNING_RATE)

args = parser.parse_args()


# ###################################################################
# LOADING DATASETS FROM CSV FILES
# ###################################################################

def main():

    wandb.config.gamma = GAMMA
    wandb.config.rewards_steps = REWARD_STEPS

    wandb.config.replay_size = REPLAY_SIZE
    wandb.config.replay_initial = REPLAY_INITIAL

    wandb.config.batch_size = BATCH_SIZE

    wandb.config.states_to_evaluate = STATES_TO_EVALUATE

    wandb.config.epsilon_start = EPS_START
    wandb.config.epsilon_final = EPS_FINAL
    wandb.config.epsilon_steps = EPS_STEPS

    # Select a device
    device = torch.device("cuda" if args.cuda else "cpu")
    wandb.config.device = device

    lr = args.lr
    wandb.config.learning_rate = lr
    wandb.config.optimizer = "adam"

    last_run_id = find_last_run_id(SAVES_DIR)

    # Select path to save results
    run_name = f"conv1d-{last_run_id}"
    saves_path = SAVES_DIR / f"{run_name}"
    saves_path.mkdir(parents=True, exist_ok=True)
    wandb.config.path = saves_path
    wandb.config.name = run_name

    ccy = str(DEFAULT_CURRENCY).upper()
    wandb.config.currency = ccy

    min_duration = 5
    wandb.config.min_duration = min_duration

    target_duration = 60
    wandb.config.target_duration = target_duration

    reach_duration_after = int(1)
    wandb.config.reach_duration_after = reach_duration_after

    # Load data into the various environments
    env = OptionsHoldToMaturityEnv(currency=ccy,
                                   conv_window_size=BARS_COUNT,
                                   start_duration=min_duration,
                                   target_duration=target_duration,
                                   reach_duration_after=reach_duration_after,
                                   name="env")

    env_tst = OptionsHoldToMaturityEnv(currency=ccy,
                                       conv_window_size=BARS_COUNT,
                                       start_duration=min_duration,
                                       target_duration=target_duration,
                                       reach_duration_after=reach_duration_after,
                                       name="env_tst")

    # Use the full range
    env_val = OptionsHoldToMaturityEnv(currency=ccy,
                                       conv_window_size=BARS_COUNT,
                                       start_duration=min_duration,
                                       target_duration=target_duration,
                                       reach_duration_after=reach_duration_after,
                                       name="env_val")

    # Build a NN model
    net = models.DQNConv1D(env.observation_space.shape, env.action_space.n).to(device)
    # net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    tgt_net = ptan.agent.TargetNet(net)

    # Action selection via Eps Greedy
    selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = ptan.actions.EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)

    # RL Agent
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # Experience Source
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    # Replay Buffer
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    optim_scheduler_steps = 10000
    optim_scheduler_gamma = 0.99
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=optim_scheduler_steps, gamma=optim_scheduler_gamma)
    wandb.config.optim_scheduler_steps = optim_scheduler_steps
    wandb.config.optim_scheduler_gamma = optim_scheduler_gamma

    def process_batch(engine, batch):

        scheduler.step()
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']})

        optimizer.zero_grad()
        loss_v = common.calc_loss(
            batch, net, tgt_net.target_model,
            gamma=GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        eps_tracker.frame(engine.state.iteration)

        if getattr(engine.state, "eval_states", None) is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.array(transition.state, copy=False)
                           for transition in eval_states]
            engine.state.eval_states = np.array(eval_states, copy=False)

        # Log to wandb.ai
        wandb.log({"loss": loss_v.item()})
        wandb.log({"epsilon": selector.epsilon})

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    # tb = common.setup_ignite(engine, exp_source, f"conv-{args.run}", extra_metrics=('values_mean',))
    tb = common.setup_ignite(engine, exp_source, f"{run_name}", extra_metrics=('values_mean',))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def sync_eval(engine: Engine):

        print(f"*"*100)
        print(f"SYNCING EVAL AND TARGET NETWORKS")
        print(f"*"*100)


        tgt_net.sync()

        mean_val = common.calc_values_of_states(
            engine.state.eval_states, net, device=device)

        engine.state.metrics["values_mean"] = mean_val
        wandb.log({"values_mean": mean_val})

        if getattr(engine.state, "best_mean_val", None) is None:
            engine.state.best_mean_val = mean_val

        if engine.state.best_mean_val < mean_val:
            print("%d: Best mean value updated %.3f -> %.3f" % (
                engine.state.iteration, engine.state.best_mean_val,
                mean_val))
            path = saves_path / ("mean_value-%.3f.data" % mean_val)
            torch.save(net.state_dict(), path)
            engine.state.best_mean_val = mean_val

    @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def periodic_model_save(engine: Engine):
        path = os.path.join(saves_path, "periodic")
        os.makedirs(path, exist_ok=True)
        run_id = find_last_run_id(path)
        torch.save(net.state_dict(), os.path.join(path, f"model-{run_id}.data"))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def validate(engine: Engine):
        res = validation.validation_run(env_tst, net, device=device)
        print("%d: tst: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_tst"] = val
        res = validation.validation_run(env_val, net, device=device)
        print("%d: val: %s" % (engine.state.iteration, res))
        for key, val in res.items():
            engine.state.metrics[key + "_val"] = val
        val_reward = res['episode_reward']
        if getattr(engine.state, "best_val_reward", None) is None:
            engine.state.best_val_reward = val_reward
        if engine.state.best_val_reward < val_reward:
            print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                engine.state.best_val_reward, val_reward
            ))
            engine.state.best_val_reward = val_reward
            path = saves_path / ("val_reward-%.3f.data" % val_reward)
            torch.save(net.state_dict(), path)

    event = ptan.ignite.PeriodEvents.ITERS_10000_COMPLETED
    tst_metrics = [m + "_tst" for m in validation.METRICS]
    tst_handler = tb_logger.OutputHandler(
        tag="test", metric_names=tst_metrics)
    tb.attach(engine, log_handler=tst_handler, event_name=event)

    val_metrics = [m + "_val" for m in validation.METRICS]
    val_handler = tb_logger.OutputHandler(
        tag="validation", metric_names=val_metrics)
    tb.attach(engine, log_handler=val_handler, event_name=event)

    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))


def find_last_run_id(directory):
    runs = os.listdir(directory)
    if not len(runs):
        last_run_id = 0
    else:
        last_run_id = max([int(r.split("-")[-1].split(".")[0]) for r in runs]) + 1
    return last_run_id


if __name__ == "__main__":
    main()
