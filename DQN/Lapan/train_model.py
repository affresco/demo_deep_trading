import wandb
import argparse
import tensorflow as tf
from A3C_discrete import Agent

# ###################################################################
# PARAMETERS
# ###################################################################

CUR_EPISODE = 0

# ###################################################################
# MODULES INIT
# ###################################################################

tf.keras.backend.set_floatx('float64')
wandb.init(name='HTM_A3C', project="deep-rl-tf2")

# ###################################################################
# ARG PARSER INPUTS
# ###################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)

args = parser.parse_args()


# ###################################################################
# LOADING DATASETS FROM CSV FILES
# ###################################################################


def main():
    # env_name = 'CartPole-v1'
    env_name = 'Options-HTM-Env-v01'
    agent = Agent(env_name)
    agent.train()


if __name__ == "__main__":
    main()
