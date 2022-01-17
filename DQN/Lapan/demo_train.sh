#!/bin/sh
export PYTHONPATH=/home/affresco/code/affresco/deep_trading:$PYTHONPATH
cd /home/affresco/code/affresco/deep_trading/dueling_DQN/Lapan/ || exit
/home/affresco/anaconda3/envs/ai/bin/python /home/affresco/code/affresco/deep_trading/dueling_DQN/Lapan/train.py --cuda -r runs -lr 0.0002
