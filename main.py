""" Reinforcement learning agent for general game"""
from RL.worker import RolloutsWorker, EvalWorker
from RL.ppo import PPO
from RL.common import eval_policy
from torch.utils.tensorboard import SummaryWriter
from GeneralGame.game import GeneralGame
import time
import os 

import torch

from RL.a2c import A2C

INPUT_SIZE = 21 # State representation
OUTPUT_SIZE = 45 #Num actions
PERSIST_STATE = True
PAYLOAD_ROOT_DIR = None
TENSORBOARD_LOC = None
MODEL_NAME = 'general_PPO_0_0_2'

# Restore 
RESTORE = False
MODEL_CHECKPOINT = None 
CHECKPOINT_EVERY_N_UPDATES = 25

LEARNING_RATE = 2.5e-3
MINIBATCH_SIZE = 1024
ROLLOUT_SIZE = 8192
EPOCHS_PER_UPDATE = 5
GAMMA = 0.99
LAMBDA = 0.95
NUMBER_OF_UPDATES = 2000

EVAL_EVERY_N_UPDATES = 5
N_GAMES_FOR_EVAL = 125

HISTOGRAM_EVERY_N_UPDATES = 50

PPO_CLIP_NORM = 0.2

LOSS_POLICY_WEIGHT = 1
LOSS_VALUE_WEIGHT = 0.20
LOSS_ENTROPY_WEIGHT = 0.01

if __name__ == "__main__":

    a2c = A2C(input_size=INPUT_SIZE,
             num_actions=OUTPUT_SIZE,
             lr=LEARNING_RATE,
             minibatch_size=MINIBATCH_SIZE,
             clip=PPO_CLIP_NORM,
             w_policy=LOSS_POLICY_WEIGHT,
             w_vf=LOSS_VALUE_WEIGHT,
             w_entropy=LOSS_ENTROPY_WEIGHT,
             writer=None,
             global_step=0,
             chkpt= None)

    # Run a2c
    a2c.run(n_workers=2,
            updates = 2,
            epochs=5,
            steps =32,
            gamma=GAMMA,
            lam= LAMBDA)