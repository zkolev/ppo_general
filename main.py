""" Reinforcement learning agent for general game"""
from RL.worker import RolloutsWorker, EvalWorker
from RL.ppo import PPO
from RL.common import eval_policy
from torch.utils.tensorboard import SummaryWriter
from GeneralGame.game import GeneralGame
import time

import torch

#HP:

INPUT_SIZE = 21 # State representation
OUTPUT_SIZE = 45 #Num actions
MODEL_ROOT_DIR = f"E:\\TB\\{int(time.time())}"
LEARNING_RATE = 2.5e-3
MINIBATCH_SIZE = 1024
ROLLOUT_SIZE = 8192
EPOCHS_PER_UPDATE = 5
GAMMA = 0.99
LAMBDA = 0.95
NUMBER_OF_UPDATES = 2000
EVAL_EVERY_N_UPDATES = 5
N_EVAL_GAMES = 250
LOSS_POLICY_WEIGHT = 1
LOSS_VALUE_WEIGHT = 0.05
LOSS_ENTROPY_WEIGHT = 0.05

if __name__ == "__main__":
    print("Init Worker")

    game = GeneralGame()
    writer = SummaryWriter(MODEL_ROOT_DIR) #TODO: Programatically assighn root dir
    w = RolloutsWorker()

    # Policy
    ppo = PPO(INPUT_SIZE, OUTPUT_SIZE,
              minibatch_size=MINIBATCH_SIZE,
              lr=LEARNING_RATE,
              writer=writer,
              w_policy=LOSS_POLICY_WEIGHT,
              w_vf=LOSS_VALUE_WEIGHT,
              w_entropy=LOSS_ENTROPY_WEIGHT
              )

    # Init shared variables:
    weights = ppo.network.state_dict()


    # Main loop
    for i in range(NUMBER_OF_UPDATES):
        print(f'POLICY UPDATE  {i} ... ')
        traj = w.roll_out(weights, ROLLOUT_SIZE, GAMMA, LAMBDA)

        weights = ppo.update(traj=[traj],
                             epochs=EPOCHS_PER_UPDATE)

        if i % EVAL_EVERY_N_UPDATES == 0:
            print(f"EVALUATING POLICY ")

            with torch.no_grad():
                with EvalWorker() as e:
                    scores = e.eval_policy(N_EVAL_GAMES, weights)

            writer.add_scalar('Score\Average', scores.mean(), global_step=i)

            if i % 10 == 0:
                writer.add_histogram('Score\Distribution', scores, global_step=i)
                writer.add_histogram('Actions\Prob', traj['old_pi_logits'], global_step=i)

                for wk in weights:
                    writer.add_histogram(wk.replace('.', '/'), weights[wk], global_step=i)

