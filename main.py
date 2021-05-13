""" Reinforcement learning agent for general game"""

from RL.a2c import A2C
import os
import time
from multiprocessing import set_start_method
from torch.utils.tensorboard import SummaryWriter


INPUT_SIZE = 21 # State representation
OUTPUT_SIZE = 45 # Num actions
PERSIST_STATE = True
PAYLOAD_ROOT_DIR = None
TENSORBOARD_LOC = None
MODEL_NAME = 'Mdl1'

# Restore 
RESTORE = False
MODEL_CHECKPOINT = None 
CHECKPOINT_EVERY_N_UPDATES = 25

LEARNING_RATE = 1e-3
MINIBATCH_SIZE = int(1 * 1024)
ROLLOUT_SIZE = (16 * 1024)
EPOCHS_PER_UPDATE = 5
GAMMA = 0.99
LAMBDA = 0.95
NUMBER_OF_UPDATES = 3001

EVAL_EVERY_N_UPDATES = 25
N_GAMES_FOR_EVAL = 200

HISTOGRAM_EVERY_N_UPDATES = 75

PPO_CLIP_NORM = 0.2

LOSS_POLICY_WEIGHT = 1
LOSS_VALUE_WEIGHT = 0.20
LOSS_ENTROPY_WEIGHT = 0.4


N_WORKERS = 4
FORCE_CPU = True

root_dir = os.path.abspath(__file__ + '/..')

if PAYLOAD_ROOT_DIR:
        root_dir = PAYLOAD_ROOT_DIR

# Create Derive the name of the model
if MODEL_NAME:
    mdl_name = str(MODEL_NAME)

else:
    # The timestamp
    mdl_name = str(int(time.time()))

fs_loc = {}
for subdir in ['checkpoints', 'tensorboard']:
    _loc = os.path.join(root_dir,'training_payload', subdir, mdl_name)
    os.makedirs(_loc, exist_ok=True)
    fs_loc[subdir] = _loc


if __name__ == "__main__":
    set_start_method('spawn')
    print(fs_loc)

    _global_step = 0
    _start_iter = 0

    # TODO: To refine the summary writer for hparams
    # with SummaryWriter(f"{fs_loc['tensorboard']}") as w:
    #     w.add_hparams(
    #         {'lr': LEARNING_RATE,
    #          'minibatch_size': MINIBATCH_SIZE,
    #          'trajectory_len': ROLLOUT_SIZE,
    #          'epochs_per_update': EPOCHS_PER_UPDATE,
    #          'PPO_CLIP_NORM': PPO_CLIP_NORM,
    #          'LOSS_POLICY_WEIGHT': LOSS_POLICY_WEIGHT,
    #          'LOSS_VALUE_WEIGHT': LOSS_VALUE_WEIGHT,
    #          'LOSS_ENTROPY_WEIGHT': LOSS_ENTROPY_WEIGHT},
    #         {'hparam/accuracy': 1}
    #     )

    # Init step and epoch writer. Since each update execute multiple epochs
    # these writers are logging information on different granularity 

    # epoch_writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'epoch_writer'}", purge_step=_global_step)
    # update_writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'update_writer'}", purge_step=_start_iter)

    a2c = A2C(input_size=INPUT_SIZE,
              num_actions=OUTPUT_SIZE,
              lr=LEARNING_RATE,
              minibatch_size=MINIBATCH_SIZE,
              clip=PPO_CLIP_NORM,
              w_policy=LOSS_POLICY_WEIGHT,
              w_vf=LOSS_VALUE_WEIGHT,
              w_entropy=LOSS_ENTROPY_WEIGHT,
              fs_loc=fs_loc,
              model_name=MODEL_NAME)

    # Run a2c
    a2c.run(n_workers=N_WORKERS,
            updates=NUMBER_OF_UPDATES,
            epochs=EPOCHS_PER_UPDATE,
            steps=ROLLOUT_SIZE,
            gamma=GAMMA,
            lam=LAMBDA,
            eval_steps=5,
            eval_iters=125)
