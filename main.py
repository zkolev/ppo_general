""" Reinforcement learning agent for general game"""
from RL.worker import RolloutsWorker, EvalWorker
from RL.ppo import PPO
from RL.common import eval_policy
from torch.utils.tensorboard import SummaryWriter
from GeneralGame.game import GeneralGame
import time
import os 

import torch



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

LOSS_POLICY_WEIGHT = 1
LOSS_VALUE_WEIGHT = 0.20
LOSS_ENTROPY_WEIGHT = 0.05

if __name__ == "__main__":
    

    # Create root dir relative to the place 
    # TODO: To export the routine as function/class 

    root_dir = os.path.abspath(__file__ + '/..')

    if PAYLOAD_ROOT_DIR:
        root_dir = payload_root_dir
    
    # Create Derive the name of the model 
    if MODEL_NAME:
        mdl_name = str(MODEL_NAME)
    else:
        # The timestamp 
        mdl_name = str(int(time.time()))

    fs_loc = {}
    for subdir in ['checkpoints', 'tensorboard']:
        _loc = os.path.join(root_dir,'training_payload', subdir, mdl_name)
        os.makedirs(_loc,exist_ok=True)
        fs_loc[subdir] = _loc

    
    
    # Init meta parameters:
    _start_iter = 0
    _global_step = 0

    # Load Latest chkpt 
    # TODO: To implement better restoring routine
    #        The curren routine relly on the naming  
    #        convention of the chekpoint 

    if RESTORE:
        # If specified restore from chkpt 
        # Else restore from latest 
        if MODEL_CHECKPOINT is None:
            try:
                torch.load(os.path.join(fs_loc['checkpoints'], MODEL_CHECKPOINT))
            except: 
                sorted(os.listdir(fs_loc['checkpoints']), reverse=True)


    # INIT OBJECTS
    game = GeneralGame()
    
    batch_writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'batch_writer'}", purge_step=_global_step)
    update_writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'update_writer'}", purge_step=_start_iter) #TODO: Programatically assighn root dir
    w = RolloutsWorker()    

    # Policy
    ppo = PPO(INPUT_SIZE, OUTPUT_SIZE,
              minibatch_size=MINIBATCH_SIZE,
              lr=LEARNING_RATE,
              writer=batch_writer,
              w_policy=LOSS_POLICY_WEIGHT,
              w_vf=LOSS_VALUE_WEIGHT,
              w_entropy=LOSS_ENTROPY_WEIGHT,
              global_step=_global_step
              )



    
    weights = ppo.network.state_dict()


    # Main loop
    for i in range(_start_iter, _start_iter + NUMBER_OF_UPDATES):

        print(f'POLICY UPDATE  {i} ... ')
        traj = w.roll_out(weights, ROLLOUT_SIZE, GAMMA, LAMBDA)

        weights = ppo.update(traj=[traj],
                             epochs=EPOCHS_PER_UPDATE)

        if i % EVAL_EVERY_N_UPDATES == 0:
            print(f"EVALUATING POLICY ")

            with torch.no_grad():
                with EvalWorker() as e:
                    scores = e.eval_policy(N_GAMES_FOR_EVAL, weights)

            update_writer.add_scalar('Score\Average', scores.mean(), global_step=i)

            if i % HISTOGRAM_EVERY_N_UPDATES == 0:
                update_writer.add_histogram('Score\Distribution', scores, global_step=i)

                # Add the weights 
                for wk in weights:
                    update_writer.add_histogram(wk.replace('.', '/'), weights[wk], global_step=i)


        # Save checkpoint 
        if i % CHECKPOINT_EVERY_N_UPDATES == 0:
            _fname = f"{int(time.time())}_{MODEL_NAME}_Update_{i}.pth" 
            chkpt = os.path.join(fs_loc['checkpoints'], _fname)
            print(f'Checkpoint to {chkpt} ... ')

            torch.save({'update': i, 
                        'global_step': ppo.global_step, 
                        'model_state_dict': weights, 
                        'optimizer_state_dict': ppo.optimizer.state_dict()}, 
                        chkpt)
