# Import multiprocessing

import multiprocessing as mp
from multiprocessing import Process, Manager, Value

from RL.ppo import PPO
from RL.parallel_worker import ParWorker
from RL.worker import  EvalWorker
import torch
from torch.utils.tensorboard import SummaryWriter

import os

import time

def init_worker(constructor, **kwargs):
    worker = constructor(**kwargs)
    worker.run()



class A2C(PPO):
    def __init__(self, model_name, fs_loc, last_update=0, checkpoint=None,  *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)

        s_manager = Manager()

        self.s_weights = s_manager.dict()

        self.s_start_rollouts = mp.Value('b', False)
        self.s_channels = [] # MP pipe

        self.__update_shared_weights()
        self.model_name = model_name

        # Filesystem location variable
        self.fs_loc = fs_loc
        self.last_update = last_update

        # Restore 

        if checkpoint:
            try:
                checkpoint = torch.load(os.path.join(fs_loc['checkpoints'], checkpoint))
                self.last_update = checkpoint['update_steps']
                self.last_epoch = checkpoint['epoch_stesp']
                self.restore_from_checkpoint(checkpoint['model'], checkpoint['optimizer'])
                
            except:
                print('The provided checkpoint is not valid! Init the model from scratch')
        else:
            print('Init the model from scratch')
        
        


        # Create tensorboard writer
        self.writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'epoch_writer'}", purge_step=self.last_epoch)
        self.update_writer = SummaryWriter(f"{fs_loc['tensorboard']}\{'update_writer'}", purge_step=self.last_update)



    def run(self, n_workers, updates, epochs, steps, gamma, lam, eval_steps, eval_iters ):
        
        ts_start = time.time()

        workers, channels, rollouts_done, inits_done = \
            self.__start_workers(n_workers,steps, gamma, lam)

        # Wait for workers init

        while not all(inits_done):

            print('Waiting for workers initialization ... ')
            time.sleep(1)

        else:
            print('Initialization done. Start training.')


        for updt in range(self.last_update, updates):
            
            ts_update_start = time.time()
 
            # restart the rollouts
            # TODO: To track the rollout time of each worker 

            self.__restart_rollouts(inits_done)

            while not all(rollouts_done):
                time.sleep(1)

            else:
                traj = [conn.recv() for conn in channels]
            
            ts_rollouts = time.time() 

            # Update the model
            new_weights = self.update(traj, epochs)

            ts_gradient_update = time.time()

            # Calculate time
            update_time = ts_gradient_update - ts_update_start
            rollout_time = ts_rollouts - ts_update_start
            grad_update = ts_gradient_update - ts_rollouts

            print(f"Update {updt}: Time {update_time:.1f}s (rendering: {rollout_time:.1f}, gradinet update: {grad_update:.1f})")

            # Update eval
            if updt % eval_steps == 0:

                with torch.no_grad():
                    with EvalWorker() as e:
                        scores = e.eval_policy(eval_iters, new_weights)
                        avg_score = scores.mean()


                self.update_writer.add_scalar('Scores\Average', scores.mean(), global_step = updt)

                # Save model:
                _fname = f"{int(time.time())}_{self.model_name}_Update_{updt}_avg_score_{int(avg_score)}.pth"
                chkpt_name = os.path.join(self.fs_loc['checkpoints'], _fname)

                checkpoint = {
                    'update_steps': updt,
                    'epoch_stesp': self.last_epoch,
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                
                torch.save(checkpoint, chkpt_name)
                print(f'Save checkpoint to {chkpt_name}')



            # Update histograms:
            if updt % 100 == 0:
                self.update_writer.add_histogram('Scores\Distribution', scores, global_step=updt)

                for wk in new_weights:
                    self.update_writer.add_histogram(wk.replace('.', '/'), new_weights[wk], global_step=updt)

        self.__terminate_workers(workers)

    # Init shared objects
    def __start_workers(self, n_workers, n_steps, gamma, lam, **kwargs):

        workers, channels, rollouts_done, inits_done = [], [], [], []

        for _i in range(n_workers):

            # Set worker params
            par_chnl, child_chnl = mp.Pipe()
            worker_name = f"Worker_{_i}"
            rollout_done = Value('b', False)
            init_done = Value('b', False)

            p = Process(target=init_worker,
                             args=(ParWorker,),
                             kwargs={"w_name": worker_name,
                                     "channel": child_chnl,
                                     "start": self.s_start_rollouts,
                                     "weights": self.s_weights,
                                     "n_steps": n_steps,
                                     "rollout_done": rollout_done,
                                     "gamma": gamma,
                                     "lam": lam,
                                     "init_done": init_done})

            p.start()

            workers.append((worker_name, p))
            channels.append(par_chnl)
            rollouts_done.append(rollout_done)
            inits_done.append(init_done)


        return workers, channels, rollouts_done, inits_done


    def __restart_rollouts(self, inits_done):
        self.s_start_rollouts.value = True
        self.__update_shared_weights()
        for i in inits_done:
            i.value = True

    def __update_shared_weights(self):
        _state_dict = self.network.state_dict()
        # Update the shared weights
        for k in _state_dict:
            self.s_weights[k] = _state_dict[k].cpu()

    @staticmethod
    def __terminate_workers(workers):
        for w_name, w in workers:
            w.terminate()