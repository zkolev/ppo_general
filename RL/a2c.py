# Import multiprocessing

import multiprocessing as mp
from multiprocessing import Process, Pipe, Manager, Value

from RL.ppo import PPO
from RL.parallel_worker import ParWorker
from RL.worker import  EvalWorker
import torch
import os

import time

def init_worker(constructor, **kwargs):
    worker = constructor(**kwargs)
    worker.run()



class A2C(PPO):
    def __init__(self, model_name,  *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)

        s_manager = Manager()

        self.s_weights = s_manager.dict()

        self.s_start_rollouts = mp.Value('b', False)
        self.s_channels = [] # MP pipe

        self.__update_shared_weights()
        self.model_name = model_name


    def run(self, n_workers, updates, epochs, steps, gamma, lam,
            step_writer, eval_steps, eval_iters, fs_loc ):

        workers, channels, rollouts_done, inits_done = \
            self.__start_workers(n_workers,steps, gamma, lam)

        # Wait for workers init

        while not all(inits_done):

            print('Waiting for workers initialization ... ')
            time.sleep(1)

        else:
            print('All workers have been initialized. Start generating trajectories ')

        for epoch in range(updates):

            # restart the rollouts
            self.__restart_rollouts(inits_done)

            while not all(rollouts_done):
                # Eval new policy while the workers
                # are generating experience
                time.sleep(1)

            else:
                traj = [conn.recv() for conn in channels]

                # Update the model
                new_weights = self.update(traj, epochs)


                # Update eval
                if epoch % eval_steps == 0:
                    print('Evaluating policy ... ')
                    with torch.no_grad():
                        with EvalWorker() as e:
                            scores = e.eval_policy(eval_iters, new_weights)

                    step_writer.add_scalar('Scores\Average', scores.mean(), global_step = epoch)

                # Update histograms:
                if epoch % 50 == 0:
                    step_writer.add_histogram('Scores\Distribution', scores, global_step=epoch)

                    for wk in new_weights:
                        step_writer.add_histogram(wk.replace('.', '/'), new_weights[wk], global_step=epoch)

                if epoch % 50 == 0:
                    _fname = f"{int(time.time())}_{self.model_name}_Update_{epoch}.pth"
                    chkpt = os.path.join(fs_loc['checkpoints'], _fname)
                    print(f'Checkpoint to {chkpt} ... ')

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
            self.s_weights[k] = _state_dict[k]

    @staticmethod
    def __terminate_workers(workers):
        for w_name, w in workers:
            w.terminate()