# Import multiprocessing

from torch.multiprocessing import Process, Pipe, Manager, Value

from RL.ppo import PPO
from RL.parallel_worker import ParWorker
from RL.worker import EvalWorker
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

        self.s_start_rollouts = Value('b', False)
        self.s_channels = [] # MP pipe

        self.__update_shared_weights()
        self.model_name = model_name


    def run(self,
            n_workers, updates, epochs, steps, gamma, lam,
            step_writer, eval_steps, eval_iters, fs_loc):


        workers, channels, rollouts_done, inits_done = \
            self.__start_workers(n_workers, steps, gamma, lam)

        # Wait for workers init
        while not all(inits_done):

            print('Waiting for workers initialization ... ')
            time.sleep(1)

        else:
            print('All workers have been initialized. ')

        # ENTER THE MAIN TRAINING LOOP:
        for update in range(updates):

            # Track time
            update_start_time = time.time()

            # Tell the environments to start generating
            # data (trajectories) for N steps
            self.__restart_rollouts(inits_done)

            while not all(rollouts_done):
                time.sleep(0.5)

            else:
                # Collect the trajectories from the workers
                traj = [conn.recv() for conn in channels]
                rendering_time = time.time()

                # Perform the gradient update
                new_weights = self.update(traj, epochs)
                gradinet_update_time = time.time()


                print(f'Step {update}: Rendering time: {rendering_time - update_start_time :.1f} sec ; '
                      f'Gradient update time {gradinet_update_time - rendering_time:.1f} sec;')

                # Evaluate the new policy
                if update % eval_steps == 0:
                    with torch.no_grad():
                        with EvalWorker() as e:
                            scores = e.eval_policy(eval_iters, new_weights)

                    step_writer.add_scalar('Scores\Average', scores.mean(), global_step=update)
                    eval_time = time.time()
                    total_eval_time_sec = eval_time - gradinet_update_time

                    print(f"Model Eval time {total_eval_time_sec:.2f} sec;"
                          f" {total_eval_time_sec/eval_steps:.2f} sec per game")

                # Track the distribution of weights over time:
                if update % 50 == 0:
                    step_writer.add_histogram('Scores\Distribution', scores, global_step=update)

                    for wk in new_weights:
                        step_writer.add_histogram(wk.replace('.', '/'), new_weights[wk], global_step=update)

                    # TODO: Save model weigths to disk
                    _fname = f"{int(time.time())}_{self.model_name}_Update_{update}.pth"
                    chkpt = os.path.join(fs_loc['checkpoints'], _fname)
                    print(f'Checkpoint to {chkpt} ... ')


        self.__terminate_workers(workers)

    # Init shared objects
    def __start_workers(self, n_workers, n_steps, gamma, lam, **kwargs):

        workers, channels, rollouts_done, inits_done = [], [], [], []

        for _i in range(n_workers):

            # Set worker params
            par_chnl, child_chnl = Pipe()
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

    def __update_shared_weights(self, dev='cpu'):
        """ Expose new weights to workers after each update """
        self.s_weights = {k: v.to(torch.device(dev))
                          for k, v in self.network.state_dict().items()}

    @staticmethod
    def __terminate_workers(workers):
        for w_name, w in workers:
            w.terminate()