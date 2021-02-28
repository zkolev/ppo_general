# Import multiprocessing

import multiprocessing as mp
from multiprocessing import Process, Pipe, Manager, Value

from RL.ppo import PPO
from RL.parallel_worker import ParWorker

import time

def init_worker(constructor, **kwargs):
    worker = constructor(**kwargs)
    worker.run()



class A2C(PPO):
    def __init__(self, *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)

        s_manager = Manager()

        self.s_weights = s_manager.dict()

        self.s_start_rollouts = mp.Value('b', False)
        self.s_channels = [] # MP pipe

        self.__update_shared_weights()



    def run(self, n_workers, updates, epochs, steps, gamma, lam):

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