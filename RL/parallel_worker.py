# Implements the worker routine with shared memory:
from RL.worker import RolloutsWorker
import time

class ParWorker(RolloutsWorker):
    def __init__(self, w_name, channel, start, weights, n_steps,
                 init_done, rollout_done, gamma, lam,  **kwargs):
        super(ParWorker, self).__init__(**kwargs)

        self.w_name = w_name

        # Shared memory
        self.s_channel = channel # MP pipe
        self.s_start = start
        self.s_weights = weights
        self.s_batch = n_steps
        self.s_rollout_done = rollout_done
        self.s_init_done = init_done

        # GAE hp
        self.s_gamma = gamma
        self.s_lam = lam

        self.s_init_done.value = True

        print(f"Worker {self.w_name} has been initialized !")

    def run(self):
        # Open endless loop
        while True:
            if self.s_start.value and self.s_init_done.value:

                traj = self.roll_out(weights=self.s_weights,
                                     batch=self.s_batch,
                                     gamma=self.s_gamma,
                                     lam=self.s_lam)

                self.s_init_done.value = False
                self.s_channel.send(traj)

            else:
                # Wait until update finishes
                time.sleep(1)