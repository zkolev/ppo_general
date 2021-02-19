"""
Worker specially designed for interraction
with general game environment
"""

from GeneralGame.game import GeneralGame
from GeneralGame.common import set_discrete_action_space, sample_action
from GeneralGame.dnn_model import build_simple_model
import numpy as np
# Worker



class Worker(object):
    def __init__(self):
        self.env = GeneralGame()  # init the environment
        self.all_actions = set_discrete_action_space(positions_count=14)
        self.num_actions = len(self.all_actions)

        # TODO: Currently the input size is hardcoded
        #       To derive it from the enrivonemnt
        self.model = build_simple_model((21), len(self.all_actions))

    def roll_out(self, batch, gamma, lam, is_eval=False):
        """
        :param n_steps int: Number of steps to execute on the environment
        :return: sarp
        """
        s, a, r, m = self.env.start_game()

        # Init placeholders
        S = np.zeros(shape=(batch, len(s)), dtype=np.int32)
        VS = np.zeros(shape=(batch+1, 1), dtype=np.float32)
        PiS = np.zeros(shape=(batch, self.num_actions), dtype=np.float32)

        A, R, M = tuple([np.zeros(shape=(batch, 1), dtype=np.int32)
                         for _ in range(3)])


        _resets = 0
        _ts = 0
        _ts_l = []

        for i in range(batch):

            print(f"TOTAL RESTARTS {_resets}")

            # Sample action
            pi_s, v_s = self.model(np.array([s]), training = False)
            act = self.__sample_action(s, pi_s)

            s, a, r, m = self.env.step(a=self.all_actions[int(act)])
            S[i, :], A[i, :], R[i, :], M[i, :], VS[i, :], PiS[i, :] = s, act, r, m, v_s, pi_s

            _ts += r

            if m:
                _resets += 1
                _ts_l.append(_ts)
                _ts = 0

        # If the rollout finished add the last vf
        VS[i+1, :] = v_s

        return self.__compute_gae(vs=VS, r = R, masks=M,  gamma = gamma, lam=lam)

    @staticmethod
    def __compute_gae(vs, r, gamma, lam, masks):
        """
        :param vs: array of shape [(batch + 1) ,  1]
        :param r: size: [batch, 1]
        :param masks:
        :param gamma scalar:
        :return:
        """

        B, D = r.shape
        gae = 0
        gaes = np.zeros(shape=(B, 1), dtype=np.float32)

        # R(s) + gamma * V(s+1) - V(s)
        advantages = vs[:-1, :] - gamma * vs[1:, :] * (1 - masks)

        for i, k in enumerate(zip(advantages[::-1, :], masks[::-1, :])):
            # gae(t) = delta(t) + gamma * lambda * mask * gae(t+1)
            gae = k[0] + gamma * lam * (1 - k[1]) * gae
            gaes[i, :] = gae

        return gaes[::-1, :]



    def __sample_action(self, s, policy_logits):

        """
        Validates actions againts the state
        The state array:
        ix 0: remaining rolls
        ix 1-5: dice faces
        ix 6:19: is position checked
        """

        mask = np.ones(self.num_actions)
        for i, j in enumerate(self.all_actions):

            # Checkout
            if j[0] == 0 and s[(i + 7)] == 1:
                mask[i] = False

            # No remaining rolls
            elif s[0] == 0 and j[0] == 1:
                mask[i] = False

            # If the position is checked
            else:
                continue

        return sample_action(policy_logits,mask)


