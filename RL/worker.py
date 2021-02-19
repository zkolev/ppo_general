"""
Worker specially designed for interraction
with general game environment
"""

from GeneralGame.game import GeneralGame
from RL.common import set_discrete_action_space, get_distribution_from_logits
from RL.ppo import ActCritNetwork
import torch


class Worker(object):
    def __init__(self):
        self.env = GeneralGame(print_state=False)  # init the environment
        self.all_actions = set_discrete_action_space(positions_count=len(self.env.score_board.positions))
        self.num_actions = len(self.all_actions)

        # Get the initial state of the game
        self.current_state, _, _, _ = self.env.start_game()
        self.state_size = len(self.current_state)

        self.current_state = torch.tensor([self.current_state], dtype=torch.float32)

        # Init the neural network

        self.network = ActCritNetwork(input_size=self.state_size,
                                      num_actions=self.num_actions)

    def roll_out(self, weights, batch, gamma, lam):

        """
        :param n_steps int: Number of steps to execute on the environment
        :return: sarp
        """

        # Set the new weights
        self.network.load_state_dict(weights)

        # Init placeholders for some of the main tensors
        S = torch.zeros(size=(batch, self.state_size), dtype=torch.float32)
        VS = torch.zeros(size=(batch+1, 1), dtype=torch.float32)
        AM = torch.zeros(size=(batch, self.num_actions), dtype=torch.bool) #AM = action_masks

        A, R, M, PiS = tuple([torch.zeros(size=(batch, 1), dtype=torch.int64)
                         for _ in range(4)])

        # Some meta data ph
        _resets = 0
        _ts = 0
        _ts_l = []

        # Append the initial state to the VS

        for i in range(batch):

            with torch.no_grad():
                v_s, pi_s = self.network(self.current_state)

            act, old_pi_s, _act_masks = self.__sample_action(pi_s)

            s, a, r, m = self.env.step(a=self.all_actions[int(act)])

            S[i, :], A[i, :], R[i, :], M[i, :] = self.current_state, act, r, m
            VS[i, :], PiS[i, :], AM[i, :] = v_s, old_pi_s, _act_masks

            self.current_state = torch.tensor([s], dtype=torch.float32)

            _ts += r

            if m:
                _resets += 1
                _ts_l.append(_ts)
                _ts = 0

        # If the rollout finished add the last vf
        with torch.no_grad():
            v_s, _ = self.network(self.current_state)

        VS[i+1, :] = v_s

        G, ADV = self.__compute_gae(vs=VS, r=R, masks=M,  gamma=gamma, lam=lam)


        return {
                'states': S,
                'actions': A,
                'action_masks': AM,
                'returns': G,
                'advantages': ADV,
                'old_pi_logits': PiS,
                'aux_data': [R, M, VS]
                }

    @staticmethod
    def __compute_gae(vs, r, gamma, lam, masks):
        """
        :param vs: array of shape [(batch + 1) ,  1]
        :param r: size: [batch, 1]
        :param masks:
        :param gamma:
        :return:
        """

        B, D = r.size()
        gae = 0
        returns = torch.zeros_like(r, dtype=torch.float32)

        # R(s) + gamma * V(s+1) - V(s)
        td = r + gamma * vs[1:, :] * (1 - masks) - vs[:-1, :]

        for i, k in enumerate(zip(td.flip(0), masks.flip(0))):
            # gae(t) = delta(t) + gamma * lambda * mask * gae(t+1)
            gae = k[0] + gamma * lam * (1 - k[1]) * gae
            returns[i, :] = gae

        returns = returns.flip(0)
        advantages = returns - vs[:-1, :]

        return returns, advantages



    def __sample_action(self, policy_logits):

        """
        s: torch tensor float32 (1 X D)
        policy_logits:
        returns: triple action torch int32 size [1]
                        log_prob torch float32 size [1x1]
                        mask torch boolean size [1 X num_actions]

        Validates actions againts the state
        The state array:
        ix 0: remaining rolls
        ix 1-5: dice faces
        ix 6:19: is position checked

        TODO: It can be improved

        """
        # Init the action mask:
        mask = torch.zeros_like(policy_logits, dtype=torch.bool)
        _s = self.current_state.squeeze(0)

        for i, j in enumerate(self.all_actions):

            # Checkout
            if j[0] == 0 and _s[(i + 7)] == 1:
                mask[0, i] = True

            # No remaining rolls
            elif _s[0] == 0 and j[0] == 1:
                mask[0, i] = True

            # If the position is checked
            else:
                continue

        # Init distribution and sample action:
        distr = get_distribution_from_logits(policy_logits=policy_logits,
                                             action_mask=mask)

        action = distr.sample()

        log_prob = torch.log(distr.probs).gather(1, action.unsqueeze(0))

        return action, log_prob, mask


