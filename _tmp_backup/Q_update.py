""" Reinforcement learning agent for general game"""

from collections import defaultdict
from GeneralGame.game import GeneralGame
from RL.common import set_discrete_action_space
import random


GAMMA = 0.9
ALPHA = 0.2
NUPDATES = int(1e5)
EVAL_GAMES = 1


def state_to_ix(state):
    'concatenate elements of list of integers to string'
    return ''.join([str(i) for i in state])


def get_valid_actions(s):
    actions = [i for i in range(6) if s[(i + 7)] != 1]

    if s[0] > 0:
        actions += list(range(6, 37))

    return actions


class Agent(object):
    def __init__(self):
        self.q_tab = defaultdict(float)
        self.env = GeneralGame()
        self.state, _, _, _ = self.env.start_game()



    def sample_env(self):
        action = random.choice(self.env.get_valid_actions_indeces())
        old_state = self.state

        new_state, reward, _, done = self.env.step(action, restart_when_done=True)

        self.state = new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):

        _state = state_to_ix(state)
        valid_actions = get_valid_actions(state)

        best_value, best_action = None, None

        for action in range(len(self.env.discrete_action_space)):
            if action not in valid_actions:
                continue

            action_value = self.q_tab[(_state, action)]

            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):

        _s = state_to_ix(s)
        _next_s = state_to_ix(next_s)

        best_v, _ = self.best_value_and_action(next_s)

        new_val = r + GAMMA * best_v
        old_val = self.q_tab[(_s, a)]



        self.q_tab[(_s, a)] = old_val * (1 - ALPHA) + new_val * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        state, _, _, _ = env.start_game()
        is_done = False
        while not is_done:
            _, action = self.best_value_and_action(state)
            s, a, r, is_done = env.step(action)
            total_reward += r
            state = s

        return total_reward


if __name__ == "__main__":
    agnt = Agent()
    test_env = GeneralGame()

    for i in range(NUPDATES):
        s, a, r, next_s = agnt.sample_env()

        agnt.value_update(s, a, r, next_s)

        if i % int(2.5e4) == 0:

            reward = 0.0
            # for _ in range(EVAL_GAMES):
                # reward += agnt.play_episode(test_env)
            print(i, reward / EVAL_GAMES, len(agnt.q_tab), len(set([k[0] for k in agnt.q_tab])))


