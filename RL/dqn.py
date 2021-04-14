__info__ = """ 
Implementation of basic DQN for General Game environment 
following the book "Deep Reinforcement Learning Hands-On" 
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
from collections import deque, namedtuple
from GeneralGame.game import GeneralGame
from GeneralGame.utils import get_valid_actions


# Experience will be collected in named tuples
Record = namedtuple('experience', ['state', 'action', 'reward', 'done', 'next_state'])


def q_loss(traj, net, t_net, device, gamma=0.99):
    states, actions, rewards, dones, next_states = \
        [i.to(device) for i in traj]

    states_av = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_av = t_net(next_states).max(1)[0]
    next_av[dones] = 0.0
    next_av = next_av.detach()
    e_av = next_av * gamma + rewards

    return nn.MSELoss()(states_av, e_av)


class Agent(object):
    def __init__(self, buffer):
        self.env = GeneralGame()
        self.xp_buffer = buffer
        self.__reset()

    def __reset(self):
        self.state, _, _, _ = self.env.start_game()
        self.total_reward = 0.0

    def play_step(self, net, device, epsilon=0.5):
        episode_reward = None

        # Get valid action indices
        valid_actions = torch.tensor(get_valid_actions(self.state), dtype=torch.int64, device=device)

        if torch.rand(1) > epsilon:
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()

        else:
            with torch.no_grad():
                state_vec = torch.tensor([self.state], dtype=torch.float32, device=device) # (1x21)

                q_value_vec = net(state_vec) # (1 x num_a)
                _, ix = torch.max(torch.index_select(q_value_vec, 1, valid_actions), dim=1)
                action = valid_actions[ix].item()

        new_state, action, reward, done = self.env.step(action)
        self.total_reward += reward

        # Create record and add it to the queue
        record = Record(self.state, action, reward, done, new_state)
        self.xp_buffer.update(record)

        self.state = new_state

        if done:
            episode_reward = self.total_reward
            self.__reset()

        return episode_reward


class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        self.in_layer = torch.nn.Linear(input_size, 10)
        self.l_1 = torch.nn.Linear(10, 10)

        self.pl_1 = torch.nn.Linear(10, 10)
        self.pl_out = torch.nn.Linear(10, num_actions)

    def forward(self, input_data):
        x = fn.relu(self.in_layer(input_data))
        x = fn.relu(self.l_1(x))
        pl = fn.relu(self.pl_1(x))

        return self.pl_out(pl)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        _current_size = min([self.size, len(self.buffer)])
        indeces = torch.randint(high=_current_size, size=(batch_size, ))

        states, actions, rewards, dones, next_s = \
            zip(*[self.buffer[i] for i in indeces])

        return  torch.tensor(states, dtype=torch.float32),\
                torch.tensor(actions, dtype=torch.int64),\
                torch.tensor(rewards, dtype=torch.float32),\
                torch.tensor(dones, dtype=torch.bool), \
                torch.tensor(next_s, dtype=torch.float32)

    def update(self, data):
        self.buffer.append(data)

