""" Reinforcement learning agent for general game"""


import time


from RL.dqn import DQN, Agent, ReplayBuffer, q_loss
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

force_cpu = True
device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')


REPLAY_BUFFER_SIZE = int(2.5e4)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = int(2.5e5)
TARGET_NETWORK_SYNC_STEPS = 5000
BATCH_SIZE = (1024)



if __name__ == '__main__':

    q_network = DQN(21, 45).to(device)
    tgt_net = DQN(21, 45).to(device)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    rl_agent = Agent(buffer=buffer)
    optimizer = Adam(q_network.parameters(),  1e-4)

    # writer = SummaryWriter(comment="General Game")

    # Data placehoders:

    n_steps = 0
    ts_steps = 0
    ts = time.time()
    best_avg_reward = None
    total_scores = []


    # Init the buffer

    while len(buffer) < REPLAY_BUFFER_SIZE:
        _ = rl_agent.play_step(net=q_network, epsilon=1.0, device=device)

    for i in range(30):

        n_steps += 1

        epsilon = max(EPSILON_END, EPSILON_START - n_steps/EPSILON_DECAY_STEPS)
        reward = rl_agent.play_step(net=q_network, epsilon=epsilon, device=device)


        if reward:
            total_scores.append(reward)
            # writer.add_scalar('reward', reward, n_steps)

            if len(total_scores) % 100 == 0:
                speed = (n_steps - ts_steps) / (time.time() - ts)
                ts_steps = n_steps
                ts = time.time()
                mean_reward = sum(total_scores[-100:]) / (len(total_scores[-100:]))
                print(f"Step {n_steps}: {len(total_scores)} full games mean score {mean_reward: .2f} "
                      f"eps {epsilon: 0.3f} speed {int(speed)} steps/s")

                # writer.add_scalar("epsilon", epsilon, n_steps)
                # writer.add_scalar("speed", speed, n_steps)
                # writer.add_scalar("mean_reawrd", mean_reward, n_steps)

        if n_steps % TARGET_NETWORK_SYNC_STEPS == 0:
            print("SYNC TARGET NETWORK...")
            tgt_net.load_state_dict(q_network.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = q_loss(batch, q_network, tgt_net, device)
        loss.backward()
        optimizer.step()


