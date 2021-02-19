""" Reinforcement learning agent for general game"""
from RL.worker import Worker
from RL.ppo import PPO

if __name__ == "__main__":
    print("Init Worker")

    w = Worker()
    ppo = PPO(21, 45, minibatch_size=1024)

    # Init shared variables:
    weights = ppo.network.state_dict()

    for i in range(10):
        print(f'Iteration {i} ... ')
        traj = w.roll_out(weights, 4096, 0.99, 0.95)
        weights = ppo.update(traj=[traj])
