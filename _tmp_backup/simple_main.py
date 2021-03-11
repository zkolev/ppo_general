""" Reinforcement learning agent for general game"""


from GeneralGame.game import  GeneralGame
from GeneralGame.utils import   get_valid_actions
from random import choice
import time

if __name__ == '__main__':
    env = GeneralGame()
    x = time.time()
    for i in range(1000):
        s, _, _, done = env.start_game()

        total_reward = 0.0
        i = 0

        while not done:
            valid_actions = get_valid_actions(s)
            a = choice(valid_actions)
            s, _, r, done = env.step(a)
            total_reward += r
            i += 1


        print(f"Total reward of {total_reward} after {i} steps")
    print(f"End after {time.time()-x} seconds")
