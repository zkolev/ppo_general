""" Reinforcement learning agent for general game"""
from GeneralGame.worker import Worker

if __name__ == "__main__":
    print("Init Worker")
    w = Worker()
    print(w.roll_out(256, gamma=0.99, lam=0.95))