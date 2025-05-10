import numpy as np
from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 500

    q_table = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9

    def epsilon_greedy(state):
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])