from collections import defaultdict

import numpy as np

from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()

    num_episodes = 10000
    gamma = 0.9

    policy = np.random.random((num_states, num_actions))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = np.zeros(num_states)

    for i in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            if np.random.rand() < 0.1:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(policy[state])

            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        visited = set()
        G=0

        for t in reversed(range(len(episode))):
            state_t, action, reward_t= episode[t]
            G = gamma * G + reward_t

            if state_t not in visited:
                visited.add(state)
                returns_sum[state_t] += G
                returns_count[state_t] += 1
                V[state_t] += returns_sum[state_t] / returns_count[state_t]

    plot_v_table(env, V)