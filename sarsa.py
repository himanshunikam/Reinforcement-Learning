import numpy as np
from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 10000
    q_table = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.999
    epsilon = 0.001
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])

        while not done:
            next_state, reward, done = env.step(action)

            if np.random.rand() < epsilon:
                next_action = np.random.randint(num_actions)
            else:
                next_action = np.argmax(q_table[next_state])

            q_table[state, action] = q_table[state, action] + alpha*(reward+gamma*q_table[next_state, next_action]- q_table[state, action])
            state = next_state
            action = next_action
    plot_q_table(env, q_table)
