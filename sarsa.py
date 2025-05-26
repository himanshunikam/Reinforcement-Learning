import numpy as np
from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 500000
    q_table = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.999
    epsilon = 0.001
    conv = 1e-4
    count = 0
    ep =0
    for episode in range(num_episodes):
        prev = q_table.copy()

        state = env.reset()
        done = False
        finished = False

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
        print(episode)
        if np.max(abs(q_table - prev)) < conv:
            ep = episode
            break
        else:
            count = 0

    plot_q_table(env, q_table)
