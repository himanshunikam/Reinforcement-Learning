import numpy as np
from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 500
    rows, cols = env.shape()

    q_table = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9
    counter =0
    n = 5

    def epsilon_greedy(state):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        else:
            return np.argmax(q_table[state])

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        T = np.inf
        t = 0
        tau = 0
        rewards = []
        states = []
        actions = []
        action = epsilon_greedy(state)

        while (T-1) > tau:
            if t<T:

                next_state, reward, done = env.step(action)
                counter += 1
                rewards.append(reward)
                states.append(state)
                if done:
                    T = t+1
                else :
                    next_action = epsilon_greedy(next_state)
                    actions.append(next_action)

            tau = t- n +1

            if tau >= 0:
                G = np.sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])

                if tau + n < T:
                    G += gamma ** n * q_table[states[tau + n][0] * cols + states[tau + n][1], actions[tau + n]]

                q_table[states[tau][0] * cols + states[tau][1], actions[tau]] += alpha * (G - q_table[states[tau][0] * cols + states[tau][1], actions[tau]])

            t += 1
            state = next_state
            action = next_action

    plot_q_table(env, q_table)