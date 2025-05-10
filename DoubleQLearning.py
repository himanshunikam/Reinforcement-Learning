from gridworld import *
from plot import *

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 100000
    q_1 = np.zeros((num_states, num_actions))
    q_2 = np.zeros((num_states, num_actions))
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:                  #explore
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_1[state]+ q_2[state])

            next_state, reward, done = env.step(action)
            if np.random.rand(1) < 0.5:
                q_1[state, action] = q_1[state, action] + alpha * (
                            reward + gamma * q_2[next_state, np.argmax(q_1[next_state])] - q_1[
                        state, action])
            else:
                q_2[state, action] = q_2[state, action] + alpha * (
                        reward + gamma * q_1[next_state, np.argmax(q_2[next_state])] - q_2[
                    state, action])
            state = next_state

    plot_q_table(env, q_1)
    plot_q_table(env, q_2)