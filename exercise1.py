"""
Small demo to illustrate how the plot function and the gridworld environment work
"""

import numpy as np
from gridworld import *
from plot import *


# Value Iteration
if __name__ == "__main__":
    # create environment
    env = Cliff()
    # create nonsense V-values and nonsense policy
    v_table = np.random.random((env.num_states()))
    q_table = np.random.random((env.num_states(), env.num_actions()))
    policy = np.random.random((env.num_states(), env.num_actions()))
    # either plot V-values and Q-values without the policy...
    # plot_v_table(env, v_table)
    # plot_q_table(env, q_table)
    # ...or with the policy
    for i in range(1000):
        updated_v_table = np.copy(v_table)
        updated_q_table = np.copy(q_table)
        for state in range(env.num_states()):
            q_values = np.zeros(env.num_actions())      # [0,0,0,0]
            for action in range(env.num_actions()):
                next_state, reward, done = env.step_dp(state, action)
                if done:
                    q_values[action] = reward
                else:
                    q_values[action] = reward+ v_table[next_state]*1     #gamma = 1
                updated_v_table[state] = max(q_values)
                updated_q_table[state] = q_values
        policy = np.eye(env.num_actions())[np.argmax(updated_q_table, axis=1)]

        v_table = updated_v_table
        q_table = updated_q_table
    plot_v_table(env, v_table, policy)
    plot_q_table(env, q_table, policy)
