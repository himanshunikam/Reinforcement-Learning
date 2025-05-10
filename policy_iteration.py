
import numpy as np
from gridworld import *
from plot import *

def policy_iteration(env):

    num_states = env.num_states()
    num_actions = env.num_actions()

    policy = np.random.random((env.num_states(), env.num_actions()))
    v_table = np.random.random(num_states)
    while True:
        v_table = policy_evaluation(env, policy, v_table,gamma=0.8, theta=0.001)

        policy, policy_stable, new_policy = policy_improvement(env, policy,v_table, gamma=0.8)
        if policy_stable:
            break
    return policy, v_table, new_policy
def policy_evaluation(env, policy, v_table, gamma=0.8, theta=0.001):
    num_states = env.num_states()
    for i in range (1000):
        delta = 0
        for state in range(num_states):
            q_value = 0
            for action, action_prob in enumerate(policy[state]):
                next_state, reward, done = env.step_dp(state, action)
                if done:
                    q_value += action_prob * reward
                else:
                    q_value += action_prob *(reward+gamma*v_table[next_state])
            delta=max(delta, np.abs(v_table[state] - q_value))
            v_table[state] = q_value
        if delta < theta:
            break

    return v_table


def policy_improvement(env, policy,v_table, gamma=0.8):
    policy_stable = True
    num_states = env.num_states()
    num_actions = env.num_actions()
    new_policy = np.random.random((env.num_states(), env.num_actions()))

    for state in range(num_states):
        for action in range(num_actions):
            next_state, reward, done = env.step_dp(state, action)
            if done:
                new_policy[state, action] = reward
            else:
                new_policy[state, action] = reward+gamma*v_table[next_state]
        best_action = np.argmax(new_policy[state])
        policy_stable = policy_stable and (np.argmax(new_policy[state]) == best_action)
        policy[state] =np.eye(num_actions)[best_action]
    return  policy, policy_stable, new_policy

if __name__ == "__main__":
    # create environment
    env = ExerciseWorld()
    policy, v_table, new_policy = policy_iteration(env)
    plot_v_table(env, v_table, policy)
    plot_q_table(env, new_policy)