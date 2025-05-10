from gridworld import *
from plot import *
from collections import defaultdict

# Q-Learning mit Optimistic Initialisation

if __name__ == '__main__':
    env = Cliff()
    num_states = env.num_states()
    num_actions = env.num_actions()
    num_episodes = 50000
    actions = ['G_UP', 'G_RIGHT', 'G_DOWN', 'G_LEFT']
    r_max=1
    q_table = np.zeros((num_states, num_actions))


    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    optimistic_value = r_max / (1 - gamma)
    for x in range(num_states):
        for y in range(num_actions):
            q_table[x][y] = optimistic_value

    visit_counts = defaultdict(int)  # count (s, a) visits
    model = {}  # (s, a): (next_state, reward)

    def plan():
        for _ in range(10):  # few value iteration steps
            for s in q_table:
                for a in env.num_actions():
                    if (s, a) in model:
                        next_s, r = model[(s, a)]
                        max_next_q = max(q_table[next_s].values())
                        q_table[s][a] = r + gamma * max_next_q
                    else:
                        # optimistic initialization
                        q_table[s][a] = r_max / (1 - gamma)


    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, np.argmax(q_table[next_state])] - q_table[state, action])
            state = next_state

    plot_q_table(env, q_table)


    '''
visit_counts = defaultdict(int)  # count (s, a) visits
model = {}  # (s, a): (next_state, reward)
q_table = defaultdict(lambda: {a: r_max / (1 - gamma) for a in actions})

# Planning from model
def plan():
    # Simple value iteration over known model
    
    '''