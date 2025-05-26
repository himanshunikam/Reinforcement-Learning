import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque


class NStepSARSA:
    def __init__(self, env, n_steps=3, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=500):
        self.env = env
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes

        # Initialize Q-table with zeros
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

        # For tracking performance
        self.returns = []

    def epsilon_greedy_policy(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            return np.argmax(self.Q[state])  # Greedy action

    def n_step_sarsa_update(self, states, actions, rewards, done):
        """Perform n-step SARSA update"""
        # Calculate the target state-action value
        n = len(states) - 1  # Number of transitions (not including final state)

        if n < self.n_steps and not done:
            # Not enough steps collected yet and episode not done
            return

        update_state = states[0]
        update_action = actions[0]

        if n < self.n_steps:  # End of episode with fewer than n steps
            # Calculate n-step return (up to termination)
            G = 0
            for i in range(n):
                G += (self.gamma ** i) * rewards[i]
        else:  # We have at least n steps
            # Calculate n-step return with bootstrapping
            G = 0
            for i in range(self.n_steps):
                G += (self.gamma ** i) * rewards[i]

            # Add the bootstrapped value
            next_state = states[self.n_steps]
            next_action = actions[self.n_steps]
            G += (self.gamma ** self.n_steps) * self.Q[next_state, next_action]

        # Update Q-value
        self.Q[update_state, update_action] += self.alpha * (G - self.Q[update_state, update_action])

    def train(self):
        """Train the agent using n-step SARSA"""
        for episode in range(self.episodes):
            # Reset environment and get initial state
            state = self.env.reset()
            action = self.epsilon_greedy_policy(state)

            # Initialize lists to store states, actions, rewards
            states = deque([state])
            actions = deque([action])
            rewards = deque()

            done = False
            episode_return = 0

            while not done:
                # Take action and observe next state, reward
                next_state, reward, done, _ = self.env.step(action)
                episode_return += reward

                # Choose next action using epsilon-greedy policy
                next_action = self.epsilon_greedy_policy(next_state)

                # Store transition
                states.append(next_state)
                actions.append(next_action)
                rewards.append(reward)

                # Apply n-step SARSA update if we have enough transitions
                if len(states) > self.n_steps:
                    self.n_step_sarsa_update(list(states), list(actions), list(rewards), done)
                    # Remove oldest state, action, reward
                    states.popleft()
                    actions.popleft()
                    rewards.popleft()

                # Update current state and action
                state = next_state
                action = next_action

            # Process remaining states at end of episode
            while len(states) > 1:
                self.n_step_sarsa_update(list(states), list(actions), list(rewards), done)
                states.popleft()
                actions.popleft()
                if rewards:
                    rewards.popleft()

            # Store episode return
            self.returns.append(episode_return)

            # Optional: Print progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Average Return: {np.mean(self.returns[-100:]):.2f}")

    def plot_returns(self):
        """Plot the returns over episodes"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.returns)
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'{self.n_steps}-Step SARSA Learning Curve')
        plt.grid(True)
        plt.show()


# Example usage on a simple environment
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')

    # Try different n_steps values
    n_steps_list = [1, 3, 5]
    for n_steps in n_steps_list:
        print(f"\nTraining with {n_steps}-step SARSA")
        agent = NStepSARSA(env, n_steps=n_steps, episodes=1000)
        agent.train()
        agent.plot_returns()