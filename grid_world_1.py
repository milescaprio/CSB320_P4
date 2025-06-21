import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pygame


default_params = {
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "episodes": 5000,
}

# Save initial Q-table for MSS comparison


def train_agent(initial_Q, Q, env, params=default_params):
    alpha = params["alpha"]
    gamma = params["gamma"]
    epsilon = params["epsilon"]
    epsilon_decay = params["epsilon_decay"]
    epsilon_min = params["epsilon_min"]
    episodes = params["episodes"]

    # Training loop
    reward_tracking = []
    steps_per_episode = []  # Store the number of steps per episode

    for episode in range(episodes):
        state_info = env.reset()
        state = (
            state_info[0] if isinstance(state_info, tuple) else state_info
        )  # Handle different Gym versions
        total_reward = 0
        done = False
        steps = 0  # Track steps per episode

        while not done:
            steps += 1  # Increment step count

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore (random action)
            else:
                action = np.argmax(Q[state])  # Exploit (best action)

            # Take action, get new state & reward
            next_state_info = env.step(action)
            next_state = (
                next_state_info[0]
                if isinstance(next_state_info, tuple)
                else next_state_info
            )
            reward = next_state_info[1]
            done = next_state_info[2]

            # Q-value update using Bellman Equation
            best_next_action = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next_action - Q[state, action]
            )

            state = next_state  # Move to next state
            total_reward += reward

        reward_tracking.append(total_reward)
        steps_per_episode.append(steps)  # Store the number of steps taken

        # Decay epsilon to shift from exploration to exploitation
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Compute Mean Sum of Squares (MSS) between initial and final Q-table
    mss = np.mean((Q - initial_Q) ** 2)
    return Q, reward_tracking, steps_per_episode, mss


def print_and_visualize_results(initial_Q, Q, steps_per_episode, mss, text=""):
    # Print Q-table comparison
    print(f"\n{text} Initial Q-Table (Before Training):")
    print(initial_Q)
    print(f"\n{text} Final Q-Table (After Training):")
    print(Q)

    # Print MSS and average steps to reach goal
    print(f"\n{text} MSS between initial and final Q-table: {mss:.6f}")
    print(f"{text} Avg Steps to Reach Goal: {np.mean(steps_per_episode):.2f}")

    # Visualizing Q-Values as a Heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(Q, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(text + " Q-Table Heatmap for FrozenLake")
    plt.xlabel("Actions (0=Left, 1=Down, 2=Right, 3=Up)")
    plt.ylabel("States (Grid Cells)")
    plt.show()


def test_trained_agent(env, Q):
    # Test the trained agent
    num_test_episodes = 5
    for test in range(num_test_episodes):
        print(f"\nTest Episode {test+1}")
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info

        env.render()
        steps = 0  # Track steps in test episode
        for _ in range(10):  # Max steps per test
            steps += 1
            action = np.argmax(Q[state])  # Choose best learned action
            print(f"Chosen action: {action}")  # Debug agent movement

            next_state_info = env.step(action)
            state = (
                next_state_info[0]
                if isinstance(next_state_info, tuple)
                else next_state_info
            )
            env.render()

            if next_state_info[2]:  # If done (goal or hole)
                break
        print(f"Steps taken in this episode: {steps}")

    env.close()
