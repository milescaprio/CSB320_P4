import numpy as np
import matplotlib.pyplot as plt

# All functions are for three-arm bandit problem


def monte_carlo_ep_greedy(
    true_rewards=[0.2, 0.5, 0.8], num_episodes=1000, epsilon=0.1, random_state=42
):
    rng = np.random.default_rng(seed=random_state)
    # Simulated 3-arm bandit with unknown rewards
    true_rewards = [0.2, 0.5, 0.8]  # Actual reward probabilities for 3 slot machines
    num_actions = len(true_rewards)

    returns = [[] for _ in range(num_actions)]  # Store returns for each action
    Q = np.zeros(num_actions)  # Estimated Q-values
    chosen = np.zeros(num_actions)  # Count of times each action was chosen
    Q_history = np.zeros((num_episodes, num_actions))  # Store Q-values per episode

    # Monte Carlo with ε-greedy exploration
    for episode in range(num_episodes):
        # Select action using ε-greedy
        if np.random.rand() < epsilon:
            action = rng.choice(num_actions)  # Explore (random action)
        else:
            action = np.argmax(Q)  # Exploit (best known arm)

        # Simulate reward from chosen action
        reward = rng.binomial(
            1, true_rewards[action]
        )  # Reward is 1 with probability true_rewards[action]

        # Store return and update Q-value (MC update: average of observed rewards)
        returns[action].append(reward)
        Q[action] = np.mean(returns[action])  # Average returns for the action

        # Track Q-values over time
        Q_history[episode] = Q
        chosen[action] += 1

    return Q, Q_history, chosen


def final_estimates(Q, num_episodes, true_rewards, chosen, do_print=True):
    # Print final estimated values
    yielded = np.dot(chosen, true_rewards)
    if do_print:
        print(f"Final Estimated Q-values after {num_episodes} episodes: {Q}")
        print(f"True Q values: {true_rewards}")
        print(f"Best arm according to Monte Carlo estimation: Arm {np.argmax(Q) + 1}")
        print(f"True best arm: Arm {np.argmax(true_rewards) + 1}")
        print(f"Potential Total Reward: {np.max(true_rewards) * num_episodes}")
        print(f"Yielded Total Reward: {yielded}")
        print(
            f"Percentage of potential reward yielded: {yielded / (np.max(true_rewards) * num_episodes) * 100:.2f}%"
        )
    else:
        return {
            "mse": np.mean((Q - true_rewards) ** 2),
            "correct_best_arm": np.argmax(Q) == np.argmax(true_rewards),
            "yielded_reward_pct": yielded / (np.max(true_rewards) * num_episodes),
        }


def plot_final_estimates(Q, true_rewards, text):
    # Bar plot of final Q-values vs true rewards
    plt.figure(figsize=(10, 5))
    x = np.arange(len(Q))
    plt.bar(x - 0.2, Q, width=0.4, label="Estimated Q-values", color="blue")
    plt.bar(x + 0.2, true_rewards, width=0.4, label="True Rewards", color="orange")
    plt.xlabel("Arms")
    plt.ylabel("Q-values")
    plt.title("Estimated Q-values vs True Rewards " + text)
    plt.xticks(x, [f"Arm {i + 1}" for i in range(len(Q))])
    plt.legend()
    plt.show()


def plot_q_convergence(Q_history, text):
    # Plot convergence of Q-values over episodes
    plt.figure(figsize=(10, 5))
    plt.plot(Q_history)
    plt.xlabel("Episodes")
    plt.ylabel("Estimated Q-Values")
    plt.title("Convergence of Q-Values Over Episodes " + text)
    plt.legend(["Arm 1", "Arm 2", "Arm 3"])
    plt.show()
