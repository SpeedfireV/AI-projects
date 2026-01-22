import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.99
epsilon = 1
decay = 0.999
min_epsilon = 0.01

def run_experiment(alpha, episodes=2000):
    global gamma, epsilon, decay, min_epsilon
    env = gym.make("Taxi-v3")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_history = []

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            best_next_action = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_action
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * decay)
        rewards_history.append(total_reward)

    env.close()
    return rewards_history


alphas = [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]
num_trials = 3
episodes = 2000
all_results = {}

for a in alphas:
    print(f"Testowanie alpha = {a} ({num_trials} powtórzenia)...")
    trial_data = []
    for t in range(num_trials):
        print(f"  Próba {t + 1}/{num_trials}")
        trial_data.append(run_experiment(a, episodes))
    all_results[a] = np.array(trial_data)

plt.figure(figsize=(12, 7))

for a in alphas:
    mean_rewards = np.mean(all_results[a], axis=0)
    window = 100
    smooth_mean = np.convolve(mean_rewards, np.ones(window) / window, mode='valid')
    plt.plot(smooth_mean, label=f'Alpha = {a}')

plt.title(f'Q-learning (gamma={gamma}, eps={epsilon}): Średnia z {num_trials} prób')
plt.xlabel('Epizod')
plt.ylabel(f'Średnia nagroda (okno {window})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

filename = f"g{gamma}_e{epsilon}_d{decay}_mine{min_epsilon}.png"
plt.savefig(filename)
print(f"Wykres został zapisany jako: {filename}")

plt.show()

gamma_to_test = 0.99
for a in alphas:
    trial_final_rewards = []
    for t in range(num_trials):
        history = run_experiment(a, episodes)
        final_avg = np.mean(history[-100:])
        trial_final_rewards.append(final_avg)

    total_avg = np.mean(trial_final_rewards)
    print(f"{a:<10} | {total_avg:<40.2f}")