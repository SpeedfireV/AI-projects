import numpy as np
import random

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 500

NUM_STATES = 5
ACTIONS = [0, 1]
GOAL_STATE = 4

q_table = np.zeros((NUM_STATES, len(ACTIONS)))


def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    else:
        return np.argmax(q_table[state])


for episode in range(EPISODES):
    state = 0
    done = False

    while not done:
        action = choose_action(state)

        if action == 1:
            next_state = min(state + 1, NUM_STATES - 1)
        else:
            next_state = max(state - 1, 0)

        if next_state == GOAL_STATE:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

print("Wyuczona Tablica Q:")
print(q_table)