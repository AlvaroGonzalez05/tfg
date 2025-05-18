

"""
hyperparameter_search_dqn_lstm.py

Exploración de hiperparámetros para el modelo DQN-LSTM mediante búsqueda por grid.
Evalúa diferentes combinaciones y guarda la mejor encontrada según la recompensa media.

Autor: Álvaro González
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd
from itertools import product

from env_dqn import EVChargingEnv
from model_dqn_lstm import DQNLSTM

# Configuración
NUM_EPISODES = 50
TARGET_UPDATE = 10
SEQUENCE_LENGTH = 4
EVAL_EPISODES = 5

# Espacio de búsqueda de hiperparámetros
batch_sizes = [64, 128, 256]
gammas = [0.95, 0.99, 0.995]
learning_rates = [1e-3, 5e-4, 1e-4]
epsilons = [0.05, 0.1, 0.2]
sequence_lengths = [4, 8, 16]

# Cargar datos
data = pd.read_csv('data_dqn/processed_ev_charging_patterns_dqn.csv')

# Función de evaluación rápida
def evaluate_model(env, model):
    rewards = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        done = False
        state_sequence = [state for _ in range(SEQUENCE_LENGTH)]
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0)
                action = model(state_tensor).argmax(1).item()
            next_state, reward, done, _ = env.step(action)
            state_sequence = state_sequence[1:] + [next_state]
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

# Grid search
best_reward = -np.inf
best_params = None

for batch_size, gamma, lr, epsilon, seq_len in product(batch_sizes, gammas, learning_rates, epsilons, sequence_lengths):
    print(f"Probando: batch_size={batch_size}, gamma={gamma}, lr={lr}, epsilon={epsilon}, sequence_length={seq_len}")

    env = EVChargingEnv(data)
    state_dim = 6
    action_dim = 2

    policy_net = DQNLSTM(state_dim, action_dim)
    target_net = DQNLSTM(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=10000)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        state_sequence = [state for _ in range(seq_len)]

        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0)
                    action = policy_net(state_tensor).argmax(1).item()

            next_state, reward, done, _ = env.step(action)
            next_state_sequence = state_sequence[1:] + [next_state]
            replay_buffer.append((state_sequence, action, reward, next_state_sequence, float(done)))
            state_sequence = next_state_sequence

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.stack(states))
                actions = torch.LongTensor(actions)
                rewards_batch = torch.FloatTensor(rewards_batch)
                next_states = torch.FloatTensor(np.stack(next_states))
                dones = torch.FloatTensor(dones)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards_batch + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    avg_reward = evaluate_model(env, policy_net)
    print(f"Recompensa media: {avg_reward:.2f}")

    if avg_reward > best_reward:
        best_reward = avg_reward
        best_params = {
            'batch_size': batch_size,
            'gamma': gamma,
            'lr': lr,
            'epsilon': epsilon,
            'sequence_length': seq_len
        }

print("✅ Búsqueda finalizada.")
print(f"Mejores hiperparámetros encontrados: {best_params}")
print(f"Mejor recompensa media: {best_reward:.2f}\n")
