"""
train_dqn_lstm.py

Entrenamiento del agente DQN usando una arquitectura LSTM para capturar dependencias temporales
en la carga de vehículos eléctricos.

Autor: Álvaro González
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pandas as pd

from env_dqn import EVChargingEnv
from model_dqn_lstm import DQNLSTM

# Parámetros de entrenamiento
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 0.1
LR = 1e-3
NUM_EPISODES = 200
TARGET_UPDATE = 10
SEQUENCE_LENGTH = 4  # Número de pasos pasados que alimentamos al LSTM

# Cargar datos preprocesados
data = pd.read_csv('data_dqn/processed_ev_charging_patterns_dqn.csv')

# Crear entorno
env = EVChargingEnv(data)

# Inicializar redes
state_dim = 6  # Ajustar si el espacio de estados cambia
action_dim = 2  # Cargar o no cargar

policy_net = DQNLSTM(state_dim, action_dim)
target_net = DQNLSTM(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=10000)

def select_action(state_sequence):
    if random.random() < EPSILON:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = random.sample(replay_buffer, BATCH_SIZE)
    state_seqs, actions, rewards, next_state_seqs, dones = zip(*batch)

    state_seqs = torch.FloatTensor(np.array(state_seqs))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_state_seqs = torch.FloatTensor(np.array(next_state_seqs))
    dones = torch.FloatTensor(dones)

    q_values = policy_net(state_seqs).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_state_seqs).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    state_sequence = [state for _ in range(SEQUENCE_LENGTH)]

    total_reward = 0

    while not done:
        action = select_action(state_sequence)
        next_state, reward, done, _ = env.step(action)

        next_state_sequence = state_sequence[1:] + [next_state]

        replay_buffer.append((state_sequence, action, reward, next_state_sequence, float(done)))

        state_sequence = next_state_sequence
        total_reward += reward

        optimize_model()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f'Episodio {episode}, Recompensa total: {total_reward:.2f}')

# Guardar modelo
torch.save(policy_net.state_dict(), '../models/dqn_lstm_model.pth')
print("✅ Modelo LSTM guardado en models/dqn_lstm_model.pth")
