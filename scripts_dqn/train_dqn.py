"""
train_dqn.py

Entrenamiento del agente DQN usando el entorno de carga de vehículos eléctricos.
Utiliza el modelo DQN definido en model_dqn.py y el entorno definido en env_dqn.py.

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
from model_dqn import DQN

# Parámetros de entrenamiento
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 0.1
LR = 1e-3
NUM_EPISODES = 200
TARGET_UPDATE = 10

# Cargar datos preprocesados
data = pd.read_csv('data_dqn/processed_ev_charging_patterns_dqn.csv')

# Crear entorno
env = EVChargingEnv(data)

# Inicializar redes
state_dim = 6  # Ajustar si el espacio de estados cambia
action_dim = 2  # Cargar o no cargar

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = deque(maxlen=10000)

def select_action(state):
    if random.random() < EPSILON:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)

        replay_buffer.append((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward

        optimize_model()

        if done:
            break

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f'Episodio {episode}, Recompensa total: {total_reward:.2f}')
    
    # Guardar modelo entrenado
    torch.save(policy_net.state_dict(), 'models/dqn_model.pth')
    print("✅ Modelo guardado en models/dqn_model.pth")
