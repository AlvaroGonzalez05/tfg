"""
model_dqn.py

Define la arquitectura de red neuronal para el agente DQN. Esta versión utiliza una red
neuronal totalmente conectada (MLP) sin memoria. Está preparada para ser extendida con
LSTM en el futuro si se desea incorporar información temporal.

Autor: Álvaro González
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Inicializa la red.

        Args:
            input_dim (int): Dimensión del vector de estado.
            output_dim (int): Número de acciones posibles.
            hidden_dim (int): Número de neuronas por capa oculta.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Propagación.

        Args:
            x (Tensor): Tensor de entrada (batch_size, input_dim)

        Returns:
            Tensor: Q-valores para cada acción (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)