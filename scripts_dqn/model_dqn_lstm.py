"""
model_dqn_lstm.py

Define la arquitectura de red neuronal para el agente DQN utilizando LSTM.
Permite capturar dependencias temporales entre estados.

Autor: Álvaro González
"""

import torch
import torch.nn as nn

class DQNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, lstm_layers=1):
        """
        Inicializa la red.

        Args:
            input_dim (int): Dimensión del vector de estado.
            output_dim (int): Número de acciones posibles.
            hidden_dim (int): Número de neuronas por capa oculta.
            lstm_layers (int): Número de capas LSTM apiladas.
        """
        super(DQNLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Propagación.

        Args:
            x (Tensor): Tensor de entrada (batch_size, sequence_length, input_dim)

        Returns:
            Tensor: Q-valores para cada acción (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        last_output = lstm_out[:, -1, :]  # Tomar el último output de la secuencia
        x = torch.relu(self.fc1(last_output))
        return self.out(x)