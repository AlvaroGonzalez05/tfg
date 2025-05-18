"""
dqn_env.py

Entorno de simulación para carga de vehículos eléctricos orientado a entrenamiento de agentes DQN.
Permite simular la evolución del estado de carga (SOC), disponibilidad del vehículo y coste de energía.

Autor: Álvaro González
"""

import numpy as np
import pandas as pd

class EVChargingEnv:
    def __init__(self, data, p_max=1.0, soc_target=0.8, initial_soc=0.2, steps_per_episode=96):
        self.data = data
        self.p_max = p_max  # Potencia máxima normalizada
        self.soc_target = soc_target
        self.initial_soc = initial_soc
        self.steps_per_episode = steps_per_episode  # 15 min slots = 96 por día
        self.reset()

    def reset(self):
        self.current_step = 0
        self.soc = self.initial_soc
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Asume que data está ordenada temporalmente
        row = self.data.iloc[self.current_step]
        time_norm = 2 * (self.current_step / self.steps_per_episode) - 1  # Normalizado a [-1,1]

        state = np.array([
            self.soc,
            row['Charging Rate (kW)'],
            row['Energy Consumed (kWh)'],
            row['Battery Capacity (kWh)'],
            self.soc_target - self.soc,
            time_norm
        ])
        return state

    def step(self, action):
        # Acción: 0 = no cargar, 1 = cargar a p_max
        power = self.p_max * action
        delta_soc = power * (0.25)  # 15 min slot → 0.25 h
        self.soc += delta_soc

        # Límite SOC máximo
        if self.soc > 1.0:
            self.soc = 1.0

        # Coste simple proporcional al uso de energía
        price = self.data.iloc[self.current_step]['Charging Rate (kW)']
        reward = - price * power  # Penaliza coste

        # Penalización por no cumplir objetivo al final
        self.current_step += 1
        if self.current_step >= self.steps_per_episode:
            self.done = True
            if self.soc < self.soc_target:
                reward -= 10 * (self.soc_target - self.soc)

        next_state = self._get_state()
        return next_state, reward, self.done, {}
    
