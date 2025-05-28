"""
dqn_env.py

Este script define un entorno de simulación para el entrenamiento de un modelo
DQN-LSTM para la gestión de carga de vehículos eléctricos. Utiliza datos 
preprocesados de patrones de carga, potencia no gestionable y disponibilidad 
del vehículo. El entorno permite simular acciones de carga y calcular 
recompensas basadas en el estado del vehículo y las condiciones de la red.

Autor: Álvaro González
"""

import numpy as np
import pandas as pd
import json

with open("data_dqn/processed_ev_charging_patterns_dqn_constants.json") as f:
    constants = json.load(f)

class EVChargingEnv:
    def __init__(self, data, p_max=constants['max_charging_power (kW)'], soc_target=0.8, initial_soc=constants['avg_initial_soc (%)'] / 100, steps_per_episode=96, p_red=7):
        self.data = data
        self.p_max = p_max  # Potencia máxima normalizada
        self.soc_target = soc_target
        self.initial_soc = initial_soc
        self.steps_per_episode = steps_per_episode  # 15 min slots = 96 por día
        self.p_red = p_red  # Potencia contratada con la red
        self.initial_price = constants['avg_price_per_kwh (USD/kWh)']
        self.current_price = self.initial_price
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_price = self.initial_price
        self.soc = np.random.standard_normal(0.03, 0.25)
        self.done = False
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        time_norm = 2 * (self.current_step / self.steps_per_episode) - 1  # Normalizado a [-1,1]

        state = np.array([
            self.soc,
            row['Charging Rate (kW)'],
            row['Energy Consumed (kWh)'],
            row['Battery Capacity (kWh)'],
            self.soc_target - self.soc,
            time_norm,
            row['P_NG_kW'],
            row['Available']
        ])
        return state

    def step(self, action):
        row = self.data.iloc[self.current_step]
        available = row['Available']
        p_ng = row['P_NG_kW']

        # Acción: 0 = no cargar, 1 = cargar
        power = self.p_max * action
        delta_soc = power * 0.25  # 15 min = 0.25 h

        reward = 0

        if available == 0 and action == 1:
            reward -= 100
            power = 0
            delta_soc = 0

        # Penalizar si se supera la potencia contratada
        if action == 1 and (power + p_ng) > self.p_red:
            reward -= 15
            power = 0
            delta_soc = 0

        # Aplicar carga si todo es válido
        self.soc += delta_soc

        # Límite SOC máximo
        if self.soc > 1.0:
            self.soc = 1.0
            power = 0
            delta_soc = 0
            
        if self.soc == 1.0 and power == 1:
            reward -= 10
            power = 0
            delta_soc = 0

        # Aproximar precio
        time_minutes = self.current_step * 15
        hour = (time_minutes // 60) % 24
        minute = time_minutes % 60
        is_night = hour < 6 or (hour == 6 and minute <= 30) or hour >= 18

        drift = np.random.normal(-0.002 if is_night else 0.002, 0.01)
        self.current_price *= (1 + drift)
        # Limitar a +-50% del inicial
        if self.current_price > 1.5 * self.initial_price or self.current_price < 0.5 * self.initial_price:
            self.current_price = self.initial_price

        price = self.current_price
        reward += - price * power  # Función objetivo

        # Promover carga, aunque sea poco
        self.current_step += 1
        if self.current_step >= self.steps_per_episode:
            self.done = True
            if self.soc < self.soc_target:
                reward += 4/(self.soc_target - self.soc)
                
            elif self.soc == self.soc_target:
                reward += 50

        next_state = self._get_state()
        return next_state, reward, self.done, {}
