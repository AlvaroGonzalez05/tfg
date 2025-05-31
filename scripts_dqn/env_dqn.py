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
import json
import pandas as pd  # ya que lo necesitamos para leer el CSV

with open("data_dqn/processed_ev_charging_patterns_dqn_constants.json") as f:
    constants = json.load(f)

class EVChargingEnv:
    def __init__(self, data, p_max = constants['max_charging_power (kW)'], 
                 soc_target = 0.8, initial_soc = constants['avg_initial_soc (%)'] / 100,
                 steps_per_episode = 96, p_red = 7):
        self.data = data
        self.p_max = p_max  # Potencia máxima normalizada
        self.soc_target = soc_target
        self.initial_soc = initial_soc
        self.steps_per_episode = steps_per_episode  # 15 min slots = 96 por día
        self.p_red = p_red  # Potencia contratada
        self.initial_price = constants['avg_price_per_kwh (USD/kWh)']
        self.current_price = self.initial_price
        df_pg = pd.read_csv("data_dqn/preprocessed/P_NG_household.csv")
        self.p_ng_series = df_pg["P_NG_kW"].values
        self.availability_series = df_pg["Available"].values
        self.price_series = df_pg["Electricity Price (EUR/kWh)"].values
        self.reset()

    def reset(self):
        self.current_step = 0
        self.current_price = self.initial_price
        self.soc = np.random.uniform(0.03, 0.25)
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
        available = self.availability_series[self.current_step]
        
        p_ng = self.p_ng_series[self.current_step]
        p_d = self.p_red - p_ng
        
        price = self.price_series[self.current_step]
        self.current_price = price

        # Acción: 0 = no cargar, 1 = cargar
        power = self.p_max * action
        delta_soc = power * 0.25  # 15 min = 0.25 h

        reward = 0
        
        # Penalizaciones y bonificaciones basadas en la disponibilidad
        # Si la disponibilidad es baja y se intenta cargar, penalizar fuertemente
        # Si la disponibilidad es alta y se carga, bonificar
        if available < 0.2 and power > 0:
            reward -= 20
        elif available < 0.5 and power > 0:
            reward -= 10 * (0.5 - available)
        elif available > 0.8 and power > 0:
            reward += 5

        # potencia_disponible = p_red - p_ng
        if power > p_d:
            reward -= 10
        else:
            reward += 10

        # p * 0.25 + soc <= 1.0
        if self.soc + delta_soc > 1.0:
            power = (1.0 - self.soc) / 0.25
            delta_soc = power * 0.25

        elif self.soc == self.soc_target and power > 0:
            reward -= power * price

        # p < p_red - p_ng
        if power > self.p_red - p_ng:
            reward -= 10

        # soc_min <= soc <= soc_max
        if self.soc + delta_soc < 0.03 or self.soc + delta_soc > 0.25:
            reward -= 10

        # Aplicar carga si todo es válido
        self.soc += delta_soc

        reward += - price * power  # Función objetivo

        next_state = self._get_state()
        return next_state, reward, self.done, {}