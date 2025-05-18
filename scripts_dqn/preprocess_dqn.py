"""
preprocess_dqn.py

Este script realiza el preprocesado específico de los datos de carga de vehículos eléctricos
para su uso en redes de aprendizaje por refuerzo. Elimina columnas irrelevantes, normaliza los
datos relevantes y genera un conjunto de datos adecuado para el entrenamiento de un agente DQN.

Autor: Álvaro González
"""

import pandas as pd
import numpy as np

def preprocess_ev_data(input_csv, output_csv):
    # Cargar datos
    df = pd.read_csv(input_csv)

    # Columnas relevantes
    relevant_cols = [
        'Battery Capacity (kWh)',
        'Charging Start Time',
        'Charging End Time',
        'Energy Consumed (kWh)',
        'Charging Rate (kW)',
        'State of Charge (Start %)',
        'State of Charge (End %)'
    ]
    df = df[relevant_cols]

    # Convertir tiempos a datetime
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])

    # Calcular duración en minutos
    df['Charging Duration (min)'] = (df['Charging End Time'] - df['Charging Start Time']).dt.total_seconds() / 60

    # Normalización simple [-1,1]
    for col in ['Battery Capacity (kWh)', 'Energy Consumed (kWh)', 'Charging Rate (kW)',
                'State of Charge (Start %)', 'State of Charge (End %)']:
        max_val = df[col].max()
        min_val = df[col].min()
        df[col] = 2 * (df[col] - min_val) / (max_val - min_val) - 1

    # Guardar preprocesado
    df.to_csv(output_csv, index=False)
    print(f'Datos preprocesados guardados en {output_csv}')

if __name__ == "__main__":
    preprocess_ev_data(
        input_csv='data_dqn/ev_charging_patterns.csv',
        output_csv='data_dqn/processed_ev_charging_patterns_dqn.csv'
    )
    