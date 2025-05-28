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
        'State of Charge (End %)',
        'Charging Cost (USD)'
    ]
    df = df[relevant_cols]

    # Convertir tiempos a datetime
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])

    # Calcular duración en minutos y horas
    df['Charging Duration (min)'] = (df['Charging End Time'] - df['Charging Start Time']).dt.total_seconds() / 60
    df['Charging Duration (hours)'] = df['Charging Duration (min)'] / 60

    # Calcular precio por kWh
    df['Electricity Price (USD/kWh)'] = df['Charging Cost (USD)'] / df['Energy Consumed (kWh)']
    df['Electricity Price (USD/kWh)'] = df['Electricity Price (USD/kWh)'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calcular métricas fijas antes de normalizar
    avg_price_per_kwh = df['Electricity Price (USD/kWh)'].mean()
    avg_battery_capacity = df['Battery Capacity (kWh)'].mean()
    max_charging_power = df['Charging Rate (kW)'].max()
    avg_initial_soc = df['State of Charge (Start %)'].mean()

    constants_dict = {
        'avg_price_per_kwh (USD/kWh)': avg_price_per_kwh,
        'avg_battery_capacity (kWh)': avg_battery_capacity,
        'max_charging_power (kW)': max_charging_power,
        'avg_initial_soc (%)': avg_initial_soc
    }

    # Guardar diccionario como JSON
    import json
    with open(output_csv.replace('.csv', '_constants.json'), 'w') as f:
        json.dump(constants_dict, f, indent=4)

    # Columnas a normalizar
    cols_to_normalize = [
        'Battery Capacity (kWh)',
        'Energy Consumed (kWh)',
        'Charging Rate (kW)',
        'State of Charge (Start %)',
        'State of Charge (End %)'
    ]

    for col in cols_to_normalize:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val != min_val:
            df[col] = 2 * (df[col] - min_val) / (max_val - min_val) - 1
        else:
            df[col] = 0  # todos los valores iguales, se pone a 0

    # Guardar preprocesado
    csv_string = df.to_csv(index=False)
    csv_string = csv_string.replace(',,', ',0,')
    with open(output_csv, 'w') as f:
        f.write(csv_string)
    print(f'Datos preprocesados guardados en {output_csv}')
    return

if __name__ == "__main__":
    preprocess_ev_data(
        input_csv='data_dqn/ev_charging_patterns.csv',
        output_csv='data_dqn/processed_ev_charging_patterns_dqn.csv'
    )