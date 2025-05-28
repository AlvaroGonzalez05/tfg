"""
merge_state_components.py

Este script une los datos de carga de vehículo eléctrico procesados con los datos
de potencia no gestionable y disponibilidad del vehículo. Se asegura que todos los
datasets estén correctamente alineados temporalmente para el entrenamiento del modelo DQN-LSTM.

Autor: Álvaro González
"""

import pandas as pd
import os

# Rutas de entrada
charging_path = "/Users/alvarogonzaleztabernero/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/ICAI/4/tfg_code/data/processed_ev_charging_patterns.csv"
p_ng_path = "/Users/alvarogonzaleztabernero/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/ICAI/4/tfg_code/data_dqn/preprocessed/P_NG_household.csv"
availability_path = "/Users/alvarogonzaleztabernero/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/ICAI/4/tfg_code/data_dqn/data_lpg/Results/SumProfiles_900s.HH1.Electricity for Car Charging.csv"

# Ruta de salida
output_path = "/Users/alvarogonzaleztabernero/Library/CloudStorage/OneDrive-UniversidadPontificiaComillas/ICAI/4/tfg_code/data_dqn/preprocessed/state_sequences_dqn_lstm.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Cargar datos
df_charge = pd.read_csv(charging_path, header=None)
df_charge.columns = [
    "Charging Duration (hours)", "SOC_Start", "SOC_End", "Charging Rate (kW)",
    "Energy Consumed (kWh)", "Battery Capacity (kWh)", "Charging Cost (USD)",
    "Distance Driven (km)", "Temperature (°C)", "Charger Type", "Electricity Price"
]

df_p_ng = pd.read_csv(p_ng_path)
df_avail = pd.read_csv(availability_path, sep=";")

# Procesar disponibilidad
df_avail["Time"] = pd.to_datetime(df_avail["Time"], dayfirst=True)
df_avail["Available"] = (df_avail["Sum [kWh]"] > 0).astype(int)
df_avail = df_avail[["Time", "Available"]]

# Procesar P_NG
df_p_ng["Time"] = pd.to_datetime(df_p_ng["Time"])

# Crear columna de tiempo sintética para df_charge
df_charge["Time"] = df_p_ng["Time"][:len(df_charge)]

# Verificar alineación completa
merged = df_charge.merge(df_p_ng, on="Time", how="inner").merge(df_avail, on="Time", how="inner")

# Calcular ΔSOC normalizado
merged["Delta_SOC"] = merged["SOC_End"] - merged["SOC_Start"]

# Crear columna de tiempo normalizado entre [-1, 1]
merged["t_norm"] = 2 * (merged.index / len(merged)) - 1

# Selección final de columnas para el modelo
columns_finales = [
    "SOC_Start", "Charging Rate (kW)", "Energy Consumed (kWh)", "Battery Capacity (kWh)",
    "Delta_SOC", "t_norm", "P_NG_kW", "Available", "Electricity Price"
]

merged_final = merged[columns_finales]

# Guardar
merged_final.to_csv(output_path, index=False)
print(f"✅ Dataset combinado guardado en {output_path}")