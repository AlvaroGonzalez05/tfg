"""
preprocess_lpg_sumprofile.py

Procesa el archivo SumProfiles_900s.Apparent.csv generado por LoadProfileGenerator
para obtener la potencia no gestionable (P_NG) por intervalos de 15 minutos, en kW.

Autor: Álvaro González
"""

import pandas as pd
import os
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

input_path = "data_dqn/data_lpg/Results/SumProfiles_900s.Apparent.csv"
output_path = "data_dqn/preprocessed/P_NG_household.csv"

# Cargar CSV
df = pd.read_csv(input_path, sep=";")

# Convertir de kWh a kW (15 min = 0.25 h)
df["P_NG_kW"] = df["Sum [kWh]"] / 0.25

# Cargar archivo de disponibilidad
availability_path = "data_dqn/data_lpg/Results/SumProfiles_900s.General.Electricity for Car Charging.csv"
df_avail = pd.read_csv(availability_path, sep=";")

data_col = df_avail.columns[2]
from utils.availability_generator import generate_availability_profile
p_avail = generate_availability_profile(profile_type="worker", noise_std=0.1)
p_avail_extended = np.tile(p_avail, len(df_avail) // 96 + 1)[:len(df_avail)]
df_avail["Available"] = p_avail_extended

# Cargar y procesar precios de electricidad
price_path = "data_dqn/precios_luz.csv"
df_price = pd.read_csv(price_path)

# Convertir de MWh a kWh
df_price["value"] = df_price["value"] / 1000

# Repetir cada precio 4 veces (15 min × 4 = 1 hora)
stretched_prices = df_price["value"].repeat(4).reset_index(drop=True)

# Ensamblar df_out
df_out = df[["Time", "P_NG_kW"]]
df_out.loc[:, "Available"] = df_avail["Available"].values

# Asegurar que longitud coincide con df_out
if len(stretched_prices) > len(df_out):
    stretched_prices = stretched_prices.iloc[:len(df_out)]
elif len(stretched_prices) < len(df_out):
    stretched_prices = stretched_prices.reindex(range(len(df_out)), method="ffill")

df_out["Electricity Price (EUR/kWh)"] = stretched_prices.values

# Debug: Verifica el formato y columnas antes de guardar
print(df_out.head())
print(df_out.columns)

# Guardado robusto y mensaje de confirmación
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)
print(f"✅ Guardado final con {len(df_out.columns)} columnas en: {output_path}")