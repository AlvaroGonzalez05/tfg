"""
preprocess_lpg_sumprofile.py

Procesa el archivo SumProfiles_900s.Apparent.csv generado por LoadProfileGenerator
para obtener la potencia no gestionable (P_NG) por intervalos de 15 minutos, en kW.

Autor: Álvaro González
"""

import pandas as pd
import os

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
df_avail["Available"] = (df_avail[data_col] > 0).astype(int)

# Guardar resultado
df_out = df[["Time", "P_NG_kW"]]
df_out["Available"] = df_avail["Available"].values
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"✅ P_NG(t) exportado a: {output_path}")