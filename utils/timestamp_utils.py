"""
Funciones auxiliares para manejo de datos temporales del dataset EV.

Álvaro González · TFG
"""

import pandas as pd
import numpy as np

def extract_user_data(df, user_id):
    user_df = df[df["User ID"] == user_id].copy()
    
    # Convert Charging Start Time to datetime, coercing errors
    user_df["Charging Start Time"] = pd.to_datetime(user_df["Charging Start Time"], errors="coerce")
    user_df = user_df.dropna(subset=["Charging Start Time"])
    user_df.sort_values("Charging Start Time", inplace=True)
    user_df.set_index("Charging Start Time", inplace=True)

    # Definir disponibilidad basada en duración de carga
    duration = int(user_df["Charging Duration (hours)"].iloc[0])
    available_vector = [1 if i < duration else 0 for i in range(len(user_df))]
    user_df["available"] = available_vector
    
    # Replace zeros in "Energy Consumed (kWh)" with NaN so they will be dropped
    user_df["Energy Consumed (kWh)"] = user_df["Energy Consumed (kWh)"].replace(0, np.nan)
    
    # Compute price per kWh from cost and energy consumed
    user_df["price"] = user_df["Charging Cost (USD)"] / user_df["Energy Consumed (kWh)"]

    # Drop rows with NaN in price or other essential columns
    user_df.dropna(subset=["price", "State of Charge (Start %)", "State of Charge (End %)", "Battery Capacity (kWh)"], inplace=True)

    # Map Charging Rate to max_power_kW, with fallback value
    user_df["max_power_kW"] = user_df["Charging Rate (kW)"].fillna(3.6)

    return user_df