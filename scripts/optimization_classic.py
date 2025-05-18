import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.timestamp_utils import extract_user_data
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

# Use the processed file which contains valid datetime strings
#INPUT_CSV = "data/processed_ev_charging_patterns.csv"
INPUT_CSV = "data/ev_charging_patterns_clean.csv"
OUTPUT_DIR = "data/charging_schedules"

def optimize_user_charging(user_data, user_id):
    if user_data.empty:
        print(f"[WARN] No data available for user {user_id}. Skipping...")
        return
    
    user_data.index = pd.to_datetime(user_data.index)
    
    print(f"\n=== Preview data for user {user_id} ===")
    print(user_data.head(5))
    print(user_data[["Charging Duration (hours)", "Battery Capacity (kWh)", "State of Charge (Start %)", "State of Charge (End %)", "Charging Rate (kW)", "Charging Cost (USD)"]].describe())
    
    timestamps = user_data.index
    n = len(timestamps)

    # Parámetros
    price = user_data["Charging Cost (USD)"]  # €/kWh
    max_power = user_data["Charging Rate (kW)"]
    soc_initial = user_data["State of Charge (Start %)"] * user_data["Battery Capacity (kWh)"] / 100
    soc_final = user_data["State of Charge (End %)"] * user_data["Battery Capacity (kWh)"] / 100
    
    avail = pd.Series(0, index=timestamps)
    start_time = timestamps[0]
    duration_hours = user_data["Charging Duration (hours)"].iloc[0]
    end_time = pd.Timestamp(start_time) + pd.Timedelta(hours=duration_hours)

    # Marcar como disponible las horas dentro del rango de carga
    avail.loc[(avail.index >= start_time) & (avail.index < end_time)] = 1
    
    if soc_final.iloc[0] <= soc_initial.iloc[0]:
        print(f"[SKIP] User {user_id}: already charged.")
        return
    
    required_energy = (soc_final - soc_initial).iloc[0]
    available_energy = (avail * max_power).sum()  # Aprox total energía que podría cargarse

    print(f"\nUser {user_id}")
    print(f"  - Initial SOC: {soc_initial.iloc[0]:.2f} kWh")
    print(f"  - Final SOC:   {soc_final.iloc[0]:.2f} kWh")
    print(f"  - Required Energy: {required_energy:.2f} kWh")
    print(f"  - Hours available: {avail.sum()} h")
    print(f"  - Max power range: {max_power.min()} - {max_power.max()} kW")
    print(f"  - Est. Available Energy: {available_energy:.2f} kWh\n")

    # Modelo
    m = Model("ev_charging_optimization")
    m.setParam("OutputFlag", 0)

    x = m.addVars(n, lb=0.0, ub=max_power.max(), name="x")  # Potencia en t
    soc = m.addVars(n, lb=0.0, name="soc")

    # Restricciones
    for t in range(n):
        if avail.iloc[t] == 0:
            m.addConstr(x[t] == 0, name=f"not_available_{t}")
        elif t == 0:
            m.addConstr(soc[t] == soc_initial.iloc[0] + x[t] * 1, name="soc_initial")
        else:
            m.addConstr(soc[t] == soc[t-1] + x[t] * 1, name=f"soc_update_{t}")

    m.addConstr(soc[n-1] >= soc_final.iloc[0], name="soc_final")

    # Minimizar coste total
    m.setObjective(quicksum(x[t] * price.iloc[t] for t in range(n)), GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        # Guardar
        result = pd.DataFrame({
            "timestamp": timestamps,
            "charging_power_kW": [x[t].X for t in range(n)],
            "SOC_kWh": [soc[t].X for t in range(n)]
        }).set_index("timestamp")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        result.to_csv(f"{OUTPUT_DIR}/optimal_schedule_user_{user_id}.csv")
        print(f"Optimization finished successfully for user {user_id}. ✅")
    else:
        print(f"Optimization did not finish optimally for user {user_id}. Status: {m.status}")


if __name__ == "__main__":
    column_names = [
        "Charging Start Time", 
        "State of Charge (Start %)", 
        "State of Charge (End %)", 
        "Battery Capacity (kWh)", 
        "Charging Duration (hours)", 
        "Charging Rate (kW)", 
        "Charging Cost (USD)", 
        "Energy Consumed (kWh)", 
        "Charger Type", 
        "User ID", 
        "User Type"
    ]
    df = pd.read_csv(INPUT_CSV, header=None, names=column_names)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df["SOC_bin"] = pd.cut(df["State of Charge (Start %)"], bins=bins)

    soc_groups = df.groupby("SOC_bin")

    for soc_range, group in soc_groups:
        if group.empty:
            continue
        print(f"\n\n### Optimizing group with SOC between {soc_range} ###")
        optimize_user_charging(group.set_index("Charging Start Time"), soc_range)