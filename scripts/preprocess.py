"""
Preprocessing Script for EV Charging Dataset

This script extracts the most relevant columns related to EV charging 
and calculates additional derived features.

Features extracted:
- Charging Duration
- SOC Start & End
- Charging Rate
- Energy Consumed
- Battery Capacity
- Charging Cost
- Distance Driven
- Temperature
- Charger Type
- Electricity Price (calculated)

Author: Álvaro González
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_dataset(filepath):
    """Loads the EV charging dataset."""
    df = pd.read_csv(filepath)

    # Selecting relevant columns
    relevant_columns = [
        "Charging Duration (hours)",
        "State of Charge (Start %)",
        "State of Charge (End %)",
        "Charging Rate (kW)",
        "Energy Consumed (kWh)",
        "Battery Capacity (kWh)",
        "Charging Cost (USD)",
        "Distance Driven (since last charge) (km)",
        "Temperature (°C)",
        "Charger Type"
    ]

    df = df[relevant_columns]

    # Compute Electricity Price (USD/kWh)
    df["Electricity Price (USD/kWh)"] = df["Charging Cost (USD)"] / df["Energy Consumed (kWh)"]
    
    # Handle infinite or NaN values (e.g., division by zero cases)
    df["Electricity Price (USD/kWh)"] = df["Electricity Price (USD/kWh)"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def encode_categorical(df):
    """Encodes categorical variables using Label Encoding."""
    df["Charger Type"] = df["Charger Type"].astype("category").cat.codes
    return df

def normalize_data(df):
    """Scales only numerical features using MinMaxScaler to [-1, 1]."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def clean_data(df):
    """Removes rows with physically impossible values."""
    df = df[
        (df["Battery Capacity (kWh)"] > 0) &
        (df["Charging Duration (hours)"] > 0) &
        (df["State of Charge (Start %)"] >= 0) &
        (df["State of Charge (End %)"] >= 0) &
        (df["Charging Rate (kW)"] > 0) &
        (df["Energy Consumed (kWh)"] > 0)
    ].copy()
    return df

def preprocess_data(filepath):
    """Loads, processes, and normalizes the dataset."""
    df = load_dataset(filepath)
    df = clean_data(df)
    print("\n📌 Preprocessed Dataset Preview:")
    print(df.head())
    df = encode_categorical(df)
    df = normalize_data(df)

    print("\n📌 Final Preprocessed Dataset Preview:")
    print(df.head())

    return df

def preprocess_for_optimization(filepath):
    """Loads, processes, and returns the cleaned dataset without encoding or normalization."""
    df = load_dataset(filepath)
    df = clean_data(df)
    print("\n📌 Cleaned Dataset Preview:")
    print(df.head())
    return df

if __name__ == "__main__":
    df = preprocess_data("data/ev_charging_patterns.csv")
    df.to_csv("data/processed_ev_charging_patterns2.csv", index=False)
    
    df_clean = preprocess_for_optimization("data/ev_charging_patterns.csv")
    df_clean.to_csv("data/ev_charging_patterns_clean.csv", index=False)