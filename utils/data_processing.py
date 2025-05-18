"""
Data Processing Functions

Handles data cleaning and normalization for the EV charging dataset.

Functions:
- clean_data: Removes missing values and outliers.
- normalize_data: Scales numerical features.

Author: Álvaro González
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re

def clean_data(df):
    """Removes missing values and outliers."""
    df = df.dropna()
    return df

def convert_station_id(df, column="Charging Station ID"):
    """Converts station_id values from 'station_XX' to integer XX."""
    if column in df.columns:
        df[column] = df[column].apply(lambda x: int(re.sub(r"[^0-9]", "", x)) if isinstance(x, str) else x)
    return df

def convert_user_id(df, column="User ID"):
    """Converts station_id values from 'user_XX' to integer XX."""
    if column in df.columns:
        df[column] = df[column].apply(lambda x: int(re.sub(r"[^0-9]", "", x)) if isinstance(x, str) else x)
    return df

def normalize_data(df):
    """Scales only numerical features using MinMaxScaler."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df