import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path):
    header_df = pd.read_csv(csv_path, nrows=1)
    original_header = header_df.iloc[0].to_dict()
    column_labels = list(header_df.columns)

    df = pd.read_csv(csv_path, skiprows=1, header=None)
    df.columns = column_labels  # assign the original column headers

    # Identify the columns to drop by index:
    # 0 -> 'User ID'
    # 1 -> 'Vehicle Model'
    # 3 -> 'Charging Station ID'
    # 4 -> 'Charging Station Location'
    # 5 -> 'Charging Start Time'
    # 6 -> 'Charging End Time'
    # 11 -> 'Time of Day'
    # 12 -> 'Day of Week'
    # 18 -> 'Charger Type'
    cols_to_drop = [df.columns[i] for i in [0, 1, 3, 4, 5, 6, 11, 12, 18]]
    df = df.drop(columns=cols_to_drop)
    
    # Convert the driver profile column 'User Type' into categorical codes
    df["User Type"] = df["User Type"].astype("category").cat.codes
    
    return df, column_labels, original_header

def normalize_data(df):
    """Scales only numerical features using MinMaxScaler to [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_array = scaler.fit_transform(df)
    return pd.DataFrame(scaled_array, columns=df.columns)

if __name__ == "__main__":
    csv_file = "data/ev_charging_patterns.csv"
    processed_df, header_labels, original_header = preprocess_data(csv_file)
    processed_df = normalize_data(processed_df)
    
    processed_df.to_csv("data/processed_ev_charging_patterns.csv", index=False)
    
    # Save the original descriptive header into a separate file for later analysis
    with open("data/ev_charging_patterns_descriptive_header.json", "w") as f:
        json.dump(original_header, f, indent=4)
    
    print("Processed data saved to 'processed_ev_charging_patterns.csv'")
    print("Descriptive header saved to 'ev_charging_patterns_descriptive_header.json'")