'''
Necessary column descriptions:
- Start_Time: Start time of the weather observation
- End_Time: End time of the weather observation
- x_RRmax: x-coordinate of the maximum precipitation
- y_RRmax: y-coordinate of the maximum precipitation
- BDL_RRmax: State of the precipitation
- LKS_RRmax: City of the precipitation
- RRmax: Maximum precipitation amount
- Duration: Duration of the precipitation event
- Area: Affected area of the precipitation event
- RRmean: Average precipitation amount
- All V3_ and V4_ columns: Wind velocity
'''

import os
import pandas as pd

def remove_columns(df): # Remove unnecessary columns for easier analysis

    columns_to_keep = [
        "Start_Time", "End_Time", "x_RRmax", "y_RRmax", "RRmax", "Duration", "Area", "RRmean","BDL_RRmax","LKS_RRmax"
    ]
    return df[[col for col in df.columns if col in columns_to_keep or col.startswith("V3_") or col.startswith("V4_")]]

def clean_and_preprocess(df):
    df = df.copy()

    # 1-check if numeric columns exist and convert them
    numeric_columns = ['duration', 'precipitation_amount', 'precipitation_intensity']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2-Handle missing values only if columns exist
    columns_to_check = ['precipitation_amount', 'precipitation_intensity']
    existing_columns = [col for col in columns_to_check if col in df.columns]
    if existing_columns:
        df = df.dropna(subset=existing_columns)

    return df

# File paths
input_file_path = "data/scraped/scraped_weather.csv"
output_directory = "data/cleaned"
os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists
output = os.path.join(output_directory, "cleaned_weather.csv")


# Read, clean, and preprocess the data
try:

    df = pd.read_csv(input_file_path)
    df = remove_columns(df)
    df = clean_and_preprocess(df)

    df.to_csv(output, index=False) # Save cleaned data
    print(f"Cleaned data saved successfully to {output}")
except FileNotFoundError:
    print(f"Input file not found: {input_file_path}")
except KeyError as e:
    print(f"Some required columns are missing in the input data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")


# Print column names to see the difference - commented out after checking
# print("Available columns:")
# print(df.columns.tolist())

# check out the difference in rows - commented out after checking
# print(f"Shape of the data: {df.shape}")
# print("\nFirst few rows of data:")
# print(df.head())
