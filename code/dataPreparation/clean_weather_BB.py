'''
filtered out everything that isnt in Berlin or Brandenburg 
now we went from 30226 rows to 2134 rows
'''

import os
import pandas as pd

# File paths
input_file_path = "data/cleaned/cleaned_weather.csv"
output_directory = "data/cleaned"
output_file_path = os.path.join(output_directory, "cleaned_weather_BB.csv")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

try:
    # Read the cleaned data
    df = pd.read_csv(input_file_path)

    # Filter rows for Berlin or Brandenburg in the 'BDL_RRmax' column
    if 'BDL_RRmax' in df.columns:
        df_filtered = df[df['BDL_RRmax'].str.contains("Berlin|Brandenburg", case=False, na=False)]

        # Save the filtered data
        df_filtered.to_csv(output_file_path, index=False)
        print(f"Filtered data saved successfully to {output_file_path}")
    else:
        print("The column 'BDL_RRmax' is missing in the input data.")

except FileNotFoundError:
    print(f"Input file not found: {input_file_path}")
except KeyError as e:
    print(f"Some required columns are missing in the input data: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
    
    
# check out the difference in rows - commented out after checking
# print(f"Shape of the data: {df_filtered.shape}")
# print("\nFirst few rows of data:")
# print(df_filtered.head())
