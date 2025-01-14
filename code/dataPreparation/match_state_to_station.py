"""
data from Kaggle too big. provide link TODO
"""

# import libraries
import pandas as pd
import os


# load data
data = pd.read_csv("/Users/luisahorlledecastro/Downloads/DBtrainrides.csv").drop(
    columns=['ID', 'line', 'path', 'eva_nr', 'category',
             'zip', 'long', 'lat', 'arrival_plan', 'departure_plan','arrival_change',
             'departure_change', 'arrival_delay_m', 'departure_delay_m', 'info',
             'arrival_delay_check', 'departure_delay_check']).drop_duplicates()

df = data[data["state"].isin(["Berlin", "Brandenburg"])]

# Directory to save the file
save_directory = "../../data/cleaned"
os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

# File name and path
csv_file_name = "station_city_match.csv"
output = os.path.join(save_directory, csv_file_name)

df.to_csv(output, index=False)
