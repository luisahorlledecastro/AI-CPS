"""
data from Kaggle too big, so we used local path to load the data
data can be found in: https://drive.google.com/drive/folders/1j-b6GL-Ng2o3Ge0tsPiIoIOGq100NpsQ
"""

# import libraries
import pandas as pd
import os


def add_states(save=False):
    # load data
    # TODO: Download the data from the link provided above and store it in the data folder
    data = pd.read_csv("./data/scraped/DBtrainrides.csv").drop(
        columns=['ID', 'line', 'path', 'eva_nr', 'category',
                 'zip', 'long', 'lat', 'arrival_plan', 'departure_plan','arrival_change',
                 'departure_change', 'arrival_delay_m', 'departure_delay_m', 'info',
                 'arrival_delay_check', 'departure_delay_check']).drop_duplicates()

    df = data[data["state"].isin(["Berlin", "Brandenburg"])]

    if save:

        # Directory to save the file
        save_directory = "../../data/cleaned"
        os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

        # File name and path
        csv_file_name = "station_city_match.csv"
        output = os.path.join(save_directory, csv_file_name)

        df.to_csv(output, index=False)

    return df
