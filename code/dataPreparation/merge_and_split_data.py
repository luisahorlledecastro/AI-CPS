

# import libraries
from meteostat import Point, Hourly
import pandas as pd
from datetime import timedelta
import os
from sklearn.model_selection import train_test_split


# load data
df_db_incomplete = pd.read_csv('../../data/cleaned/cleaned_data.csv', header=0)
df_weather = pd.read_csv('../../data/cleaned/cleaned_weather.csv')
df_city = pd.read_csv('../../data/cleaned/station_city_match.csv')

df_db = pd.merge(df_db_incomplete, df_city, on='station', how='left').dropna(how='any')
df_db['departure_plan_datetime'] = pd.to_datetime(df_db['departure_plan_datetime'])


def fetch_weather_batch(df):
    """
    Fetch temperature and rain status for unique location-time combinations in the DataFrame.
    """
    # Create a dictionary to store results
    weather_cache = {}

    # Identify unique (lat, long, start_time) combinations
    unique_locations = df[['lat', 'long', 'departure_plan_datetime']].drop_duplicates()

    # Convert start_time to datetime
    unique_locations['departure_plan_datetime'] = pd.to_datetime(unique_locations['departure_plan_datetime'])

    # Iterate over unique combinations
    for _, row in unique_locations.iterrows():
        location = Point(row['lat'], row['long'])
        start = row['departure_plan_datetime']
        end = start + timedelta(hours=1)

        try:
            # Fetch hourly weather data
            hourly_data = Hourly(location, start, end).fetch()

            if not hourly_data.empty:
                # Calculate temperature average and rain status
                temp_avg = hourly_data['temp'].mean()
                rained = (hourly_data['prcp'] > 0).any()
                weather_cache[(row['lat'], row['long'], row['departure_plan_datetime'])] = (temp_avg, rained)
            else:
                weather_cache[(row['lat'], row['long'], row['departure_plan_datetime'])] = (None, None)
        except Exception as e:
            print(f"Error fetching data for {row}: {e}")
            weather_cache[(row['lat'], row['long'], row['departure_plan_datetime'])] = (None, None)

    return weather_cache


def add_weather_data(df):
    """
    Add temperature and rain status columns to the DataFrame using batched weather data.
    """
    # Fetch weather data for unique combinations
    weather_cache = fetch_weather_batch(df)

    # Map weather data back to the original DataFrame
    df[['temperature', 'rained']] = df.apply(
        lambda row: pd.Series(
            weather_cache.get((row['lat'], row['long'], row['departure_plan_datetime']), (None, None))
        ),
        axis=1
    )

    return df


def save_df(df, output):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved.
    output (str): The path to the output CSV file.
    """
    # Directory to save the file
    save_directory = "../../data/cleaned"
    os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

    # File name and path
    csv_file_name = output
    output = os.path.join(save_directory, csv_file_name)
    try:
        df.to_csv(output, index=False)
        print(f"DataFrame successfully saved to {output}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")


def split_and_save_data(df, train_data_file, test_data_file, activation_file):
    """
    Splits a dataset into training and test datasets, and saves them to CSV files.
    Additionally, creates a CSV file with one entry from the test dataset.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    output_dir (str): The directory where the CSV files will be saved.
    """
    # Directory to save the file
    save_directory = "../../data/cleaned"
    os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

    try:
        # Split the dataset into training (80%) and test (20%) datasets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # Save training data to CSV
        train_path = os.path.join(save_directory, train_data_file)
        train_df.to_csv(train_path, index=False)
        print(f"Training data saved to {train_path}")

        # Save test data to CSV
        test_path = os.path.join(save_directory, test_data_file)
        test_df.to_csv(test_path, index=False)
        print(f"Test data saved to {test_path}")

        # Save one entry from the test data to a separate CSV
        single_entry_path = os.path.join(save_directory, activation_file)
        test_df.iloc[:1].to_csv(single_entry_path, index=False)
        print(f"Single test entry saved to {single_entry_path}")

    except Exception as e:
        print(f"Error processing the data: {e}")


if __name__ == '__main__':
    df_db = add_weather_data(df_db)
    save_df(df_db, 'joint_data_collection.csv')
    split_and_save_data(df_db, "training_data.csv", "test_data.csv",
                        "activation_data.csv")
