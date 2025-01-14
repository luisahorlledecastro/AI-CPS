

# import libraries
from meteostat import Point, Hourly
import pandas as pd
from datetime import datetime, timedelta


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


# Add temperature and rain status to the DataFrame
df_db = add_weather_data(df_db)

# Display the updated DataFrame
print(df_db.head())




