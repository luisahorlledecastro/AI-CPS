# load libraries
import pandas as pd

# import modules
from clean_train_data import clean_and_preprocess
from match_state_to_station import add_states
from merge_and_split_data import add_weather_data, split_and_save_data


def prep_data():
    # step 1
    # load data
    df_trains = pd.read_csv('../../data/scraped/scraped_data.csv')
    df_city = pd.read_csv('../../data/cleaned/station_city_match.csv')

    # step 2
    # clean and preprocess data (trains)
    df_trains = clean_and_preprocess(df_trains)

    # step 3
    # add state data
    df = pd.merge(df_trains, df_city, on='station', how='left').dropna(how='any')

    # step 4
    # add weather data
    df = add_weather_data(df_trains)

    # step 5
    # split and save files
    #split_and_save_data(df, "training_data.csv", "test_data.csv",
                     #   "activation_data.csv")

    return df


if __name__ == '__main__':
    print(prep_data())
