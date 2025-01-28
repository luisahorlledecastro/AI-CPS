import pandas as pd
import os

# Create cleaned directory if it doesn't exist
save_directory = "../../data/cleaned"
os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

# Read the CSV file
df = pd.read_csv('../../data/scraped/scraped_data.csv')

# Print column names to see what we're working with - commented out after checking
# print("Available columns:")
# print(df.columns.tolist())

# Display first few rows to understand the data structure - commented out after checking
# print("\nFirst few rows of data:")
# print(df.head())

import pandas as pd


def filter_iqr(df, target, lower_bound=0.25, upper_bound=0.75):
    """
    Filters the rows of a dataframe to keep only the data within slected bounds for a specific column.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data.
        column (str): The column name to filter based on the IQR.

    Returns:
        pd.DataFrame: A filtered dataframe with rows within the IQR range for the specified column.
    """
    q1 = df[target].quantile(lower_bound)
    q3 = df[target].quantile(upper_bound)
    iqr = q3 - q1  # Interquartile range

    # Filter rows within the IQR
    filtered_df = df[(df[target] >= q1) & (df[target] <= q3)]

    return filtered_df


def clean_and_preprocess(df, target='arrival_delay_m'):
    # Make a copy of the dataframe
    df = df.copy()
    
    # Convert departure_plan_datetime to datetime
    df['departure_plan_datetime'] = pd.to_datetime(df['departure_plan_datetime'])
    
    # Convert numeric columns
    numeric_cols = ['category', 'long', 'lat', 'arrival_delay_m', 
                   'departure_delay_m', 'departure_plan_hour']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with all missing values
    df = df.dropna(how='all')
    
    # Handle extreme outliers in delay columns
    df = filter_iqr(df, target, lower_bound=0.25, upper_bound=0.75)
    
    # Add useful features
    df['date'] = df['departure_plan_datetime'].dt.date
    df['hour'] = df['departure_plan_datetime'].dt.hour
    df['day_of_week'] = df['departure_plan_datetime'].dt.day_of_week
    df['arrival_delay_m'] = round(df['arrival_delay_m'])
    
    return df


# Apply cleaning and preprocessing
cleaned_df = clean_and_preprocess(df)

# Save the cleaned dataset
cleaned_df.to_csv('../../data/cleaned/cleaned_data.csv', index=False)

print("Data cleaning completed. File saved as '../../data/cleaned/cleaned_data.csv'")

#  TODO add data source to README.learningBase.md
