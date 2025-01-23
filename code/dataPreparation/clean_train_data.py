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


def clean_and_preprocess(df):
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
    delay_cols = ['arrival_delay_m', 'departure_delay_m']
    for col in delay_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df[col] = df[col].clip(lower_bound, upper_bound)
    
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
