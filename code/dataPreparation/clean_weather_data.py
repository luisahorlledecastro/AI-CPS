import pandas as pd
import numpy as np

# Read the CSV file from the data folder
df = pd.read_csv('data/scraped/scraped_weather.csv', sep=';')

# Print column names to see what we're working with - commented out after checking
# print("Available columns:")
# print(df.columns.tolist())

# Display first few rows to understand the data structure - commented out after checking
# print("\nFirst few rows of data:")
# print(df.head())

def clean_and_preprocess(df):
    # Make a copy of the dataframe
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

# Apply cleaning and preprocessing
cleaned_df = clean_and_preprocess(df)

# Save cleaned data
cleaned_df.to_csv('data/cleaned/cleaned_weather.csv', index=False)

print("\nCleaning and preprocessing completed. Saved as 'cleaned_weather.csv'")
