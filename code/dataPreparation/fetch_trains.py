import pandas as pd
import requests

def scrape_csv_data():
    url = "https://raw.githubusercontent.com/Nokkyuu/Fantastic_trains_and_when_to_find_them/refs/heads/main/data/heatmap.csv"
    
    try:
        # Fetch the CSV data from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(url)
        
        # Save the data to a local CSV file
        df.to_csv('./data/scraped_data.csv', index=False)
        
        print("Data successfully scraped and saved to 'scraped_heatmap_data.csv'")
        print(f"Shape of the data: {df.shape}")
        print("\nFirst few rows of the data:")
        print(df.head())
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    scrape_csv_data()
