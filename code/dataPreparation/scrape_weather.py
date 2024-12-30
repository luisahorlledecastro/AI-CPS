import os
import requests
from bs4 import BeautifulSoup

# URL of the directory to scrape
url = "https://opendata.dwd.de/climate_environment/CDC/event_catalogues/germany/precipitation/CatRaRE_v2024.01/data/"

# Directory to save the file
save_directory = "data/"
os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

# File name and path
csv_file_name = "scraped_weather.csv"
save_path = os.path.join(save_directory, csv_file_name)

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Print all available links
    print("Available links:")
    for link in soup.find_all('a', href=True):
        print(link['href'])

    # Search for the specific CSV file
    csv_link = None
    for link in soup.find_all('a', href=True):
        if link['href'].endswith('.csv'):  # Adjust to find CSV files
            csv_link = link['href']
            break

    if csv_link:
        # Construct the full URL for the CSV file
        full_csv_url = url + csv_link

        # Download the CSV file
        csv_response = requests.get(full_csv_url)

        if csv_response.status_code == 200:
            # Save the file in the desired directory
            with open(save_path, 'wb') as file:
                file.write(csv_response.content)
            print(f"File downloaded and saved successfully: {save_path}")
        else:
            print(f"Failed to download the CSV file. Status code: {csv_response.status_code}")
    else:
        print("No CSV file found in the directory.")
else:
    print(f"Failed to access the website. Status code: {response.status_code}")
