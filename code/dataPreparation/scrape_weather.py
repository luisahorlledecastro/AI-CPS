import os
import requests
from bs4 import BeautifulSoup

# URL of the directory to scrape
url = "https://opendata.dwd.de/climate_environment/CDC/event_catalogues/germany/precipitation/CatRaRE_v2024.01/data/"

# Directory to save the file
save_directory = "data/scraped"
os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

# File name and path
csv_file_name = "scraped_weather.csv"
save_path = os.path.join(save_directory, csv_file_name)


def fetch_page_content(url):
    """Fetches the HTML content of the given URL.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        requests.Response: The response object containing the HTML content.
    """
    return requests.get(url)


def extract_links(html_content):
    """Extracts all links from the given HTML content.

    Args:
        html_content (str): The HTML content of the page.

    Returns:
        list: A list of links found in the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    return [link['href'] for link in soup.find_all('a', href=True)]


def find_csv_link(links):
    """Finds the first CSV file link in a list of links.

    Args:
        links (list): A list of hyperlinks extracted from the webpage.

    Returns:
        str or None: The first CSV file link found, or None if no CSV is found.
    """
    for link in links:
        if link.endswith('.csv'):
            return link
    return None


def download_csv_file(csv_url, save_path):
    """Downloads a CSV file from the given URL and saves it to the specified path.

    Args:
        csv_url (str): The URL of the CSV file to download.
        save_path (str): The local file path to save the downloaded CSV file.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    response = requests.get(csv_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    return False


# Fetch the page content
response = fetch_page_content(url)

if response.status_code == 200:
    # Extract all links from the page
    links = extract_links(response.text)
    print("Available links:")
    for link in links:
        print(link)

    # Find the first CSV link
    csv_link = find_csv_link(links)

    if csv_link:
        full_csv_url = url + csv_link  # Construct full URL

        if download_csv_file(full_csv_url, save_path):
            print(f"File downloaded and saved successfully: {save_path}")
        else:
            print("Failed to download the CSV file.")
    else:
        print("No CSV file found in the directory.")
else:
    print(f"Failed to access the website. Status code: {response.status_code}")
