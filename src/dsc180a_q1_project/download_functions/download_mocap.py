import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Get the repo root (two levels up from this script)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
datasets_dir = os.path.join(repo_root, "datasets")  # DSC180A-Q1-Project/datasets

def download_files_by_extension(links, base_url, extension, download_dir, download_limit = 1000):
    """
    A helper function to find and download all files with a specific extension.
    
    Args:
        links (list): A list of all <a> tag elements from BeautifulSoup.
        base_url (str): The main URL of the site to join relative paths.
        extension (str): The file extension to look for (e.g., ".bvh").
        download_dir (str): The folder name to save files into.
    """
    print(f"\n--- Starting process for {extension} files ---")
    
    # --- 1. Create the download directory if it doesn't exist ---
    os.makedirs(download_dir, exist_ok=True)
    print(f"Files will be saved in: {download_dir}")

    # --- 2. Find all links matching the extension ---
    target_links = []
    for link in links:
        href = link.get('href')
        
        # Check if the link exists, ends with the extension, AND doesn't start with 'index'
        # to avoid downloading the animation pages
        if href and href.endswith(extension) and not os.path.basename(href).startswith('index'):
            full_url = urljoin(base_url, href)
            target_links.append(full_url)

    if not target_links:
        print(f"No {extension} files found on the page.")
        return

    print(f"Found {len(target_links)} {extension} files. Starting download...")

    # --- 3. Download each file ---
    downloaded_count = 0
    skipped_count = 0

    for file_url in target_links:
        if downloaded_count >= download_limit:
            print(f"Download limit ({download_limit}) reached. Halting further downloads.")
            break
        filename = os.path.basename(file_url)
        save_path = os.path.join(download_dir, filename)

        if os.path.exists(save_path):
            print(f"Skipping {filename} (already exists)")
            skipped_count += 1
            continue

        try:
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(file_response.text)
            
            print(f"Successfully downloaded {filename}")
            downloaded_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}. Error: {e}")

    print(f"\n--- {extension} Download Complete ---")
    print(f"Successfully downloaded: {downloaded_count} files")
    print(f"Skipped (already exist): {skipped_count} files")

def main_downloader(download_limit = 1000):
    base_url = "https://mocap.cs.sfu.ca/"
    
    # --- 1. Fetch the main page content ONCE ---
    print(f"Connecting to {base_url}...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the main page: {e}")
        return

    # --- 2. Parse the HTML to find all links ---
    print("Parsing website for all links...")
    soup = BeautifulSoup(response.text, 'html.parser')
    all_links = soup.find_all('a') # Get all links from the page

    # --- 3. Run the download process for each file type ---
    # download_files_by_extension(all_links, base_url, ".bvh", "bvh_files")
    # download_files_by_extension(all_links, base_url, ".vsk", "vsk_files")
    txt_download_dir = os.path.join(datasets_dir, "txt_files")
    download_files_by_extension(all_links, base_url, ".txt", txt_download_dir, download_limit = download_limit)

    print("\nAll tasks complete.")

# --- Run the main downloader function ---
if __name__ == "__main__":
    main_downloader()

