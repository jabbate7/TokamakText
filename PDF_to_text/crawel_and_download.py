from selenium import webdriver
from selenium.webdriver.common.by import By
import time
#import wget
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://iopscience.iop.org/'
JOURNAL_URL = BASE_URL + 'journal/volume/issue'  # Ensure this is the correct URL

# Start a new browser session


def fetch_pdf_links(journal_url):
    response = requests.get(journal_url)
    
    if response.status_code != 200:
        print(f"Failed to fetch {journal_url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting all anchor tags
    links = soup.find_all('a', href=True)

    # Filtering for the specific pattern of the PDF URLs
    pdf_links = [link['href'] for link in links if 'article' in link['href'] and link['href'].endswith('/pdf')]
    return pdf_links
'''
driver = webdriver.Chrome()
def fetch_pdf_links(journal_url):
    driver.get(journal_url)
    time.sleep(5)  # Wait for the page to load, can be adjusted
    
    # Find links with the specific pattern
    pdf_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/article/') and contains(@href, '/pdf')]")
    pdf_links = [element.get_attribute('href') for element in pdf_elements]
    return pdf_links

'''

def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)

def download_pdfs(pdf_links, download_folder="."):
    for link in pdf_links:
        file_name = link.split("/")[-2] + ".pdf"
        destination = f"{download_folder}/{file_name}"
        #wget.download(link, destination)

        print('link')
        print(link)
        download_file(link, destination)

pdf_links = fetch_pdf_links(JOURNAL_URL)
print(pdf_links)
download_pdfs(pdf_links,'./NF')

#driver.quit()  # Close the browser session
