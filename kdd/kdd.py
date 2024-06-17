import os
import requests
from bs4 import BeautifulSoup

def download_page(url, save_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_content = soup.get_text()

    filename = os.path.join(save_dir, 'page.html')
    
    print(page_content)
    #with open(filename, 'w', encoding='utf-8') as file:
    #    file.write(page_content)

url = 'https://www.marketwatch.com/investing/stock/msft'
save_dir = 'web_pages'
os.makedirs(save_dir, exist_ok=True)
download_page(url, save_dir)


import requests
from llama_index.core import Document
from llama_index.core.node_parser import HTMLNodeParser

url ='/Users/dako22/Downloads/MSFT.html'

LOCAL=True
if not LOCAL:
    # Send a GET request to the URL
    response = requests.get(url)
    print(response)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the HTML content from the response
        html_doc = response.text
        document = Document(id_=url, text=html_doc)
        # Initialize the HTMLNodeParser with optional list of tags
        parser = HTMLNodeParser(tags=["p", "h1", "h2", "h3", "h4", "h5"])
        # Parse nodes from the HTML document
        nodes = parser.get_nodes_from_documents([document])
        # Print the parsed nodes
        print(nodes)
    else:
        # Print an error message if the request was unsuccessful
        print("Failed to fetch HTML content:", response.status_code)
else:
    file_path = url
    with open(file_path, 'r', encoding='utf-8') as file:
        html_doc = file.read()
    document = Document(id_=url, text=html_doc)
    # Initialize the HTMLNodeParser with optional list of tags
    parser = HTMLNodeParser(tags=["p", "h1", "h2", "h3", "h4", "h5"])
    # Parse nodes from the HTML document
    nodes = parser.get_nodes_from_documents([document])
    # Print the parsed nodes
    print(nodes)

    print("\n\n\n\n\n\n")
    soup = BeautifulSoup(html_doc, 'html.parser')
    page_content = soup.get_text()
    print(page_content)
