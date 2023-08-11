import requests

def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            out_file.write(chunk)

url = "https://iopscience.iop.org/article/10.1088/1741-4326/ac9b76/pdf"
destination = "./file.pdf"

download_file(url, destination)
