import os
import re
import requests
from bs4 import BeautifulSoup

def download_pdf(url, folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        filepath = os.path.join(folder, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Download concluído: {filename}")
    else:
        print(f"Erro ao baixar {url}: {response.status_code}")


base_url = "https://portal.inmet.gov.br/boletinsagro"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(base_url, headers=headers)

if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")

    output_folder = "boletins_pdfs"
    os.makedirs(output_folder, exist_ok=True)

    pdf_pattern = re.compile(r"pdfcall\('([^']+\.pdf)'\)")

    pdf_links = pdf_pattern.findall(html_content)

    for pdf_url in pdf_links:
        download_pdf(pdf_url, output_folder)

    print("Todos os PDFs foram baixados.")
else:
    print(f"Erro ao acessar a página: {response.status_code}")
