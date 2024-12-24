import os

try:
    import docling
except:
    raise Exception("dependencia 'docling' não instalada - instale com 'pip install docling'")

path_pdfs = './boletins_pdfs'
output_folder = './boletins_markdown'

os.makedirs(output_folder,exist_ok=True)

for pdf in os.listdir(path_pdfs):
    print(f'convertendo {pdf}')
    os.system(f"docling {path_pdfs}/{pdf} --output {output_folder}")
