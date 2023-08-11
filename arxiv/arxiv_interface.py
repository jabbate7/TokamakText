import arxiv
import requests
import click

import PyPDF2
import os
from tqdm import tqdm

'''
@click.command()
@click.argument('out_dir', type=click.Path())
@click.argument('query', default="tokamak", type=click.STRING)
@click.argument('n_papers', default=10, type=click.INT)
'''


def extract_text_from_pdf(pdf_path):
    print('0')
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    print('1')
    with open(pdf_path[:-3]+'txt', 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

    print('2')

    # remove the file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"{pdf_path} has been removed!")
    else:
        print(f"{pdf_path} does not exist!")
    print('3')
    return 0

def run(out_dir, query, n_papers):
    search = arxiv.Search(
        query=query,
        max_results=n_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )
    for res in tqdm(search.results()):
        path = res.download_pdf(out_dir)
        print(path)
        tmp=extract_text_from_pdf(str(path))
if __name__ == '__main__':
    run(out_dir='./data',query='tokamak', n_papers=2)
