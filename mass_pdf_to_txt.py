import PyPDF2
import os
from tqdm import tqdm
import shutil
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def pdf_to_txt(input_pdf_path, output_txt_path):
    text_content = extract_text_from_pdf(input_pdf_path)
    with open(output_txt_path, 'w',) as txt_file:
        txt_file.write(text_content)


if __name__ == "__main__":
    pdf_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    
    pdf_files = os.listdir(pdf_dir)
    
    for pdf in tqdm(pdf_files):
        pdf_to_txt(pdf_dir+f'/{pdf}', txt_dir+f'/{pdf[:-3]}.txt')

    print("Files moved successfully!")
