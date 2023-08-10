import PyPDF2
import os
from tqdm import tqdm
import shutil

def find_pdfs_in_directory(directory):
    pdf_files = []

    # Walk through directory
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(foldername, filename)
                pdf_files.append(full_path)

    return pdf_files

def extract_text_from_pdf(pdf_path):


    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def pdf_to_txt(input_pdf_path = './042503_1_online.pdf',output_txt_path = './042503_1_online.txt'):

    text_content = extract_text_from_pdf(input_pdf_path)
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)
    return 0

def move_txt_files(source_directory, target_directory):
    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Walk through source directory
    for foldername, subfolders, filenames in os.walk(source_directory):
        for filename in filenames:
            if filename.lower().endswith('.txt'):
                source_path = os.path.join(foldername, filename)
                target_path = os.path.join(target_directory, filename)
                
                # Rename the file if it already exists in the target directory
                counter = 1
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(filename)
                    target_path = os.path.join(target_directory, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(source_path, target_path)

if __name__ == "__main__":
    directory_to_search='./../../tmp/PDFs'
    target_dir='./../../tmp/TXTs'

    source_dir=directory_to_search
    '''
    directory_to_search = input("Enter the path to the directory to search: ")
    '''
    
    pdf_files = find_pdfs_in_directory(directory_to_search)
    
    for pdf in tqdm(pdf_files):
        print(pdf)
        pdf_to_txt(input_pdf_path = pdf,\
                output_txt_path = pdf[:-3]+'txt')

    

    move_txt_files(source_dir, target_dir)
    print("Files moved successfully!")