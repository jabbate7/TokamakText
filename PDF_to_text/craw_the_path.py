import os

def find_pdfs_in_directory(directory):
    pdf_files = []

    # Walk through directory
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(foldername, filename)
                pdf_files.append(full_path)

    return pdf_files

if __name__ == "__main__":
    directory_to_search = input("Enter the path to the directory to search: ")
    pdf_files = find_pdfs_in_directory(directory_to_search)

    for pdf in pdf_files:
        print(pdf)
