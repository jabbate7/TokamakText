import os
import shutil

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
    source_dir='./../../tmp/PDFs'
    target_dir='./../../tmp/TXTs'
    '''
    source_dir = input("Enter the path to the directory to search for .txt files: ")
    target_dir = input("Enter the path to the target directory where .txt files should be moved: ")
    
    '''
    
    
    move_txt_files(source_dir, target_dir)
    print("Files moved successfully!")
