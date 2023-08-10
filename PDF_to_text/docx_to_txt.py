import docx
#pip install python-docx

def docx_to_txt(docx_path):
    # Load the .docx document
    doc = docx.Document(docx_path)
    
    # Extract text from the document
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
        
    return '\n'.join(full_text)

if __name__ == '__main__':
    '''
    input_docx_path = input("Enter the path to the .docx file: ")
    output_txt_path = input("Enter the desired path for the output .txt file: ")
    '''
    
    input_docx_path = './042503_1_online.docx'
    output_txt_path = './042503_1_online.txt'

    # Get text content from .docx
    text_content = docx_to_txt(input_docx_path)

    # Write text content to .txt
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)

    print(f"Conversion complete! The text content is saved at {output_txt_path}.")
