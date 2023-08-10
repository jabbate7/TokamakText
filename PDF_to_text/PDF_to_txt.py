
import PyPDF2

def extract_text_from_pdf(pdf_path):


    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':

    
    input_pdf_path = './042503_1_online.pdf'
    output_txt_path = './042503_1_online.txt'
    '''
    input_pdf_path = input("Enter the path to the PDF file: ")
    output_txt_path = input("Enter the desired path for the output .txt file: ")
    '''

    text_content = extract_text_from_pdf(input_pdf_path)
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)
