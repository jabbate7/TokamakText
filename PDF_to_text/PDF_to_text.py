from pdf2docx import Converter

def convert_pdf_to_word(pdf_path, docx_path):
    # Create a Converter instance
    cv = Converter(pdf_path)
    
    # Convert the PDF to a Word document
    cv.convert(docx_path, start=0, end=None)
    
    # Close the converter and release its resources
    cv.close()

if __name__ == '__main__':
    '''
    input_pdf_path = input("Enter the path to the PDF file: ")
    output_docx_path = input("Enter the desired path for the output .docx file: ")
    '''
    
    input_pdf_path='./042503_1_online.pdf'
    output_docx_path='./042503_1_online.docx'

    convert_pdf_to_word(input_pdf_path, output_docx_path)
    print(f"Conversion complete! The Word document is saved at {output_docx_path}.")
