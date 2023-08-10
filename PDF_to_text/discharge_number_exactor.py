import os
import re
import csv
import argparse


def extract_numbers_near_string(file_path):
    """Extract numbers following certain patterns from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # Pattern to capture numbers immediately following specific patterns
        pattern = r'(DIII-D discharge|DIII-D shot|DIII-D|DIII-D shot number|DIII-D discharge number|DIIID discharge|DIIID shot|DIIID|DIIID shot number|DIIID discharge number)\s*([12]\d{5}|\b\d{5}\b)'

        # Find the numbers based on the pattern
        matches = re.findall(pattern, content)

        # Extract only the numbers from the matched pairs
        numbers = [match[1] for match in matches]

        return numbers




def main(directory_to_search, csv_path):
    """Main function to extract 6-digit numbers and save to CSV."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File Name", "6-digit Numbers"])

        for foldername, _, filenames in os.walk(directory_to_search):
            for filename in filenames:
                if filename.lower().endswith('.txt'):
                    txt_path = os.path.join(foldername, filename)
                    numbers = extract_numbers_near_string(txt_path)

                    if numbers:
                        writer.writerow([filename, ' '.join(numbers)])


if __name__ == "__main__":
    '''
    parser = argparse.ArgumentParser(description="Extract 6-digit numbers from TXT files to CSV.")
    parser.add_argument('--source', default='./../../tmp/TXTs', help='Directory to search for TXT files')
    parser.add_argument('--csv', default='./output.csv', help='Path to save the CSV file')

    args = parser.parse_args()
    '''
    
    source_path='./../../tmp/TXTs'
    csv_file='1.csv'
    main(source_path, csv_file)
