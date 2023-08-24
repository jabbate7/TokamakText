import chromadb
import sys
import os
from text_helpers import make_document_dic_from_string

def read_file(file_name):
    with open(file_name) as f:
        text = f.read()
    return text

# document_type "miniproposal"
def main(text_dir, document_type):
    files = os.listdir(text_dir)
    text_data = {}
    for fn in files:
        text = read_file(os.path.join(text_dir,fn))
        text_data.update(make_document_dic_from_string(text, fn, document_type))

    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection(f'{document_type}_embeddings')
    keys = list(text_data.keys())
    values = list(text_data.values())
    collection.add(ids = keys, documents = values)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
