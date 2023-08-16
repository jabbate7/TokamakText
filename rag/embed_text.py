import chromadb
import sys
import os


def read_file(file_name):
    with open(file_name) as f:
        text = f.read()
    return text

def chunk_text(text, chunking=1700):
    # Average shotlog entry is around 1700 characters
    total_len = len(text)
    ind = 0
    chunks = []

    while ind < total_len:
        chunks.append(text[ind:ind+chunking])
        ind += chunking
        
    return chunks

def main(text_dir):
    files = os.listdir(text_dir)
    text_data = {}
    for fn in files:
        text = read_file(text_dir+fn)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            text_data[fn+f'_{i}'] = chunk

    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection('test_embeddings')
    keys = list(text_data.keys())
    values = list(text_data.values())
    collection.add(ids = keys, documents = values)

    

if __name__ == '__main__':
    main(sys.argv[1])
