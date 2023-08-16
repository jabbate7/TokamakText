import sys
import h5py
import chromadb

def read_file(file_name):
    data = h5.File(file_name, 'r')
    return data


def process_text_data(text_data):
    new_data = {}
    for key in text_data.keys():
        for i in len(text_data[key]['text']):
            new_data[key+f'_{i}'] = text_data[key]['text'][i]
            
    return new_data

def main(targt):
    data = read_file(targt)

    text_data = get_summaries(data)
    data.close()
    
    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection('test_embeddings')
    keys = list(text_data.keys())
    values = list(text_data.values())
    collection.add(ids = keys, documents = values)


if __name__ == '__main__':
    main(sys.argv[1])
