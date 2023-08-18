import sys
import h5py
import chromadb
from tqdm import tqdm
from text_helpers import make_document_dic_from_string

def read_file(file_name):
    data = h5py.File(file_name, 'r')
    return data

def process_text_data(text_data):
    new_data = {}
    for run in text_data:
        strs = []
        brief=text_data[run]['brief'].decode('utf-8')
        for entry_ind in range(len(text_data[run]['text'])):
            text=text_data[run]['text'][entry_ind].decode('utf-8')
            topic=text_data[run]['topic'][entry_ind].decode('utf-8')
            username=text_data[run]['username'][entry_ind].decode('utf-8')
            strs.append(f"{brief}: user {username} ({topic}): {text}")
        new_data.update(make_document_dic_from_string("\n".join(strs),
                                                      run,
                                                      'run'))
    return new_data

def main(targt):
    data = read_file(targt)
    text_data = {}
    for run in tqdm(data):
        vals = data[run]
        brief = vals['brief'][()]
        text = vals['text'][:]
        # summary = vals['summary_sql']
        topic = vals['topic'][:]
        username = vals['username'][:]
        text_data[run] = {
                'brief': brief,
                'text': text,
                'topic': topic,
                'username': username,
                }

    data.close()
    processed_text_data = process_text_data(text_data)
    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection('run_embeddings')
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids = keys, documents = values)


if __name__ == '__main__':
    main(sys.argv[1])
