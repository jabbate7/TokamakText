from dotenv import load_dotenv
load_dotenv()
import sys
import h5py
from tqdm import tqdm
# from embedding.llm_utils import embed_sentences
import chromadb
from text_helpers import make_document_dic_from_string

def read_file(file_name):
    data = h5py.File(file_name, 'r')
    return data

def process_text_data(text_data):
    new_data = {}
    for shot in text_data:
        strs = []
        run=text_data[shot]['run'].decode('utf-8')
        for entry_ind in range(len(text_data[shot]['text'])):
            text = text_data[shot]['text'][entry_ind].decode('utf-8')
            topic = text_data[shot]['topic'][entry_ind].decode('utf-8')
            username = text_data[shot]['username'][entry_ind].decode('utf-8')
            if username not in ('pcsops'):
                strs.append(f"user {username} ({topic}): {text}")
        new_data.update(make_document_dic_from_string("\n".join(strs),
                                                      shot,
                                                      'shot'))
    return new_data


def main(targt):
    data = read_file(targt)
    text_data = {}
    for shot in tqdm(data):
        if shot in ('spatial_coordinates', 'times'):
            continue
        vals = data[shot]
        run = vals['run_sql'][()]
        text = vals['text_sql'][:]
        # summary = vals['summary_sql']
        topic = vals['topic_sql'][:]
        username = vals['username_sql'][:]
        text_data[shot] = {
                'run': run,
                'text': text,
                'topic': topic,
                'username': username,
                }

    data.close()
    processed_text_data = process_text_data(text_data)
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("shot_embeddings")
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids = keys,
                   documents = values)
    # print(f'embeddings: {embeddings}')

if __name__ == '__main__':
    main(sys.argv[1])
