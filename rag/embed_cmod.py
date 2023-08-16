from dotenv import load_dotenv
load_dotenv()
import sys
# from embedding.llm_utils import embed_sentences
import chromadb
from tqdm import tqdm
import pickle
import itertools

def process_text_data(data):
    new_data = dict()
    shots = list(data.keys())
    subset_data = {k: data[k] for k in shots[:5]}
    for shot, shot_data in subset_data.items():
        print(f"shot: {shot}")
        shot_text = "Shot: {shot}\n"
        for entry in shot_data:
            entry_string = f"entry topic: {entry['topic']}\nentry user: {entry['user']}\nentry text:"
            shot_text += "\n" + entry_string
        new_data[shot] = shot_text
    return new_data

def read_file(pickle_file):
    # laod the pickle file:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def main(targt):
    data = read_file(targt)
    processed_text_data = process_text_data(data)
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("cmod_embeddings")
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids = keys,
                   documents = values)
    # print(f'embeddings: {embeddings}')

if __name__ == '__main__':
    pickle_file = "/nobackup1/allenw/Scratch/logbook.pkl"
    main(pickle_file)