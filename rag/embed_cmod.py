from dotenv import load_dotenv
load_dotenv()
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import pickle
import openai
import os
import time
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_type = "azure"
# openai.api_base = "https://test-oai69420.openai.azure.com/"
# openai.api_version = "2023-05-15"

def process_text_data(data, topics_exclude=['rf_monitor'], max_string_length=4*512):
    # Exclude rf_monitor as it tends to have a lot of raw data for whatever reason.
    new_data = dict()
    shots = list(data.keys())
    subset_data = {k: data[k] for k in shots}
    for shot, shot_data in tqdm(subset_data.items()):
        shot_header = f"CMOD Shot Number: {shot}\n"
        for i, entry in enumerate(shot_data):
            if entry['topic'] in topics_exclude:
                continue
            entry_key = f"{shot}_{i}"
            entry_string = f"{shot_header}ENTRY TOPIC: {entry['topic']}\nENTRY USER: {entry['user']}\nENTRY TEXT:\n{entry['text']}\n"
        
            if len(entry_string) <= max_string_length:
                new_data[entry_key] = entry_string
    return new_data

def read_file(pickle_file):
    # laod the pickle file:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def batch_dictionary(dictionary, batch_size):
    keys = list(dictionary.keys())
    batches = []

    for i in range(0, len(keys), batch_size):
        batch = {key: dictionary[key] for key in keys[i:i+batch_size]}
        batches.append(batch)

    return batches


def main(targt):            
    data = read_file(targt)
    processed_text_data = process_text_data(data)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai.api_key,
                model_name="text-embedding-ada-002"
            )

    client = chromadb.PersistentClient(path="/home/awang/chatcmod_entrywise_db")
    collection = client.get_or_create_collection("cmod_text-embedding-ada-002", embedding_function=openai_ef)

    batches = batch_dictionary(processed_text_data, 1500)
    for batch in tqdm(batches):
        keys = list(batch.keys())
        values = list(batch.values())
        collection.add(ids = keys, documents = values)
        print(f'added batch of {len(keys)} documents to collection')
        time.sleep(60) # Wait a minute to not exceed rate limits.
    # print(f'embeddings: {embeddings}')

if __name__ == '__main__':
    pickle_file = "/home/awang/logbook.pkl"
    main(pickle_file)