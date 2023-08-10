from dotenv import load_dotenv
load_dotenv()
import sys
import h5py
# from embedding.llm_utils import embed_sentences
import chromadb


def read_file(file_name):
    data = h5py.File(file_name, 'r')
    return data


def process_text_data(text_data):
    new_data = {}
    for k, values in text_data.items():
        strs = []
        for i in range(len(values['text'])):
            text = values['text'][i].decode('utf-8')
            topic = values['topic'][i].decode('utf-8')
            strs.append(f"{text}: {topic}")
        new_data[k] = "\n".join(strs)
    return new_data


def main(targt):
    data = read_file(targt)
    text_data = {}
    for k in data:
        if k in ('spatial_coordinates', 'times'):
            continue
        vals = data[k]
        run = vals['run_sql'][()]
        text = vals['text_sql'][:]
        # summary = vals['summary_sql']
        topic = vals['topic_sql'][:]
        username = vals['username_sql'][:]
        text_data[k] = {
                'run': run,
                'text': text,
                'topic': topic,
                'username': username,
                }

    data.close()
    processed_text_data = process_text_data(text_data)
    sns = processed_text_data.keys()
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("test_embeddings")
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids = keys,
                   documents = values)
    # print(f'embeddings: {embeddings}')


if __name__ == '__main__':
    main(sys.argv[1])
