import torch
import pickle
import h5py
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_model_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model


def get_embedding(model, text):
    inputs = tokenizer("Your input text here", return_tensors="pt", padding=True)
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_ids)
        sequence_embedding = encoder_outputs.last_hidden_state
    return sequence_embedding[0].mean(dim=0)

def process_text(text_data):
    text = ""
    text += f"{text_data['text']}\n"
    text += f"{text_data['run_text']}\n"
    return text



def embed_all_shots(shots_path, run_path):
    data = h5py.File(shots_path, 'r')
    tokenizer, model = get_model_tokenizer()
    run_dump_f = h5py.File(run_path, 'r')
    embeddings = {}
    for key in tqdm(data.keys()):
        if key in ('spatial_coordinates', 'times'):
            continue
        vals = data[key]
        shot_info = {}
        run = shot_info['run'] = vals['run_sql'][()]
        shot_info['text'] = vals['text_sql'][:]
        # summary = vals['summary_sql']
        shot_info['topic'] = vals['topic_sql'][:]
        shot_info['username'] = vals['username_sql'][:]
        run_info = run_dump_f[run]
        shot_info['run_brief'] = run_info['brief'][()]
        shot_info['run_mp'] = run_info['miniproposal'][()]
        shot_info['run_text'] = run_info['text'][()]
        shot_info['run_topic'] = run_info['topic'][()]
        shot_info['run_username'] = run_info['username'][()]
        text = process_text(shot_info)
        sequence_embedding = get_embedding(model, text)
        embeddings[key] = sequence_embedding
    data.close()
    run_dump_f.close()
    return embeddings

if __name__ == "__main__":
    tokenizer, model = get_model_tokenizer()
    text = 'lol I am dumb'
    sequence_embedding = get_embedding(model, text)
    print(f"{text=}")
    print(f"{sequence_embedding=}")

    embeddings = embed_all_shots('../data/example_194528.h5', '../data/run_dump.h5')
    with open('text_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
