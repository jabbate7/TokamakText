import torch
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

if __name__ == "__main__":
    tokenizer, model = get_model_tokenizer()
    text = 'lol I am dumb'
    sequence_embedding = get_embedding(model, text)
    print(f"{text=}")
    print(f"{sequence_embedding=}")
