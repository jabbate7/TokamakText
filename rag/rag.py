import os
import chromadb
# import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

with open('prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read()
with open('prompts/user_prompt.txt', 'r') as f:
    USER_PROMPT = f.read()
with open('prompts/query_system_prompt.txt', 'r') as f:
    QUERY_SYSTEM_PROMPT = f.read()

client = chromadb.PersistentClient(path="db/")
print(f"{client.list_collections()}")
collection_name = "test_embeddings"

class Model:
    def __init__(self,
                 model_name = "upstage/llama-30b-instruct-2048",
                 cache_dir="/nobackup1/allenw/Scratch/") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            cache_dir=cache_dir,
        )
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def query(self, system, user_input):
        inputs = self.generate_input(system, user_input)
        output = self.model.generate(**inputs, streamer=self.streamer, use_cache=True, max_new_tokens=float('inf'))
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

    def generate_input(self, system, user_input):
        input_string = f"### System:\n{system}\n### User:\n{user_input}\n### Assistant:\n"
        inputs = self.tokenizer(input_string, return_tensors="pt").to(self.model.device)
        del inputs["token_type_ids"]
        return inputs

hf_model = Model()

def get_chat_completion_huggingface(system_message, user_message):
    res = hf_model.query(system_message, user_message)
    return res

# def get_chat_completion(system_message, user_message, model="gpt-3.5-turbo"):
#     completion = openai.ChatCompletion.create(
#       model=model,
#       messages=[
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": user_message}
#       ]
#     )

#     return completion.choices[0].message.content

def retrieve(question):
    print(f'initial question: {question}')
    query_text = get_chat_completion_huggingface(QUERY_SYSTEM_PROMPT, question)
    print(f'query text: {query_text}')
    collection = client.get_collection(collection_name)
    qr = collection.query(query_texts=question, n_results=5)
    ids = qr['ids'][0]
    documents = qr['documents'][0]
    # change this into a dict or something
    res = {k: v for k, v in zip(ids, documents)}
    return res

def process_results(results):
    processed_results = f""
    for k, v in results.items():
        processed_results = processed_results + f"{k}: {v}\n"
    return processed_results

def rag_answer_question(question, results):
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    return get_chat_completion(SYSTEM_PROMPT, formatted_user_prompt, model='gpt-4')


def test():
    question = "What should I do if we are getting tearing modes early in the shot?"
    results = retrieve(question)
    answer = rag_answer_question(question, results)
    print(answer)



if __name__ == '__main__':
    test()
