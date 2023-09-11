import os
import chromadb
from chromadb.utils import embedding_functions
import openai
from text_helpers import document_info
from llm_interface import LLMInterface, get_llm_interface

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

with open('prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read()
with open('prompts/user_prompt.txt', 'r') as f:
    USER_PROMPT = f.read()
with open('prompts/query_system_prompt.txt', 'r') as f:
    QUERY_SYSTEM_PROMPT = f.read()

client = chromadb.PersistentClient(path="/home/awang/chatcmod_db/")
print(f"{client.list_collections()=}")
collection_name = "cmod_text-embedding-ada-002"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model_name="text-embedding-ada-002"
        )

def get_chat_completion(system_message, user_message, model="gpt-4"):
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
      ]
    )

    return completion.choices[0].message.content

def retrieve(question):
    print(f'initial question: {question}')
    collection = client.get_collection(collection_name, embedding_function=openai_ef)
    qr = collection.query(query_texts=question, n_results=10)
    ids = qr['ids'][0]
    documents = qr['documents'][0]
    # change this into a dict or something
    res = {k: v for k, v in zip(ids, documents)}
    return res

def process_results(results):
    processed_results = f""
    for k, v in results.items():
        processed_results = processed_results + f"{k}: {v}\n"
    print(len(processed_results))
    return processed_results

def rag_answer_question(question, results, model: LLMInterface):
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    return get_chat_completion(SYSTEM_PROMPT, formatted_user_prompt, model='gpt-4')


def test():
    question = "Tell me about shots that struggled with tearing modes"
    model = get_llm_interface("openai")
    results = retrieve(question)
    answer = rag_answer_question(question, results, model)
    print(f"Model {model.model_name} answer:\n{answer}")

if __name__ == '__main__':
    test()
