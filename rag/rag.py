import os
import chromadb
from llm_interface import LLMInterface, OpenAIInterface

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()




with open('prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read()
with open('prompts/user_prompt.txt', 'r') as f:
    USER_PROMPT = f.read()
with open('prompts/query_system_prompt.txt', 'r') as f:
    QUERY_SYSTEM_PROMPT = f.read()

client = chromadb.PersistentClient(path="db/")
print(f"{client.list_collections()}")
collection_name = "test_embeddings"

def retrieve(question):
    print(f'initial question: {question}')
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

def rag_answer_question(question, results, model: LLMInterface):
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    return model.query(SYSTEM_PROMPT, formatted_user_prompt)

def test():
    question = "What should I do if we are getting tearing modes early in the shot?"
    model = OpenAIInterface(model_name="gpt-3.5-turbo")
    results = retrieve(question)
    answer = rag_answer_question(question, results, model)
    print(answer)



if __name__ == '__main__':
    test()
