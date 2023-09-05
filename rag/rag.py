import os
import chromadb
from chromadb.utils import embedding_functions
import openai
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def get_chat_completion(system_message, user_message, model="gpt-3.5-turbo-16k"):
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
    query_text = get_chat_completion(QUERY_SYSTEM_PROMPT, question)
    print(f'query text: {query_text}')
    collection = client.get_collection(collection_name, embedding_function=openai_ef)
    qr = collection.query(query_texts=query_text, n_results=10)
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

def rag_answer_question(question, results):
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    return get_chat_completion(SYSTEM_PROMPT, formatted_user_prompt, model='gpt-3.5-turbo-16k')


def test():
    question = "What should I do if we are getting tearing modes early in the shot?"
    results = retrieve(question)
    answer = rag_answer_question(question, results)
    print(answer)



if __name__ == '__main__':
    test()
