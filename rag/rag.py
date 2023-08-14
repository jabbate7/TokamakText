import os
import chromadb
from llm_interface import LLMInterface, OpenAIInterface, HuggingFaceInterface

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

def retrieve(question, model: LLMInterface, n_results, max_result_length=1500):
    print(f'initial question: {question}')
    query_text = model.query(QUERY_SYSTEM_PROMPT, question)
    print(f'query text: {query_text}')
    collection = client.get_collection(collection_name)
    qr = collection.query(query_texts=question, n_results=n_results)
    ids = qr['ids'][0]
    documents = qr['documents'][0]
    # change this into a dict or something
    res = {k: v for k, v in zip(ids, documents) if len(v) < max_result_length}
    return res

def format_results(results):
    formatted_results = f""
    for k, v in results.items():
        formatted_results += f"Shot {k}:\n {v}\n"
    return formatted_results

def rag_answer_question(question, results, model: LLMInterface):
    formatted_results = format_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=formatted_results)
    return model.query(SYSTEM_PROMPT, formatted_user_prompt)

def response_to_thinking_and_answer(generated_response):
    split_response = generated_response.split("ANSWER:")
    display_thinking = split_response[0].strip()
    if len(split_response) == 1:
        display_answer = ""
    elif len(split_response) == 2:
        display_answer = split_response[1].strip()
    else:
        raise ValueError(f"Invalid response: {generated_response}")
    return display_thinking, display_answer

def rag_query(question, model: LLMInterface, n_logs_per_pass, n_pass = 1):
    retrieve_results = retrieve(question, model, n_results=n_logs_per_pass * n_pass)
    log_batches = []
    thinking_batches = []
    answer_batches = []
    for i in range(n_pass):
        # Get the batch of logs for this pass.
        start_idx = i*n_logs_per_pass
        end_idx = (i+1)*n_logs_per_pass
        batch_retrieves = dict(list(retrieve_results.items())[start_idx:end_idx])

        # Answer the question for this batch.
        response = rag_answer_question(question, batch_retrieves, model)

        # Parse and store results.
        batch_thinking, batch_answer = response_to_thinking_and_answer(response)
        log_batches.append(format_results(batch_retrieves)) # TODO: this is redundant..
        thinking_batches.append(batch_thinking)
        answer_batches.append(batch_answer)
    
    if n_pass == 1:
        return log_batches[0], thinking_batches[0], answer_batches[0]
    else:
        return log_batches, thinking_batches, answer_batches

def test():
    question = "Tell me about shots that struggled with tearing modes"
    model = OpenAIInterface(model_name="gpt-3.5-turbo")
    logs, thinking, answer = rag_query(question, model, n_logs_per_pass=5)
    print(f"Model {model.model_name} thinking: {thinking}\n answer: {answer}")

    model2 = HuggingFaceInterface()
    logs, thinking2, answer2 = rag_query(question, model, n_logs_per_pass=5)
    print(f"Model {model2.model_name} thinking: {thinking2}\n answer: {answer2}")


def test_multipass():
    question = "Tell me if these shots have a diagnostic or sensor problem."
    model = OpenAIInterface(model_name="gpt-3.5-turbo")
    log_batches, thinking_batches, answer_batches = rag_query(question, model, n_pass=10, n_logs_per_pass=5)
if __name__ == '__main__':
    test_multipass()