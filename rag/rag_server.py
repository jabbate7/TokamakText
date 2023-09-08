from flask import Flask, render_template_string, request
from rag import retrieve, rag_answer_question
from llm_interface import get_llm_interface
import click
from dotenv import load_dotenv
import h5py

@click.command()
@click.option('--llm_type', type=click.Choice(['openai', 'huggingface']), default="openai", help='The model to run.')
def main(llm_type):
    """
    Load the language model to use.
    """
    llm_interface = get_llm_interface(llm_type)
    
    """
    Run the app.
    """
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        question = ""
        retrieved_results = ""
        generated_answer = ""
        display_results = False

        if request.method == "POST":
            question = request.form.get("question")
            results = retrieve(question)
            retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
            generated_answer = rag_answer_question(question, results, llm_interface)
            display_results = True

def log_result(dict_to_log, filename="log.h5"):
    # Create or open an HDF5 file in 'a' mode
    with h5py.File(filename, 'a') as f:
        # Create a new group for each dictionary
        new_group_name = 'example_number' + str(len(f.keys()))
        new_group = f.create_group(new_group_name)
        for key, value in dict_to_log.items():
            new_group.create_dataset(key, data=value)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    retrieved_results = ""
    generated_answer = ""
    display_results = False

    if request.method == "POST":
        question = request.form.get("question")
        results = retrieve(question)
        retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
        generated_answer = rag_answer_question(question, results)
        display_results = True

    split_response = generated_answer.split("ANSWER:")
    thinking = split_response[0].strip()
    if len(split_response) == 1:
        answer = ""
    else:
        answer = split_response[1].strip()

    if len(answer) > 0:
        log_dic = {
            "question": question,
            "retrieved_results": retrieved_results,
            "thinking": thinking,
            "answer": answer,
        }
        
        log_result(log_dic)

    return render_template_string("""
    <form method="post">
        <label for="question">Ask a Question:</label>
        <textarea id="question" name="question" cols="40" rows="5" required></textarea>
        <input type="submit" value="Submit">
    </form>
    {% if display_results %}
        <h3>Question:</h3>
        <pre style="white-space: pre-wrap;">{{ question }}</pre>
        <h3>Thinking:</h3>
        <pre style="white-space: pre-wrap;">{{ thinking }}</pre>
        <h3>Generated Answer:</h3>
        <pre style="white-space: pre-wrap;">{{ answer }}</pre>
        <h3>Retrieved Shot Logs:</h3>
        <pre style="white-space: pre-wrap;">{{ retrieved_results }}</pre>
    {% endif %}
    """, question=question, retrieved_results=retrieved_results, thinking=thinking, answer=answer, display_results=display_results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # app.run(debug=True, port=5000)
