from flask import Flask, render_template_string, request
from rag import retrieve, rag_answer_question
from llm_interface import OpenAIInterface, HuggingFaceInterface
import click

def get_llm_interface(model_name):
    if model_name == "openai":
        return OpenAIInterface()
    elif model_name == "huggingface":
        return HuggingFaceInterface()
    else:
        raise ValueError(f"Invalid model name: {model_name}")


@click.command()
@click.option('--model', type=click.Choice(['openai', 'huggingface']), default="openai", help='The model to run.')
def main(model_name):
    """
    Load the language model to use.
    """
    llm_interface = get_llm_interface(model_name)
    
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
            results = retrieve(question, llm_interface)
            retrieved_results = "\n".join([f"{key}: {value}" for key, value in results.items()])
            generated_answer = rag_answer_question(question, results, llm_interface)
            display_results = True

        split_response = generated_answer.split("ANSWER:")
        thinking = split_response[0].strip()
        if len(split_response) == 1:
            answer = ""
        else:
            answer = split_response[1].strip()

        return render_template_string("""
        <form method="post">
            <label for="question">Ask a Question:</label>
            <textarea id="question" name="question" cols="40" rows="5" required></textarea>
            <input type="submit" value="Submit">
        </form>
        {% if display_results %}
            <h3>Retrieved Results:</h3>
            <pre style="white-space: pre-wrap;">{{ retrieved_results }}</pre>
            <h3>Question:</h3>
            <pre style="white-space: pre-wrap;">{{ question }}</pre>
            <h3>Thinking:</h3>
            <pre style="white-space: pre-wrap;">{{ thinking }}</pre>
            <h3>Generated Answer:</h3>
            <pre style="white-space: pre-wrap;">{{ answer }}</pre>
        {% endif %}
        """, question=question, retrieved_results=retrieved_results, thinking=thinking, answer=answer, display_results=display_results)
    
    app.run(debug=True)
if __name__ == "__main__":
    main()